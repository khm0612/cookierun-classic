"""Self-improving farm — runs the FULL farm flow (Double-Coins boost gate + Head Start +
card-solver via monitor.py + wedge recovery), banking real coins, WHILE recording every run
and periodically retraining the model on its OWN longest-survival runs. Bank coins AND get
smarter over time.

Every retrain is ANCHORED on the human demos (demo2/3/4) so a bad batch can't drift the model
into garbage. Survival is a noisy signal (boosts add tanking + zone luck), so improvement is
gradual, not guaranteed each cycle — watch the trend on W&B, not one cycle.

The default base + retrain arch is the FAST small_cnn (~35fps live). This matters: LearnedAgent
stacks frames at the training fps (1/35s) at inference, so recording must ALSO be ~35fps for the
K-stack to match. A slow arch (mobilenet ~16fps) records temporally-too-wide stacks and every
retrain would DEGRADE the model — so self-training only makes sense on an arch that plays >=35fps.

  python scripts/self_farm.py [cycles] [--arch small_cnn] [--every 15] [--keep 4] [--wandb]

Stop cleanly anytime:  touch data/_selffarm/STOP   (finishes the current run, then exits)

Hardened for UNATTENDED overnight running (adversarial-review fixes):
  - every per-run cycle is exception-guarded (one bad run never ends the night);
  - teardown (monitor kill, retrain kill, device stop) runs in finally, even on fatal error;
  - ensure_running failures ESCALATE to a game restart instead of spinning silently;
  - a hung background retrain is killed on a timeout so hot-swaps + disk-pruning resume;
  - the recent[] buffer + its tmp frame dirs are capped so disk can't fill during a long retrain;
  - degenerate (blind/false-start) runs are dropped, never promoted into the training buffer;
  - the card-solver (monitor.py) is polled + relaunched if it dies.
"""
from __future__ import annotations
import os
import sys
import json
import time
import queue
import shutil
import threading
import traceback
import subprocess

from _runtime import CONFIG, DATA, ROOT
import cv2
from cookierun_bot.config import load_config
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot.device import open_device
from cookierun_bot.policies.learned import LearnedAgent
from cookierun_bot.gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from cookierun_bot import farm
from cookierun_bot.farm_common import _in_run


_PIT_TPL = cv2.imread(str(ROOT / "templates" / "pitlift_norm.png"), cv2.IMREAD_GRAYSCALE)


def _pitfall(frame) -> bool:
    if _PIT_TPL is None:
        return False
    h, w = frame.shape[:2]
    crop = cv2.cvtColor(
        frame[int(h * 0.830):int(h * 0.956), int(w * 0.372):int(w * 0.684)],
        cv2.COLOR_BGR2GRAY,
    )
    crop = cv2.resize(crop, (_PIT_TPL.shape[1], _PIT_TPL.shape[0]),
                      interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(crop, _PIT_TPL, cv2.TM_CCOEFF_NORMED)[0, 0]) >= 0.55


def _arg(flag, default):
    for a in sys.argv:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(flag) + 1]
    return default


TOTAL_RUNS = int(next((a for a in sys.argv[1:] if a.isdigit()), 0))   # 0 = run until STOP file
ARCH = _arg("--arch", "small_cnn")          # FAST arch so live play + recording stay ~35fps
RETRAIN_EVERY = int(_arg("--every", 15))
KEEP_TOP = int(_arg("--keep", 4))
BUFFER = int(_arg("--buffer", 12))          # rolling # of self-demos kept across cycles
BASE_MODEL = _arg("--base", None)           # e.g. small_cnn -> deploy data/demo/<name>.pt as start
USE_WANDB = "--wandb" in sys.argv
# base demo(s) anchored in EVERY retrain (anti-collapse). Default hf2 = the CURRENT-setup base for the
# 60fps model — NOT the stale 35fps demo2/3/4. Override with --anchors a,b (or "" for pure self-play).
ANCHORS = [a for a in _arg("--anchors", "hf2").split(",") if a]
SAVE_W = 960
SAVE_FPS = 35                               # overridden just below to the DEPLOYED model's fps (60 for hf2)
                                            # so recorded self-runs match the model's stack spacing
EXPLORE = "--explore" in sys.argv           # sample non-greedy on a fraction of runs to find better play
EXPLORE_FRAC = min(1.0, max(1e-6, float(_arg("--explore-frac", 0.25))))     # frac of runs that explore
EXPLORE_STRENGTH = min(1.0, max(0.0, float(_arg("--explore-strength", 0.3))))  # per-frame explore prob

# PROMOTION GATE: a retrain is deployed ONLY if it BEATS the currently-deployed champion on the
# held-out human demos (model_score). The old blind hot-swap let survival-noise drift the model
# sideways/worse (adversarially-verified flat, never improving); the gate makes a retrain strictly
# non-worsening. --no-gate restores the blind swap; --gate-margin raises the improvement bar.
GATE = "--no-gate" not in sys.argv
GATE_MARGIN = float(_arg("--gate-margin", 0.0))
GATE_DEMOS = [d for d in _arg("--gate-demos", "demo2,demo3,demo4").split(",") if d]
_LAST_GATE = {"champion": None, "challenger": None}   # for W&B trend logging

# --- unattended-safety knobs ---
MAX_CRASHES = 6            # consecutive per-run failures before aborting the night
ENSURE_FAILS_ESCALATE = 3  # consecutive ensure_running failures before a game restart
DEGEN_ESCALATE = 3         # consecutive 0s/degenerate runs before restarting the game app
                           # (the game can CRASH to the launcher — ensure_running "succeeds"
                           #  but play_until_death returns ~0s because no game is running)
RETRAIN_TIMEOUT_S = 1800   # kill a background retrain hung past this (frees GPU + resumes swaps)
MONITOR_MAX_RESTARTS = 20  # cap on card-solver relaunches
MIN_DUR_S = 5.0            # drop a run shorter than this (false start / wedge) from training
MIN_FRAMES = 100          # ...or with fewer frames than this (blind capture) — never promote it
RECENT_CAP = RETRAIN_EVERY + KEEP_TOP + 4  # bound recent[]+tmp dirs even while a retrain runs long
SHUTDOWN_RETRAIN_WAIT_S = 600  # on clean exit, wait this long for the in-flight retrain to finish

BASE = str(DATA)
REC = os.path.join(BASE, "demo")            # deployed model.pt / model_meta.json live here
try:                                        # match recording fps to the deployed model (60 for hf2)
    SAVE_FPS = int(round(float(json.load(open(os.path.join(REC, "model_meta.json"))).get("fps", 35.0))))
except Exception:
    pass
WORK = os.path.join(BASE, "_selffarm")
os.makedirs(WORK, exist_ok=True)
STOP_FILE = os.path.join(WORK, "STOP")
PY = sys.executable

cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg, log=print)
def _calibrated_device(old=None):
    """Open+start the capture device, RETRYING until the game-area calibrates to the full
    frame. A drifted/cropped game-area (width < ~1000px on the ~1126-wide game) means the
    LDPlayer window isn't rendering the game full-size, so the model can't see the HUD /
    obstacles and won't act — the exact 'not jumping/sliding' failure. If this never passes,
    RESTART LDPlayer (the game must fill its window, no black padding)."""
    if old is not None:
        try:
            old.stop()
        except Exception:
            pass
    d = None
    for attempt in range(6):
        d = open_device(cfg)
        d.start()
        time.sleep(0.8)
        ga = getattr(d, "_ga", None)
        if not isinstance(ga, tuple) or ga[0] > 1000:      # non-ldplayer (no _ga) OR full-frame
            if isinstance(ga, tuple):
                print(f">> capture OK: game-area {ga} conf {round(float(getattr(d,'_calib_conf',0) or 0),2)}", flush=True)
            return d
        print(f">> BAD capture calibration (game-area {ga}) — reopening, retry {attempt+1}/6", flush=True)
        try:
            d.stop()
        except Exception:
            pass
        time.sleep(1.0)
    print("!! capture never calibrated full-frame — proceeding; RESTART LDPlayer if the model won't act", flush=True)
    return d


dev = _calibrated_device()
matcher = TemplateMatcher(cfg.templates_dir)

# start from an explicit base model if asked (else use whatever model.pt is already deployed)
if BASE_MODEL and os.path.exists(os.path.join(REC, f"{BASE_MODEL}.pt")):
    shutil.copy(os.path.join(REC, f"{BASE_MODEL}.pt"), os.path.join(REC, "model.pt"))
    shutil.copy(os.path.join(REC, f"{BASE_MODEL}_meta.json"), os.path.join(REC, "model_meta.json"))
    print(f">> base model set to {BASE_MODEL}", flush=True)

wb = None
if USE_WANDB:
    import wandb as wb
    wb.init(project="cookierun-selffarm", name=f"selffarm-{time.strftime('%Y%m%d-%H%M%S')}",
            config={"arch": ARCH, "retrain_every": RETRAIN_EVERY, "keep_top": KEEP_TOP,
                    "buffer": BUFFER, "anchors": ANCHORS, "save_fps": SAVE_FPS})


def _launch_monitor():
    """(Re)start the independent card-game solver. Its own adb screencap grabber — no dxcam
    contention with the farm. Returns the Popen (or None on failure)."""
    try:
        with open(str(ROOT / "selffarm_monitor.out"), "a") as log_fh:
            p = subprocess.Popen([PY, "-u", "scripts/monitor.py"], cwd=str(ROOT),
                                 stdout=log_fh, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            rc = None
        if rc is not None:
            print(f"!! monitor exited during startup (rc={rc}) -- emulator may be owned elsewhere",
                  flush=True)
            return None
        print(">> card-game solver (monitor.py) launched", flush=True)
        return p
    except Exception as e:
        print(f"!! monitor launch failed (cards won't auto-solve): {e}", flush=True)
        return None


def _recording_usable(complete, duration, frame_count):
    return bool(complete) and duration >= MIN_DUR_S and frame_count >= MIN_FRAMES


def activate_headstart():
    """Press the Head Start ⏩ prompt at run start (settle-gated centre tap) — from ai_farm."""
    wf = getattr(dev, "wait_frame", None)
    t0, prev = time.time(), None
    while time.time() - t0 < 8:
        f = wf(0.1) if wf else dev.last_frame()
        if f is None:
            continue
        p = matcher.find(f, "headstart", 0.55)
        if p and abs(p[0] - 1220) < 400 and abs(p[1] - 690) < 260:
            if prev and abs(p[0] - prev[0]) < 20 and abs(p[1] - prev[1]) < 20:
                dev.tap(*p); time.sleep(0.35); dev.tap(*p)
                print(f">> Head Start pressed at {p}", flush=True)
                return True
            prev = p
    print(">> Head Start prompt NOT found in 8s (skipped)", flush=True)
    return False


def make_recorder(run_dir):
    """on_step that saves frames (background writer, ~35fps, 960px) + derives key-presses from
    the model's FIRED actions — mirrors the human demo format so train2.py can train on it."""
    fdir = os.path.join(run_dir, "frames")
    os.makedirs(fdir, exist_ok=True)
    wq = queue.Queue(maxsize=512)
    frames, keys, pit_times = [], [], []
    frames_lock = threading.Lock()
    accept_results = [True]
    writer_error = [None]

    def writer():
        while True:
            item = wq.get()
            try:
                if item is None:
                    return
                idx, timestamp, small = item
                written = cv2.imwrite(os.path.join(fdir, f"{idx:06d}.jpg"), small,
                                      [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not written:
                    raise OSError(f"failed to write frame {idx}")
                with frames_lock:
                    if accept_results[0]:
                        frames.append({"idx": idx, "t": timestamp})
            except Exception as exc:
                with frames_lock:
                    if writer_error[0] is None:
                        writer_error[0] = exc
            finally:
                wq.task_done()
    # ONE JPEG writer maxes ~47fps; a 60fps recording needs a small POOL or it drops frames, which
    # would make the recorded self-run's effective fps < the model's meta fps (out-of-distribution).
    _tws = [threading.Thread(target=writer, daemon=True) for _ in range(4 if SAVE_FPS >= 50 else 1)]
    for _tw in _tws:
        _tw.start()

    st = {"idx": 0, "last": 0.0, "cur": ACTION_NOOP, "last_seen": 0.0, "last_pit": -9.0}
    gap = 1.0 / SAVE_FPS

    def on_step(now, f, decision):
        st["last_seen"] = now
        if now - st["last_pit"] > 4.0 and _pitfall(f):
            st["last_pit"] = now
            pit_times.append(now)
        if now - st["last"] >= gap:
            h, w = f.shape[:2]
            small = cv2.resize(f, (SAVE_W, int(h * SAVE_W / w)))
            try:
                wq.put_nowait((st["idx"], now, small))
                st["idx"] += 1
            except queue.Full:
                pass
            st["last"] = now
        a = decision.action                      # the model's fired action (NOOP incl. cooldown)
        if a != st["cur"]:
            if st["cur"] in (ACTION_JUMP, ACTION_SLIDE) and keys:
                keys[-1]["dur"] = round(now - keys[-1]["t"], 4)
            if a in (ACTION_JUMP, ACTION_SLIDE):
                keys.append({"t": now, "action": "jump" if a == ACTION_JUMP else "slide", "dur": 0.0})
            st["cur"] = a

    def close(timeout=5.0):
        if st["cur"] in (ACTION_JUMP, ACTION_SLIDE) and keys:
            keys[-1]["dur"] = round(max(0.0, st["last_seen"] - keys[-1]["t"]), 4)
        deadline = time.monotonic() + timeout
        sentinels = 0
        while sentinels < len(_tws) and time.monotonic() < deadline:
            try:
                wq.put(None, timeout=min(0.1, max(0.0, deadline - time.monotonic())))
                sentinels += 1
            except queue.Full:
                continue
        for _tw in _tws:
            _tw.join(timeout=max(0.0, deadline - time.monotonic()))
        closed = all(not _tw.is_alive() for _tw in _tws)
        with frames_lock:
            accept_results[0] = False
            frames.sort(key=lambda frame: frame["idx"])
        return closed, list(pit_times), writer_error[0]
    return on_step, frames, keys, close


def _conf():
    srp = os.path.join(REC, "sweep_results.json")
    try:
        return json.load(open(srp)).get("conf", 0.6) if os.path.exists(srp) else 0.6
    except Exception:
        return 0.6


def _base_k():
    """K of the currently-deployed model — retrains match it so a hot-swap can't change the
    temporal window out from under the fps-consistent recording."""
    try:
        return int(json.load(open(os.path.join(REC, "model_meta.json"))).get("K", 6))
    except Exception:
        return 6


def load_agent():
    return LearnedAgent(cfg, os.path.join(REC, "model.pt"), os.path.join(REC, "model_meta.json"), conf=_conf())


_GATE_SCORE_CACHE = {}   # (path, mtime) -> score: an unchanged champion isn't re-scored each retrain


def _gate_score(model_pt, meta_json):
    """Held-out dodge-quality score (model_score.score_model) — None on any failure so the
    caller can fail-closed (keep the proven champion) rather than deploy an unvetted model.
    Cached by file mtime so the deployed champion (unchanged between retrains) is scored once."""
    path = os.path.join(REC, model_pt)
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        key = None
    if key is not None and key in _GATE_SCORE_CACHE:
        return _GATE_SCORE_CACHE[key]
    try:
        from model_score import score_model
        s = score_model(path, os.path.join(REC, meta_json), eval_demos=GATE_DEMOS)["score"]
    except Exception as e:
        print(f"!! gate scoring failed ({e!r})", flush=True)
        return None
    if key is not None:
        _GATE_SCORE_CACHE[key] = s
    return s


def deploy_retrain():
    """Atomically swap the freshly-trained selffarm.pt into model.pt (temp file + os.replace),
    so the farm never reads a half-written checkpoint — but ONLY if the challenger passes the
    promotion gate (beats the deployed champion on the held-out human demos). On a scoring
    error, or a non-improving challenger, KEEP the current model (fail-closed)."""
    if not os.path.exists(os.path.join(REC, "selffarm.pt")):
        return False
    if GATE:
        champ = _gate_score("model.pt", "model_meta.json")
        chall = _gate_score("selffarm.pt", "selffarm_meta.json")
        _LAST_GATE["champion"], _LAST_GATE["challenger"] = champ, chall
        if champ is None or chall is None:
            print(">> promotion gate: scoring unavailable — KEEPING current model (fail-closed)", flush=True)
            return False
        from model_score import gate_accepts
        if not gate_accepts(champ, chall, GATE_MARGIN):
            print(f">> promotion gate: challenger {chall:.4f} does NOT beat champion {champ:.4f} "
                  f"(margin {GATE_MARGIN}) — KEEPING current model", flush=True)
            return False
        print(f">> promotion gate: challenger {chall:.4f} BEATS champion {champ:.4f} — deploying", flush=True)
    for a, b in (("selffarm.pt", "model.pt"), ("selffarm_meta.json", "model_meta.json")):
        tmp = os.path.join(REC, b + ".tmp")
        shutil.copy(os.path.join(REC, a), tmp)
        os.replace(tmp, os.path.join(REC, b))
    return True


def spawn_retrain(recent):
    """Filter recent runs -> promote the best to demo_self_* -> prune -> launch train2 in the
    BACKGROUND (Popen, non-blocking) so farming never pauses. Retrain K matches the deployed base."""
    # Action 1 (adversarially-verified 2026-07-05): promote ONLY greedy runs. Explore runs
    # survive by inference stochasticity (luck), not policy quality, and are jump-heavier —
    # promoting them as "demonstrations" feeds noise into the demo pool. Fall back to all runs
    # only if a whole batch happened to be explore (never promote nothing).
    pool = [b for b in recent if not b.get("explore")] or recent
    pool.sort(key=lambda b: -b["dur"])
    best = pool[:KEEP_TOP]
    print(f">> retrain trigger — best survivals {[round(b['dur']) for b in best]}s "
          f"of {[round(b['dur']) for b in recent]}s (greedy-only promotion)", flush=True)
    promoted = set()
    for i, b in enumerate(best):
        dest = os.path.join(BASE, f"demo_self_{b['run']}_{i}")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(b["dir"], dest)
        promoted.add(b["dir"])
    for b in recent:                              # prune everything not promoted (explore runs + greedy losers)
        if b["dir"] not in promoted:
            shutil.rmtree(b["dir"], ignore_errors=True)
    allself = sorted((d for d in os.listdir(BASE) if d.startswith("demo_self_")),
                     key=lambda d: os.path.getmtime(os.path.join(BASE, d)))
    for old in allself[:-BUFFER]:
        shutil.rmtree(os.path.join(BASE, old), ignore_errors=True)
    keep = sorted(d for d in os.listdir(BASE) if d.startswith("demo_self_"))
    cmd = [PY, "-u", "scripts/train2.py", "--arch", ARCH, "--k", str(_base_k()),
           "--meta-from", os.path.join(REC, "model_meta.json"),   # inherit the deployed model's arch
           "--out-prefix", "selffarm", "--runs", ",".join(ANCHORS + keep)]
    if USE_WANDB:
        cmd += ["--wandb", "--wandb-project=cookierun-selffarm"]
    print(f">> BACKGROUND retrain launched (arch {ARCH} K{_base_k()} | {len(ANCHORS)} human + "
          f"{len(keep)} self demos) — farm keeps banking", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT),
                            stdout=open(str(ROOT / "selffarm_train.out"), "w"), stderr=subprocess.STDOUT)


def _kill_proc(p, wait_s=10):
    """terminate() then, if it lingers, kill()."""
    if p is None:
        return
    try:
        p.terminate()
        p.wait(timeout=wait_s)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _game_foreground() -> bool:
    """True if CookieRun is the focused app. Returns False ONLY when we can positively see the
    Android launcher is focused (a stray-BACK app-exit — see the 'sliding card' wedge, 2026-07-05).
    Any adb failure or ambiguous output -> True, so we never force a needless relaunch."""
    adb = cfg.adb_path or "adb"
    base = [adb] + (["-s", cfg.device_serial] if cfg.device_serial else [])
    try:
        out = subprocess.run(base + ["shell", "dumpsys window | grep mCurrentFocus"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return True
    if "com.devsisters.crg" in out:
        return True
    if "launcher" in out.lower():
        return False
    return True


# ------------------------------------------------------------------ main loop ------
try:
    os.remove(os.path.join(WORK, "card_active"))   # drop any stale card-veto flag a hard-killed
except OSError:                                    # monitor left behind (else nav's first BACK is
    pass                                           # wrongly vetoed for up to 90s)
monitor = _launch_monitor()
if monitor is None:
    dev.stop()
    raise SystemExit("card-game monitor unavailable -- refusing to share the emulator")
wallet0 = farm.read_wallet(dev, cfg, matcher)
print(f">> SELF-FARM start | wallet {wallet0} | arch {ARCH} K{_base_k()} | ASYNC retrain every "
      f"{RETRAIN_EVERY} runs, keep {KEEP_TOP} | save {SAVE_FPS}fps | "
      f"{'until STOP' if TOTAL_RUNS == 0 else str(TOTAL_RUNS)+' runs'}", flush=True)
agent = load_agent()
retrain_proc, retrain_started, retrain_n = None, 0.0, 0
run_id, since, recent = 0, 0, []
consec_crash, consec_ensure_fail, monitor_restarts, consec_degen = 0, 0, 0, 0

try:
    while (TOTAL_RUNS == 0 or run_id < TOTAL_RUNS) and not os.path.exists(STOP_FILE):
        try:
            # 0. keep the card-solver alive
            if monitor is not None and monitor.poll() is not None:
                if monitor_restarts < MONITOR_MAX_RESTARTS:
                    monitor_restarts += 1
                    monitor = _launch_monitor()
                    if monitor is None:
                        raise SystemExit("card-game monitor unavailable -- stopping self-farm")
                    print(f">> monitor.py had died -> relaunched (#{monitor_restarts})", flush=True)
                else:
                    raise SystemExit("monitor.py restart cap reached -- stopping self-farm")

            # 0b. self-heal: if a stray BACK (or crash) kicked us out to the Android launcher,
            #     relaunch CookieRun so a card-screen wedge can't strand the farm idle for hours.
            if not _game_foreground():
                print(">> game not in foreground (kicked to launcher?) — relaunching", flush=True)
                try:
                    farm._restart_game(cfg, log=print)
                except Exception as e:
                    print(f"!! relaunch failed: {e!r}", flush=True)

            # 1. hot-swap a finished retrain (guarded) OR kill a hung one — at run boundary only
            if retrain_proc is not None:
                rc = retrain_proc.poll()
                if rc is not None:                       # retrain exited
                    _gate_log = {f"gate/{k}": v for k, v in _LAST_GATE.items() if v is not None}
                    if rc != 0:
                        print(f">> background retrain failed (rc={rc}) — kept current model", flush=True)
                    elif deploy_retrain():               # passed the promotion gate + swapped in
                        try:
                            agent = load_agent()          # bad checkpoint -> keep current agent
                            retrain_n += 1
                            print(f">> retrain #{retrain_n} finished -> model HOT-SWAPPED", flush=True)
                            if wb:
                                wb.log({"retrain": retrain_n, "run": run_id, **_gate_log})
                        except Exception as e:
                            print(f"!! hot-swap load failed ({e!r}) — kept current agent", flush=True)
                    else:                                 # trained fine but gate kept the champion
                        print(">> retrain finished but NOT deployed (promotion gate kept the "
                              "current model) — still banking", flush=True)
                        if wb:
                            wb.log({"run": run_id, **_gate_log})
                    retrain_proc = None
                elif time.time() - retrain_started > RETRAIN_TIMEOUT_S:
                    print(f"!! retrain hung >{RETRAIN_TIMEOUT_S}s — killing it (frees GPU + resumes swaps)", flush=True)
                    _kill_proc(retrain_proc)
                    retrain_proc = None

            # 2. reach a fresh run, ESCALATING if we keep failing (never a silent spin)
            spend = {}
            if not farm.ensure_running(dev, matcher, cfg, log=print, cycle=spend, gift_state={"depleted": True}):
                consec_ensure_fail += 1
                print(f"!! could not reach a run (fail #{consec_ensure_fail}) at {time.strftime('%H:%M:%S')}", flush=True)
                if consec_ensure_fail % ENSURE_FAILS_ESCALATE == 0:
                    tier = consec_ensure_fail // ENSURE_FAILS_ESCALATE
                    print(f">> escalating (tier {tier}) — force-restarting the game", flush=True)
                    try:
                        farm._restart_game(cfg, log=print)
                    except Exception as e:
                        print(f"!! restart_game failed: {e!r}", flush=True)
                    if tier >= 2:   # game restarts didn't help -> capture may be dead, reopen it
                        print(">> escalating — reopening the capture device", flush=True)
                        try:
                            dev = _calibrated_device(dev)
                        except Exception as e:
                            print(f"!! device reopen failed: {e!r}", flush=True)
                time.sleep(30 if consec_ensure_fail >= 30 else 3)   # back off hard on a persistent wedge
                continue
            consec_ensure_fail = 0

            # 3. play + record one run — farming NEVER pauses, even while a retrain runs on the GPU.
            #    Head Start is pressed INSIDE ensure_running (farm_boosts._watch_headstart, armed at
            #    the boost-Play tap — the prompt's window is too short for a post-return presser). So
            #    here we just hand to play_until_death the instant the run is live (do NOT re-run an
            #    8s activate_headstart — the prompt is already gone, so it would only stall the run
            #    uncontrolled for 8s).
            _tw = time.time()
            while time.time() - _tw < 6.0 and not _in_run(dev, matcher):
                time.sleep(0.1)
            agent.reset()
            run_id += 1
            # exploration: on ~EXPLORE_FRAC of runs, let the agent sample non-greedy at uncertain
            # frames to discover better play; the rest stay greedy (banking). Survival selection
            # keeps whichever helped. Confident dodges are never randomised (see LearnedAgent).
            exploring = EXPLORE and (run_id % max(1, round(1 / EXPLORE_FRAC)) == 0)
            agent.explore = EXPLORE_STRENGTH if exploring else 0.0
            if exploring:
                print(f">> run {run_id}: EXPLORING (p={EXPLORE_STRENGTH} at uncertain frames)", flush=True)
            run_dir = os.path.join(WORK, f"tmp_{run_id}")
            shutil.rmtree(run_dir, ignore_errors=True)
            on_step, frames, keys, close = make_recorder(run_dir)
            try:
                dur = farm.play_until_death(dev, cfg, agent, matcher, max_s=1800, min_s=8.0,
                                            log=lambda *a: None, on_step=on_step)
            finally:
                recording_closed, pit_times, recording_error = close()
            complete = recording_closed and recording_error is None and bool(frames)
            with open(os.path.join(run_dir, "frames.json"), "w") as fh:
                json.dump({"frames": frames, "save_w": SAVE_W, "duration_s": dur,
                           "actual_fps": round(len(frames) / max(dur, 0.1), 1),
                           "pit_times": pit_times, "complete": complete}, fh)
            with open(os.path.join(run_dir, "keys.json"), "w") as fh:
                json.dump(keys, fh)
            if not complete:
                print(f"!! recording incomplete -- excluded from training: "
                      f"{recording_error or 'writer did not stop/empty capture'}", flush=True)
            if not recording_closed or recording_error is not None:
                raise SystemExit("recording writer failed -- stopping self-farm")
            res = farm.read_run_result(dev, cfg, matcher)
            coins = res.get("coins", 0) if res.get("read_ok") else None
            jz = sum(1 for k in keys if k["action"] == "jump")
            print(f">> run {run_id}: {dur:.0f}s | coins {coins} | jump {jz} slide {len(keys)-jz} | "
                  f"fps {round(len(frames)/max(dur,0.1),1)} | "
                  f"retrain {'RUNNING' if retrain_proc is not None else 'idle'}", flush=True)
            if wb:
                wb.log({"run/survival_s": dur, "run/coins": coins or 0, "run/slides": len(keys) - jz,
                        "run/fps": round(len(frames) / max(dur, 0.1), 1),
                        "run": run_id, "retrain_inflight": int(retrain_proc is not None)})

            # 4. degenerate-run floor — never train on it; and if the game has CRASHED to the
            #    launcher (repeated 0s runs), restart the game app so we don't spin the night away
            if not _recording_usable(complete, dur, len(frames)):
                consec_degen += 1
                print(f">> run {run_id} unusable (complete={complete} dur={dur:.0f}s "
                      f"frames={len(frames)}) — dropped (#{consec_degen})", flush=True)
                shutil.rmtree(run_dir, ignore_errors=True)
                if consec_degen % DEGEN_ESCALATE == 0:
                    tier = consec_degen // DEGEN_ESCALATE
                    print(f">> {consec_degen} degenerate runs — game likely crashed/wedged; restarting game app (tier {tier})", flush=True)
                    try:
                        farm._restart_game(cfg, log=print)
                    except Exception as e:
                        print(f"!! restart_game failed: {e!r}", flush=True)
                    if tier >= 2:   # restart alone didn't help -> capture may be stale too
                        print(">> also reopening the capture device", flush=True)
                        try:
                            dev = _calibrated_device(dev)
                        except Exception as e:
                            print(f"!! device reopen failed: {e!r}", flush=True)
            else:
                consec_degen = 0
                recent.append({"dir": run_dir, "dur": dur, "coins": coins, "run": run_id,
                               "explore": exploring})
                since += 1

            # 5. bound recent[] + its tmp dirs so disk can't fill during a long-running retrain
            if len(recent) > RECENT_CAP:
                recent.sort(key=lambda b: -b["dur"])
                for b in recent[RECENT_CAP:]:
                    shutil.rmtree(b["dir"], ignore_errors=True)
                recent = recent[:RECENT_CAP]

            # 6. spawn a background retrain when a batch is ready and none is in flight
            if since >= RETRAIN_EVERY and retrain_proc is None:
                retrain_proc = spawn_retrain(recent)
                retrain_started = time.time()
                since, recent = 0, []

            consec_crash = 0     # a full clean iteration clears the crash streak

        except KeyboardInterrupt:
            raise
        except Exception as e:
            consec_crash += 1
            print(f"!! run-loop error (#{consec_crash}/{MAX_CRASHES}) at {time.strftime('%H:%M:%S')}: {e!r}", flush=True)
            traceback.print_exc()
            if consec_crash >= MAX_CRASHES:
                print("!! too many consecutive crashes — aborting the night", flush=True)
                break
            # recovery: reopen the capture device + restart the game, then carry on
            try:
                dev = _calibrated_device(dev)   # returns a NEW device handle
            except Exception as e2:
                print(f"!! device reopen failed: {e2!r}", flush=True)
            try:
                farm._restart_game(cfg, log=print)
            except Exception as e3:
                print(f"!! restart_game failed: {e3!r}", flush=True)
            time.sleep(5)

finally:
    # unconditional teardown — releases the device + kills BOTH subprocesses even on a fatal error
    print(">> shutting down — cleanup", flush=True)
    if retrain_proc is not None:
        print(">> finishing the in-flight retrain before exit...", flush=True)
        try:
            retrain_proc.wait(timeout=SHUTDOWN_RETRAIN_WAIT_S)
            if retrain_proc.returncode == 0:
                deploy_retrain()
        except Exception:
            print(">> retrain didn't finish in time — killing it", flush=True)
            _kill_proc(retrain_proc)
    try:
        walletN = farm.read_wallet(dev, cfg, matcher)
    except Exception:
        walletN = "?"
    print(f">> SELF-FARM done | {run_id} runs | {retrain_n} retrains | wallet {wallet0} -> {walletN}", flush=True)
    _kill_proc(monitor)
    if wb:
        try:
            wb.finish()
        except Exception:
            pass
    try:
        dev.stop()
    except Exception:
        pass
