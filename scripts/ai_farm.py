"""Monitored AI farm: continuous Double-Coins-gated runs driven by the LearnedAgent, with
per-hit diagnostics for improving the model. For every HP drop it dumps the pre-hit window:
what the model SAW (frames at ~-0.7s/-0.3s/impact) and what it THOUGHT (class/prob/action
trace), so failures can be categorized (blind / fired-but-hit / cooldown-blocked) instead
of guessed at. Also logs per-run coin results. Runs until max runs or Ctrl+C."""
import sys, os, time, json, queue, threading
from _runtime import CONFIG, DATA, ROOT
import cv2, numpy as np
from collections import deque
from cookierun_bot.config import load_config
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot.device import open_device
from cookierun_bot.policies.learned import LearnedAgent
from cookierun_bot.gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from cookierun_bot import farm

REC = str(DATA / "demo")
OUT = str(DATA / "ai_hits")
os.makedirs(OUT, exist_ok=True)
MAX_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ACT = {ACTION_NOOP: "none", ACTION_JUMP: "jump", ACTION_SLIDE: "slide"}
# Resolution of the diagnostic hit/death clips (the frames correct.py shows for labeling).
# 960x540 matches the demo recorder's 960-wide frames + is crisp to label. The DECISION-time
# ring holds ~160 of these (2.5s @ 60fps), so RAM ~= W*H*3*160: 960x540 ~= 250MB. This is
# ONLY for human labeling — the model always trains/infers on a 96x224 grayscale crop, so the
# clip resolution has ZERO effect on model performance. (1920x1080 works too but ~1GB ring.)
CLIP_WH = (960, 540)
_DIAG_SESSION = f"{int(time.time())}{os.getpid() % 100000:05d}"


def _diag_stem(run_no, session_id=None):
    """Keep the historical r*_h* shape while preventing relaunch overwrites."""
    return f"r{session_id or _DIAG_SESSION}{run_no:03d}"


def _parse_hybrid_confs(raw, default):
    import math
    if not raw:
        return default, default
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise SystemExit("AIFARM_HYBRID_CONFS must contain exactly two numbers") from exc
    if len(values) != 2 or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise SystemExit("AIFARM_HYBRID_CONFS must contain exactly two numbers between 0 and 1")
    return values


class _RecordingWriter:
    """Bounded JPEG writer whose metadata contains only completed writes."""

    def __init__(self, run_dir, maxsize=1024):
        self.run_dir = os.fspath(run_dir)
        os.makedirs(os.path.join(self.run_dir, "frames"), exist_ok=True)
        self.q = queue.Queue(maxsize=maxsize)
        self.frames = []
        self._frames_lock = threading.Lock()
        self._accept_results = True
        self._closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                idx, timestamp, payload = item
                if self.error is not None:
                    continue
                try:
                    path = os.path.join(self.run_dir, "frames", f"{idx:06d}.jpg")
                    with open(path, "wb") as fh:
                        if fh.write(payload) != len(payload):
                            raise OSError(f"short write: {path}")
                    with self._frames_lock:
                        if self._accept_results:
                            self.frames.append({"idx": idx, "t": timestamp})
                except Exception as exc:
                    self.error = exc
            finally:
                self.q.task_done()

    def submit(self, idx, timestamp, payload):
        if self._closed or self.error is not None or not self.thread.is_alive():
            return False
        try:
            self.q.put_nowait((idx, timestamp, payload))
            return True
        except queue.Full:
            return False

    def close(self, timeout=10.0):
        if self._closed:
            return not self.thread.is_alive()
        self._closed = True
        deadline = time.monotonic() + timeout
        while self.thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self.q.put(None, timeout=min(0.1, remaining))
                break
            except queue.Full:
                continue
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        closed = not self.thread.is_alive()
        with self._frames_lock:
            self._accept_results = False
            self.frames.sort(key=lambda frame: frame["idx"])
        if not closed and self.error is None:
            self.error = TimeoutError("recording writer did not stop before timeout")
        return closed

def hp_frac(f):
    strip = f[125:158, 240:820]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255])) > 0).mean())


# BONUSTIME detector: bonus phases overlay/wash the HP-bar strip, so hp_frac "drops" there
# are ARTIFACTS, not hits (verified 2026-07-11: 10/10 sampled hit clips + all 3 death clips
# were bonus phases). Detect the fixed-position "BONUSTIME" banner by grayscale template
# match in a scale-normalized crop (TM_CCOEFF_NORMED = gain-invariant, since the banner
# PULSES bright-yellow<->gray; measured: hit clips score >=0.3 on the bright phase, 2000
# normal demo frames max 0.22). The 3s latch in on_step bridges the fade dips.
_BT_TPL = cv2.imread(str(ROOT / "templates" / "bonustime_norm.png"), cv2.IMREAD_GRAYSCALE)


def bonustime(f) -> bool:
    if _BT_TPL is None:
        return False
    h, w = f.shape[:2]
    c = cv2.cvtColor(f[0:int(h * 0.083), int(w * 0.04):int(w * 0.31)], cv2.COLOR_BGR2GRAY)
    c = cv2.resize(c, (_BT_TPL.shape[1], _BT_TPL.shape[0]), interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(c, _BT_TPL, cv2.TM_CCOEFF_NORMED)[0, 0]) >= 0.30


# PIT-FALL detector: the "5 for 1 Pit Lift" revive pill appears at EVERY fall (the user's
# setup tanks up to 3 falls before true death, so falls cause NO HP drop and NO terminal —
# they were invisible to hits.jsonl AND to the IQL reward until this). Same fixed-position
# scale-normalized match pattern as bonustime; fractions measured off r01_death_d11.jpg.
_PIT_TPL = cv2.imread(str(ROOT / "templates" / "pitlift_norm.png"), cv2.IMREAD_GRAYSCALE)


def pitfall(f) -> bool:
    if _PIT_TPL is None:
        return False
    h, w = f.shape[:2]
    c = cv2.cvtColor(f[int(h * 0.830):int(h * 0.956), int(w * 0.372):int(w * 0.684)],
                     cv2.COLOR_BGR2GRAY)
    c = cv2.resize(c, (_PIT_TPL.shape[1], _PIT_TPL.shape[0]), interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(c, _PIT_TPL, cv2.TM_CCOEFF_NORMED)[0, 0]) >= 0.55

cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg, log=print)
dev = open_device(cfg); dev.start()
matcher = TemplateMatcher(cfg.templates_dir)
# deploy confidence: the sweep stores the winner's best-scoring gate alongside the model
conf = 0.6
_sr = os.path.join(REC, "sweep_results.json")
if os.path.exists(_sr):
    conf = json.load(open(_sr)).get("conf", 0.6)
    print(f"deploy conf from sweep winner: {conf}", flush=True)
# K-stack geometry for per-hit correction frames (must match training: fps spacing + K)
TRAIN_FPS, K_STACK = 35.0, 4
_mm = os.path.join(REC, "model_meta.json")
if os.path.exists(_mm):
    _meta = json.load(open(_mm))
    TRAIN_FPS = float(_meta.get("fps", 35.0))
    K_STACK = int(_meta.get("K", 4))
# JUMP gate capped at 0.60: the sweep's 0.95 was tuned on a val metric (hit-rate vs
# false-fires) that never prices PIT DEATHS. Probed live 2026-07-11: all 3 batch runs died
# falling at BONUSTIME platform gaps with the model predicting jump at 0.61-0.75 — below
# 0.95, so the gate turned "hesitant jump" into "no jump" into the pit. The asymmetry
# (learned.py docstring): a wrong jump costs a healable HP tick; a missed jump is a pit =
# run over. Slide keeps its own strict 0.90 gate (conf_slide) — wrong slides ARE fatal.
# AIFARM_JUMP_CAP overrides the cap for gate A/Bs (film models predict pit-jumps at lower
# confidence than the champion — a lower cap lets them fire without touching the code).
_JUMP_CAP = float(os.environ.get("AIFARM_JUMP_CAP", "0.60"))
if _JUMP_CAP != 0.60:
    print(f"jump-gate cap OVERRIDE: {_JUMP_CAP} (AIFARM_JUMP_CAP)", flush=True)
# AIFARM_SLIDE_CONF likewise overrides the slide gate for A/Bs (default: cfg/LearnedAgent)
_SLIDE_CONF = os.environ.get("AIFARM_SLIDE_CONF")
if _SLIDE_CONF is not None:
    print(f"slide-gate OVERRIDE: {_SLIDE_CONF} (AIFARM_SLIDE_CONF)", flush=True)
# EMULATOR-REFRESH exit: LDPlayer degrades over long batches (measured capture fps
# 70 -> 37 over ~26h) and low fps directly worsens the bot's reaction timing. When the
# measured decision fps stays below AIFARM_FPS_MIN for 2 CONSECUTIVE runs, exit with
# REFRESH_EXIT at the run boundary — supervisor.py hands the code up and monitor.py
# does a full ldconsole quit+launch refresh, then relaunches us for the remaining runs
# (a fresh process also re-initializes dxcam cleanly). AIFARM_FPS_MIN=0 disables it.
REFRESH_EXIT = 17            # unused elsewhere (2 = the blind-run exit below)
FPS_MIN = float(os.environ.get("AIFARM_FPS_MIN", "45"))
_low_fps = 0                 # consecutive sub-FPS_MIN runs


def _mk_agent(model_name, jump_conf):
    return LearnedAgent(cfg, os.path.join(REC, f"{model_name}.pt"),
                        os.path.join(REC, f"{model_name}_meta.json"), conf=jump_conf,
                        conf_slide=float(_SLIDE_CONF) if _SLIDE_CONF is not None else None)


# AIFARM_HYBRID="base,bonus" = phase-aware two-model agent (policies/hybrid_phase.py):
# base earns in normal stages, bonus (clean dodger) takes the BONUSTIME pit gauntlets.
# AIFARM_HYBRID_CONFS="0.60,0.45" sets per-model jump gates (default min(conf, cap) both).
# DURABLE deploy: data/demo/hybrid.json {"base","bonus","confs":[..]} activates the hybrid
# without env vars (delete the file to revert to plain model.pt); env vars still override.
_HYBRID = os.environ.get("AIFARM_HYBRID")
_HYB_FILE = os.path.join(REC, "hybrid.json")
if not _HYBRID and os.path.exists(_HYB_FILE):
    _hj = json.load(open(_HYB_FILE))
    _HYBRID = f"{_hj['base']},{_hj['bonus']}"
    if _hj.get("confs"):
        os.environ.setdefault("AIFARM_HYBRID_CONFS",
                              ",".join(str(c) for c in _hj["confs"]))
    print(f"hybrid.json deploy: {_HYBRID} confs={_hj.get('confs')}", flush=True)
if _HYBRID:
    from cookierun_bot.policies.hybrid_phase import HybridPhaseAgent
    _bn, _xn = [s.strip() for s in _HYBRID.split(",")]
    _hc = os.environ.get("AIFARM_HYBRID_CONFS", "")
    _bc, _xc = _parse_hybrid_confs(_hc, min(conf, _JUMP_CAP))
    agent = HybridPhaseAgent(_mk_agent(_bn, _bc), _mk_agent(_xn, _xc),
                             templates_dir=str(ROOT / "templates"))
    print(f"HYBRID agent: base={_bn}@{_bc} | bonus={_xn}@{_xc}", flush=True)
else:
    agent = _mk_agent("model", min(conf, _JUMP_CAP))

# AIFARM_HAZARD="hazard" wraps the agent with the learned pit-fall trigger (M4.2): the base
# policy is BLIND to the pits that kill it (M1.1), so this detector forces a jump when P(pit)
# crosses AIFARM_HAZARD_THR (default 0.7) outside BONUSTIME. Off by default => deployed
# behaviour unchanged. hazard.pt is trained by scripts/train_hazard.py.
_HAZ = os.environ.get("AIFARM_HAZARD")
if _HAZ:
    from cookierun_bot.policies.hazard_trigger import HazardTrigger
    _hpath = _HAZ if _HAZ.endswith(".pt") else os.path.join(REC, f"{_HAZ}.pt")
    _hmeta = _hpath[:-3] + "_meta.json"
    agent = HazardTrigger(agent, _hpath, _hmeta,
                          thr=float(os.environ.get("AIFARM_HAZARD_THR", "0.7")),
                          check_every=int(os.environ.get("AIFARM_HAZARD_EVERY", "3")),
                          confirm_reads=int(os.environ.get("AIFARM_HAZARD_CONFIRM", "1")),
                          max_per_episode=int(os.environ.get("AIFARM_HAZARD_MAXJUMP", "2")))
    print(f"HAZARD trigger: {os.path.basename(_hpath)} @ thr "
          f"{os.environ.get('AIFARM_HAZARD_THR', '0.7')}", flush=True)
print(f"backend: {type(dev).__name__} | dxcam: {getattr(dev,'_use_dx',None)} | model: {agent._device}", flush=True)

diag = open(os.path.join(OUT, "hits.jsonl"), "a")
totals = {"coins": 0, "runs": 0}
unread = 0        # runs whose Result screen we couldn't read (banked, but uncounted)
# ground-truth wallet baseline: the per-run Result OCR is fragile (a card game / level-up
# modal can hide the Result screen — 3/15 last marathon read 0 for exactly this reason),
# so also snapshot the menu coin balance at start; the wallet delta is survival-independent.
wallet0 = farm.read_wallet(dev, cfg, matcher)
print(f">> WALLET start: {wallet0}" if wallet0 is not None
      else ">> WALLET start: (not on menu — skipped)", flush=True)

for run_no in range(1, MAX_RUNS + 1):
    print(f"\n===== RUN {run_no}/{MAX_RUNS} — boost gate (Double Coins) =====", flush=True)
    cycle = {}
    # gift draws DISABLED: the gift branch false-matched its way into the Party Run
    # mode-select screen and looped there for a whole session (observed live 2026-07-04).
    # depleted=True skips the branch entirely; farming never depends on it.
    if not farm.ensure_running(dev, matcher, cfg, log=print, cycle=cycle,
                               gift_state={"depleted": True}):
        print("!! could not reach a run — stopping", flush=True)
        break
    print(f">> RUN LIVE (boost spend: {cycle})", flush=True)
    # ensure_running owns the primary Head Start watcher; play_until_death keeps its guarded
    # fallback. A third tap loop here raced stale frames and could double-tap the prompt.
    agent.reset()

    # JPEG-compressed ring (~40KB/frame): 600 entries ~= 10s at 60fps for ~25MB, so the
    # death dump can reach back to the FATAL moment (the old 2.5s raw ring filled entirely
    # with the post-death static scene — every death clip was the same frozen frame).
    ring = deque(maxlen=600)      # (t, jpg_bytes, cls, prob, action)
    # AIFARM_RECORD=1: persist EVERY run as training data (data/botrun_*/ in the demo
    # recording schema) — the frames are ALREADY JPEG-encoded for the ring, so recording
    # is just writing those bytes to disk on a worker thread. This is the data flywheel:
    # each farm run yields on-policy trajectories + auto-mineable pit-fall labels for the
    # next IQL iteration (falls were unmeasurable before; 25 mined examples were too few).
    vrec = None                   # NOT `rec` — on_step's hit logger already binds that name
    if os.environ.get("AIFARM_RECORD") == "1":
        _rdir = os.path.join(os.path.dirname(REC), f"botrun_{time.strftime('%m%d_%H%M%S')}")
        vrec = {"dir": _rdir, "writer": _RecordingWriter(_rdir), "keys": [], "idx": 0,
                "cur": ACTION_NOOP}
        print(f">> RECORDING run to {_rdir}", flush=True)
    hp_hist = []; hits = []; steps = [0]; last_hit = [0.0]
    last_bt = [-9.0]              # bonustime latch: last time the banner was seen
    bt_skipped = [0]
    pits = [0]                    # PIT FALLS this run (revive-tanked falls included — the
    last_pit = [-9.0]             # setup absorbs 3, so these never show in hp/terminals)
    pit_times = []
    # REBOUND-CONFIRM: a real hit leaves hp PERSISTENTLY lower; an overlay sweeping the
    # HP strip (bonus washes, zone-title cards — background-dependent, so the banner
    # template alone can't catch them all: the bright-jungle zone dropped its score
    # below threshold and 200+ artifacts/run flooded the log) REBOUNDS within a frame
    # or two. Hold each dip as pending and only log it if hp is still depressed 0.4s
    # later. Detection-free and zone-independent; the 10s ring keeps the pre-dip
    # frames intact for the dump either way.
    pending = [None]              # {"t", "rmax", "hp"} awaiting confirmation
    rebounds = [0]
    _MAX_HIT_DUMPS = 60           # per-run image-dump cap (a flood once wrote ~2.8k jpgs/run)
    acts = {ACTION_NOOP: 0, ACTION_JUMP: 0, ACTION_SLIDE: 0}

    def _ring_img(entry):
        return cv2.imdecode(np.frombuffer(entry[1], np.uint8), cv2.IMREAD_COLOR)

    def on_step(now, f, decision):
        steps[0] += 1
        acts[decision.action] = acts.get(decision.action, 0) + 1
        # decision.reason = "model:<cls>:<p>" | "model:jump-cooldown"
        parts = decision.reason.split(":")
        cls = parts[1] if len(parts) > 1 else "?"
        prob = float(parts[2]) if len(parts) > 2 else 0.0
        small = cv2.resize(f, (640, 360))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            _bts = buf.tobytes()
            ring.append((now, _bts, cls, prob, ACT.get(decision.action, "?")))
            if vrec is not None:
                if vrec["writer"].submit(vrec["idx"], now, _bts):
                    vrec["idx"] += 1
        if vrec is not None and decision.action != vrec["cur"]:
            if vrec["cur"] in (ACTION_JUMP, ACTION_SLIDE) and vrec["keys"]:
                vrec["keys"][-1]["dur"] = round(max(0.0, now - vrec["keys"][-1]["t"]), 4)
            if decision.action in (ACTION_JUMP, ACTION_SLIDE):
                vrec["keys"].append({"t": now, "action": ACT[decision.action], "dur": 0.0})
            vrec["cur"] = decision.action
        if bonustime(f):
            last_bt[0] = now
        if now - last_pit[0] > 4.0 and pitfall(f):     # prompt shows ~1-2s; 4s refractory
            pits[0] += 1
            last_pit[0] = now
            pit_times.append(now)
            print(f"PIT FALL #{pits[0]} @ {now:.0f}s", flush=True)
        hp = hp_frac(f)
        hp_hist.append((now, hp))
        while hp_hist and now - hp_hist[0][0] > 1.1: hp_hist.pop(0)
        rmax = max(h for _, h in hp_hist)
        # confirm (or discard) a pending dip once 0.4s has passed
        if pending[0] is not None and now - pending[0]["t"] >= 0.4:
            p_t, p_rmax, p_hp = pending[0]["t"], pending[0]["rmax"], pending[0]["hp"]
            pending[0] = None
            if hp > p_rmax - 0.05:
                rebounds[0] += 1               # hp bounced back => overlay artifact, not damage
            else:
                k = len(hits)
                trace = [(round(t - p_t, 2), c, round(p, 2), a) for t, _, c, p, a in ring
                         if p_t - t <= 1.5 and t <= p_t]
                if k < _MAX_HIT_DUMPS:
                    # dump what the model saw at decision time (~0.7s/0.3s before) + impact
                    for tag, dt in (("pre07", 0.7), ("pre03", 0.3), ("impact", 0.0)):
                        cand = min(ring, key=lambda r: abs((p_t - dt) - r[0]))
                        cv2.imwrite(os.path.join(OUT, f"{_diag_stem(run_no)}_h{k:03d}_{tag}.jpg"),
                                    _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    # k-frames: a true K-stack at TRAINING fps spacing ending at -0.3s so
                    # correction labels (scripts/correct.py) train with real motion context
                    try:
                        for ki in range(K_STACK):
                            dt = 0.3 + (K_STACK - 1 - ki) / TRAIN_FPS
                            cand = min(ring, key=lambda r: abs((p_t - dt) - r[0]))
                            cv2.imwrite(os.path.join(OUT, f"{_diag_stem(run_no)}_h{k:03d}_k{ki}.jpg"),
                                        _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    except Exception:
                        pass
                rec = {"run": int(_diag_stem(run_no)[1:]), "batch_run": run_no,
                       "hit": k, "t": round(p_t, 1),
                       "hp": [round(p_rmax, 2), round(p_hp, 2)], "trace": trace}
                hits.append(rec)
                diag.write(json.dumps(rec) + "\n"); diag.flush()
                print(f"HIT r{run_no} #{k} @ {p_t:.0f}s hp {p_rmax:.2f}->{p_hp:.2f}", flush=True)
        if rmax - hp > 0.06 and now > 4 and now - last_hit[0] > 0.6 and pending[0] is None:
            last_hit[0] = now
            hp_hist[:] = [(now, hp)]
            # bonus phases wash/overlay the HP strip => hp_frac drops are artifacts there.
            # The banner latch catches the detectable ones cheaply; rebound-confirm above
            # catches the rest (including zones where the banner template fails).
            if now - last_bt[0] < 3.0:
                bt_skipped[0] += 1
                return
            pending[0] = {"t": now, "rmax": rmax, "hp": hp}

    dur = 0.0
    recording_failed = False
    play_completed = False
    try:
        dur = farm.play_until_death(dev, cfg, agent, matcher, max_s=1800, min_s=8.0,
                                    log=print, on_step=on_step)
        play_completed = True
    finally:
        if vrec is not None:
            recorded_dur = max(dur, ring[-1][0] if ring else 0.0)
            if vrec["cur"] in (ACTION_JUMP, ACTION_SLIDE) and vrec["keys"]:
                vrec["keys"][-1]["dur"] = round(
                    max(0.0, recorded_dur - vrec["keys"][-1]["t"]), 4)
            closed = vrec["writer"].close(timeout=10)
            written_frames = list(vrec["writer"].frames)
            complete = (play_completed and closed and vrec["writer"].error is None
                        and bool(written_frames))
            with open(os.path.join(vrec["dir"], "frames.json"), "w") as fh:
                json.dump({"frames": written_frames, "save_w": 640, "duration_s": recorded_dur,
                           "actual_fps": round(len(written_frames) / max(recorded_dur, 0.1), 1),
                           "pit_times": pit_times, "complete": complete}, fh)
            with open(os.path.join(vrec["dir"], "keys.json"), "w") as fh:
                json.dump(vrec["keys"], fh)
            if not complete:
                print(f"!! recording incomplete: {vrec['writer'].error or 'run/writer incomplete'}",
                      flush=True)
            print(f">> RECORDED {len(written_frames)} frames + {len(vrec['keys'])} actions "
                  f"-> {os.path.basename(vrec['dir'])}", flush=True)
            recording_failed = play_completed and not complete
    # DEATH CLIP: pit falls (the classic killer) end the run WITHOUT an HP drop, so the hit
    # logger never sees them — this dump does. h999 is the death sentinel: correct.py's
    # rNN_hMMM globbing queues it for labeling like any hit. The post-death scene sits
    # static for seconds, so ALSO dump a coarse d00..dNN sequence over the ring's full ~10s
    # reach — the fatal action is in there, not in the last frozen second.
    try:
        if ring:
            t_end = ring[-1][0]
            for tag, dt in (("pre07", 0.7), ("pre03", 0.3), ("impact", 0.0)):
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"{_diag_stem(run_no)}_h999_{tag}.jpg"),
                            _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
            for ki in range(K_STACK):
                dt = 0.3 + (K_STACK - 1 - ki) / TRAIN_FPS
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"{_diag_stem(run_no)}_h999_k{ki}.jpg"),
                            _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
            for di, dt in enumerate((9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0,
                                     1.6, 1.2, 0.9, 0.6, 0.3, 0.0)):
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"{_diag_stem(run_no)}_death_d{di:02d}.jpg"),
                            _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass
    fps = steps[0] / max(dur, 0.1)
    if steps[0] == 0:
        # BLIND RUN: play_until_death saw zero HUD frames — the capture stack is broken
        # (happens after emulator/window/adb disruptions; observed repeatedly). A fresh
        # process re-initializes dxcam/monitor/adb cleanly — exit so the supervisor
        # recycles us instead of burning more boosted runs blind.
        print("!! BLIND RUN detected (0 decisions) — exiting for a clean restart", flush=True)
        dev.stop()
        sys.exit(2)
    _contact = (len(hits) + rebounds[0]) / max(dur / 60, 0.01)   # every HP dip incl. the
    print(f">> RUN {run_no} OVER @ {dur:.0f}s | {len(hits)} hits ({len(hits)/max(dur/60,0.01):.1f}/min, "
          f"{bt_skipped[0]} bonus-artifact skipped, {rebounds[0]} rebound-discarded) "
          f"| PITS={pits[0]} | contact={_contact:.1f}/min "
          f"| fps={fps:.0f} | jump={acts[ACTION_JUMP]} slide={acts[ACTION_SLIDE]}"
          f"{f' | hazard-fires={agent.fires}' if hasattr(agent, 'fires') else ''}", flush=True)
    res = farm.read_run_result(dev, cfg, matcher)
    # audit trail: save the Result frame — a clipped OCR region misread 11,411 for what
    # should have been ~6 digits (observed live); the saved frame settles disputes.
    try:
        rf = farm._nav_read(dev)
        if rf is not None:
            cv2.imwrite(os.path.join(OUT, f"result_{_diag_stem(run_no)}.jpg"),
                        cv2.resize(rf, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass
    totals["runs"] += 1
    # NOTE: keep the ">> RESULT:" prefix in BOTH branches — the supervisor counts completed
    # runs by that prefix, and an unread run IS a completed run (it just wasn't tallied).
    if res.get("read_ok"):
        totals["coins"] += res.get("coins", 0)
        print(f">> RESULT: {{'coins': {res.get('coins', 0)}, "
              f"'ingredients': {res.get('ingredients', 0)}}} | session total: "
              f"{totals['coins']} coins / {totals['runs']} runs"
              f"{f' ({unread} unread)' if unread else ''}", flush=True)
    else:
        unread += 1
        print(f">> RESULT: UNREAD — Result screen missed (a card game / level-up / box "
              f"screen pre-empted it); coins were banked but can't be counted. | session "
              f"total: {totals['coins']} coins / {totals['runs']} runs ({unread} unread)",
              flush=True)

    if recording_failed:
        print("!! stopping after the completed run because its recorder failed", flush=True)
        diag.close()
        dev.stop()
        sys.exit(2)  # supervisor has counted RESULT and may restart without leaking the writer

    # EMULATOR-REFRESH check — run boundary ONLY (the >> RESULT: line above is already
    # out, so the supervisor has counted this run; nothing is ever cut off mid-run).
    # Two consecutive slow runs = a degraded emulator, not one unlucky run. On the LAST
    # run there is nothing left to farm — finish normally instead of forcing a refresh.
    if FPS_MIN > 0 and fps < FPS_MIN:
        _low_fps += 1
    else:
        _low_fps = 0
    if _low_fps >= 2 and run_no < MAX_RUNS:
        print(f">> FPS-DEGRADED: fps {fps:.0f} < {FPS_MIN:.0f} for {_low_fps} consecutive "
              f"runs — exiting {REFRESH_EXIT} for an emulator refresh", flush=True)
        diag.close()
        dev.stop()
        sys.exit(REFRESH_EXIT)

diag.close()
walletN = farm.read_wallet(dev, cfg, matcher)   # best-effort end balance (None if not on menu)
dev.stop()
if wallet0 is not None and walletN is not None:
    wallet_tail = f" | wallet {wallet0}->{walletN} = net {walletN - wallet0:+} (ground truth)"
elif walletN is not None:
    wallet_tail = f" | wallet now {walletN}"
else:
    wallet_tail = ""
print(f"\nDONE: {totals['runs']} runs, {totals['coins']} coins counted"
      f"{f', {unread} UNREAD (banked, uncounted)' if unread else ''}{wallet_tail}",
      flush=True)
