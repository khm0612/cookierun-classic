"""Monitored AI farm: continuous Double-Coins-gated runs driven by the LearnedAgent, with
per-hit diagnostics for improving the model. For every HP drop it dumps the pre-hit window:
what the model SAW (frames at ~-0.7s/-0.3s/impact) and what it THOUGHT (class/prob/action
trace), so failures can be categorized (blind / fired-but-hit / cooldown-blocked) instead
of guessed at. Also logs per-run coin results. Runs until max runs or Ctrl+C."""
import sys, os, time, json
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
    _bc, _xc = ([float(x) for x in _hc.split(",")] if _hc
                else [min(conf, _JUMP_CAP)] * 2)
    agent = HybridPhaseAgent(_mk_agent(_bn, _bc), _mk_agent(_xn, _xc),
                             templates_dir=str(ROOT / "templates"))
    print(f"HYBRID agent: base={_bn}@{_bc} | bonus={_xn}@{_xc}", flush=True)
else:
    agent = _mk_agent("model", min(conf, _JUMP_CAP))
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
    # Head Start via SHARP adb frames (user directive: press it EVERY round). The in-run
    # dxcam path proved flaky at run start (stale frames = prompt never seen); sharp
    # screencaps score the prompt 0.90. play_until_death's own check stays as backstop.
    # Head Start watch on the FAST capture path (user directive: press it EVERY round).
    # The prompt lives only ~2s and can appear the instant the HUD shows — a sharp-frame
    # poll (~1s per look, starts after setup prints) proved structurally too slow
    # ("no centre match at all" while the user watched the prompt come and go). At 60fps
    # every prompt frame is sampled; the strict CENTRE position gate makes the low
    # threshold safe (all known false matches sit far outside the centre box).
    t_hs = time.time()
    activated = False
    hs_prev = None
    wait_frame = getattr(dev, "wait_frame", None)
    while time.time() - t_hs < 8:
        fhs = wait_frame(0.1) if wait_frame else dev.last_frame()
        if fhs is None:
            continue
        p = matcher.find(fhs, "headstart", 0.55)
        # stable-point tap: taps at the historical rest point failed — the button's live
        # settle position drifts; tap where two consecutive matches agree (<20px)
        if p and abs(p[0] - 1220) < 400 and abs(p[1] - 690) < 260:
            if hs_prev and abs(p[0] - hs_prev[0]) < 20 and abs(p[1] - hs_prev[1]) < 20:
                print(f">> Head Start settled at {p} — tapping it", flush=True)
                dev.tap(*p)
                time.sleep(0.35)
                dev.tap(*p)
                activated = True
                break
            hs_prev = p
    if not activated:
        print(">> Head Start: prompt not seen within 8s of run start", flush=True)
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
    rec = None
    if os.environ.get("AIFARM_RECORD") == "1":
        import queue as _q
        import threading as _th
        _rdir = os.path.join(os.path.dirname(REC), f"botrun_{time.strftime('%m%d_%H%M%S')}")
        os.makedirs(os.path.join(_rdir, "frames"), exist_ok=True)
        _wq = _q.Queue(maxsize=1024)

        def _writer():
            while True:
                item = _wq.get()
                if item is None:
                    return
                _i, _b = item
                with open(os.path.join(_rdir, "frames", f"{_i:06d}.jpg"), "wb") as _fh:
                    _fh.write(_b)

        _wt = _th.Thread(target=_writer, daemon=True)
        _wt.start()
        rec = {"dir": _rdir, "q": _wq, "thread": _wt, "frames": [], "keys": [], "idx": 0}
        print(f">> RECORDING run to {_rdir}", flush=True)
    hp_hist = []; hits = []; steps = [0]; last_hit = [0.0]
    last_bt = [-9.0]              # bonustime latch: last time the banner was seen
    bt_skipped = [0]
    pits = [0]                    # PIT FALLS this run (revive-tanked falls included — the
    last_pit = [-9.0]             # setup absorbs 3, so these never show in hp/terminals)
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
            if rec is not None:
                try:
                    rec["q"].put_nowait((rec["idx"], _bts))
                    rec["frames"].append({"idx": rec["idx"], "t": now})
                except Exception:
                    pass                       # full queue: drop frame, keep idx monotonic
                rec["idx"] += 1
                if decision.action != ACTION_NOOP:
                    rec["keys"].append({
                        "t": now, "action": ACT.get(decision.action, "none"),
                        "dur": 0.5 if decision.action == ACTION_SLIDE else 0.1})
        if bonustime(f):
            last_bt[0] = now
        if now - last_pit[0] > 4.0 and pitfall(f):     # prompt shows ~1-2s; 4s refractory
            pits[0] += 1
            last_pit[0] = now
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
                        cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h{k:03d}_{tag}.jpg"),
                                    _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    # k-frames: a true K-stack at TRAINING fps spacing ending at -0.3s so
                    # correction labels (scripts/correct.py) train with real motion context
                    try:
                        for ki in range(K_STACK):
                            dt = 0.3 + (K_STACK - 1 - ki) / TRAIN_FPS
                            cand = min(ring, key=lambda r: abs((p_t - dt) - r[0]))
                            cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h{k:03d}_k{ki}.jpg"),
                                        _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    except Exception:
                        pass
                rec = {"run": run_no, "hit": k, "t": round(p_t, 1),
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

    dur = farm.play_until_death(dev, cfg, agent, matcher, max_s=1800, min_s=8.0,
                                log=print, on_step=on_step)
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
                cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h999_{tag}.jpg"),
                            _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
            for ki in range(K_STACK):
                dt = 0.3 + (K_STACK - 1 - ki) / TRAIN_FPS
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h999_k{ki}.jpg"),
                            _ring_img(cand), [cv2.IMWRITE_JPEG_QUALITY, 85])
            for di, dt in enumerate((9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0,
                                     1.6, 1.2, 0.9, 0.6, 0.3, 0.0)):
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_death_d{di:02d}.jpg"),
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
    if rec is not None:                        # finalize the training recording
        rec["q"].put(None)
        rec["thread"].join(timeout=10)
        json.dump({"frames": rec["frames"], "save_w": 640, "duration_s": dur,
                   "actual_fps": round(len(rec["frames"]) / max(dur, 0.1), 1)},
                  open(os.path.join(rec["dir"], "frames.json"), "w"))
        json.dump(rec["keys"], open(os.path.join(rec["dir"], "keys.json"), "w"))
        print(f">> RECORDED {len(rec['frames'])} frames + {len(rec['keys'])} actions "
              f"-> {os.path.basename(rec['dir'])}", flush=True)
    _contact = (len(hits) + rebounds[0]) / max(dur / 60, 0.01)   # every HP dip incl. the
    print(f">> RUN {run_no} OVER @ {dur:.0f}s | {len(hits)} hits ({len(hits)/max(dur/60,0.01):.1f}/min, "
          f"{bt_skipped[0]} bonus-artifact skipped, {rebounds[0]} rebound-discarded) "
          f"| PITS={pits[0]} | contact={_contact:.1f}/min "
          f"| fps={fps:.0f} | jump={acts[ACTION_JUMP]} slide={acts[ACTION_SLIDE]}", flush=True)
    res = farm.read_run_result(dev, cfg, matcher)
    # audit trail: save the Result frame — a clipped OCR region misread 11,411 for what
    # should have been ~6 digits (observed live); the saved frame settles disputes.
    try:
        rf = farm._nav_read(dev)
        if rf is not None:
            cv2.imwrite(os.path.join(OUT, f"result_r{run_no:02d}.jpg"),
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
