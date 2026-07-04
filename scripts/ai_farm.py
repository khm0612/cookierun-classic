"""Monitored AI farm: continuous Double-Coins-gated runs driven by the LearnedAgent, with
per-hit diagnostics for improving the model. For every HP drop it dumps the pre-hit window:
what the model SAW (frames at ~-0.7s/-0.3s/impact) and what it THOUGHT (class/prob/action
trace), so failures can be categorized (blind / fired-but-hit / cooldown-blocked) instead
of guessed at. Also logs per-run coin results. Runs until max runs or Ctrl+C."""
import sys, os, time, json
from _runtime import CONFIG, DATA
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

def hp_frac(f):
    strip = f[125:158, 240:820]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255])) > 0).mean())

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
agent = LearnedAgent(cfg, os.path.join(REC, "model.pt"), os.path.join(REC, "model_meta.json"),
                     conf=conf)
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

    ring = deque(maxlen=160)      # ~2.5s at 60fps: (t, small_frame, cls, prob, action)
    hp_hist = []; hits = []; steps = [0]; last_hit = [0.0]
    acts = {ACTION_NOOP: 0, ACTION_JUMP: 0, ACTION_SLIDE: 0}

    def on_step(now, f, decision):
        steps[0] += 1
        acts[decision.action] = acts.get(decision.action, 0) + 1
        # decision.reason = "model:<cls>:<p>" | "model:jump-cooldown"
        parts = decision.reason.split(":")
        cls = parts[1] if len(parts) > 1 else "?"
        prob = float(parts[2]) if len(parts) > 2 else 0.0
        small = cv2.resize(f, (640, 360))
        ring.append((now, small, cls, prob, ACT.get(decision.action, "?")))
        hp = hp_frac(f)
        hp_hist.append((now, hp))
        while hp_hist and now - hp_hist[0][0] > 1.1: hp_hist.pop(0)
        rmax = max(h for _, h in hp_hist)
        if rmax - hp > 0.06 and now > 4 and now - last_hit[0] > 0.6:
            last_hit[0] = now
            k = len(hits)
            trace = [(round(t - now, 2), c, round(p, 2), a) for t, _, c, p, a in ring
                     if now - t <= 1.5]
            # dump what the model saw at decision time (~0.7s and ~0.3s before) + impact
            for tag, dt in (("pre07", 0.7), ("pre03", 0.3), ("impact", 0.0)):
                cand = min(ring, key=lambda r: abs((now - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h{k:03d}_{tag}.jpg"), cand[1],
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            # k-frames: a true K-stack at TRAINING fps spacing ending at -0.3s so
            # correction labels (scripts/correct.py) train with real motion context
            # (k0 = oldest ... k{K-1} = the -0.3s labeled frame; matches train2's
            # oldest->newest stack order)
            try:
                for ki in range(K_STACK):
                    dt = 0.3 + (K_STACK - 1 - ki) / TRAIN_FPS
                    cand = min(ring, key=lambda r: abs((now - dt) - r[0]))
                    cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h{k:03d}_k{ki}.jpg"),
                                cand[1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                pass
            rec = {"run": run_no, "hit": k, "t": round(now, 1),
                   "hp": [round(rmax, 2), round(hp, 2)], "trace": trace}
            hits.append(rec)
            diag.write(json.dumps(rec) + "\n"); diag.flush()
            print(f"HIT r{run_no} #{k} @ {now:.0f}s hp {rmax:.2f}->{hp:.2f}", flush=True)
            hp_hist[:] = [(now, hp)]

    dur = farm.play_until_death(dev, cfg, agent, matcher, max_s=1800, min_s=8.0,
                                log=print, on_step=on_step)
    # DEATH CLIP: the final ~1.5s of the run. Pit falls (the classic killer) end the run
    # WITHOUT an HP drop, so the hit logger never sees them — this dump does. h999 is the
    # death sentinel: correct.py's rNN_hMMM globbing queues it for labeling like any hit.
    try:
        if ring:
            t_end = ring[-1][0]
            for tag, dt in (("pre07", 0.7), ("pre03", 0.3), ("impact", 0.0)):
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h999_{tag}.jpg"), cand[1],
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            for ki in range(K_STACK):
                dt = 0.3 + (K_STACK - 1 - ki) / TRAIN_FPS
                cand = min(ring, key=lambda r: abs((t_end - dt) - r[0]))
                cv2.imwrite(os.path.join(OUT, f"r{run_no:02d}_h999_k{ki}.jpg"), cand[1],
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    print(f">> RUN {run_no} OVER @ {dur:.0f}s | {len(hits)} hits ({len(hits)/max(dur/60,0.01):.1f}/min) "
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
