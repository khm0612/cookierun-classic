"""Live test of the imitation-learned policy (model.pt from train2.py). Navigates to a
plain run, drives LearnedAgent through play_until_death, logs every HP drop + the model's
action mix + effective decision fps. Compare against dodge_check.py (rule-based baseline:
35 hit-events / 120s, all 'clear')."""
import sys, time, subprocess
from _runtime import CONFIG, DATA
import cv2, numpy as np
from cookierun_bot.config import load_config
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot.device import open_device
from cookierun_bot.policies.learned import LearnedAgent
from cookierun_bot.gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from cookierun_bot import farm

REC = DATA / "demo"

def hp_frac(f):
    strip = f[125:158, 240:820]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255])) > 0).mean())

cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg, log=print)
dev = open_device(cfg); dev.start()
matcher = TemplateMatcher(cfg.templates_dir)
agent = LearnedAgent(cfg, str(REC / "model.pt"), str(REC / "model_meta.json"), conf=0.6)
print("backend:", type(dev).__name__, "| model on:", agent._device, flush=True)
serial = cfg.device_serial
def tap(x, y): subprocess.run(["adb", "-s", serial, "shell", "input", "touchscreen", "swipe",
                               str(x), str(y), str(x), str(y), "70"], timeout=10)

# REAL farm navigation: full boost gate — 3 tiles checked + Double Coins Multi-Buy
# (standing user rule: NEVER start a run without Double Coins). ensure_running navigates
# on sharp adb frames via _nav_read/nav_frame and refuses to Play un-boosted.
print(">> navigating to a run (boost gate + Double Coins)...", flush=True)
cycle = {}
if not farm.ensure_running(dev, matcher, cfg, log=print, cycle=cycle):
    print("!! could not reach a run", flush=True); dev.stop(); raise SystemExit
print(f">> boost spend this cycle: {cycle}", flush=True)
print(">> RUN DETECTED — learned agent driving", flush=True)

steps = [0]; hits = [0]; acts = {ACTION_NOOP: 0, ACTION_JUMP: 0, ACTION_SLIDE: 0}
hp_hist = []; last_hit = [0.0]
def on_step(now, f, decision):
    steps[0] += 1
    acts[decision.action] = acts.get(decision.action, 0) + 1
    hp = hp_frac(f)
    hp_hist.append((now, hp))
    while hp_hist and now - hp_hist[0][0] > 1.1: hp_hist.pop(0)
    rmax = max(h for _, h in hp_hist)
    # 0.6s refractory: at 70fps one collision's multi-frame HP drain re-triggered the
    # 0.06 drop check many times (110 "hits"/min was an artifact, not more collisions)
    if rmax - hp > 0.06 and now > 4 and now - last_hit[0] > 0.6:
        hits[0] += 1
        last_hit[0] = now
        print(f"HIT #{hits[0]} @ {now:.0f}s hp {rmax:.2f}->{hp:.2f} ({decision.reason})", flush=True)
        hp_hist[:] = [(now, hp)]

dur = farm.play_until_death(dev, cfg, agent, matcher, max_s=300, min_s=4.0,
                            log=lambda *a: None, on_step=on_step)
fps = steps[0] / max(dur, 0.1)
print(f">> RUN OVER @ {dur:.0f}s | {hits[0]} hit-events | {hits[0]/max(dur/60,0.01):.1f} hits/min | "
      f"decisions fps={fps:.1f} | actions: jump={acts[ACTION_JUMP]} slide={acts[ACTION_SLIDE]} "
      f"noop={acts[ACTION_NOOP]}", flush=True)
dev.stop(); print("DONE", flush=True)
