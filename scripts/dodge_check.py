"""Self-contained dodge diagnostic AT THE NEW CAPTURE FPS. Navigates to a plain run (no
boost spend — dodging is independent of boosts), then runs the rule-based agent with an
on_step hook that (a) counts effective decision fps and (b) logs every HP drop. Answers the
one question the 1fps capture made unanswerable: does the cookie actually dodge at ~15fps?"""
import sys, time, subprocess
from _runtime import CONFIG
import cv2, numpy as np
from cookierun_bot.config import load_config
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot.device import open_device
from cookierun_bot.policies.rule_based import StreamingRuleBasedAgent
from cookierun_bot.gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from cookierun_bot import farm

ACT = {ACTION_NOOP: ".", ACTION_JUMP: "J", ACTION_SLIDE: "S"}

def hp_frac(f):
    strip = f[125:158, 240:820]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255])) > 0).mean())

cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg, log=print)
dev = open_device(cfg); dev.start()
print("backend:", type(dev).__name__, flush=True)
matcher = TemplateMatcher(cfg.templates_dir)
agent = StreamingRuleBasedAgent(cfg)
serial = cfg.device_serial
# 70ms swipe-press, NOT `input tap`: on a fresh LDPlayer boot plain taps are swallowed in-game
def tap(x, y): subprocess.run(["adb", "-s", serial, "shell", "input", "touchscreen", "swipe",
                               str(x), str(y), str(x), str(y), "70"], timeout=10)

# navigate to a plain run (no boost buys)
print(">> navigating to a run...", flush=True)
for _ in range(24):
    f = dev.last_frame()
    if f is None: time.sleep(0.3); continue
    if matcher.present(f, "slide", 0.60): break
    if matcher.find(f, "ok", 0.82): tap(1280, 1232)
    elif matcher.find(f, "tile_hp", 0.80) or matcher.find(f, "chesttile", 0.80): tap(1765, 1197)
    elif matcher.find(f, "play", 0.80): tap(*matcher.find(f, "play", 0.80))
    else:
        for n in ("confirm", "confirm2", "close", "close2", "openall"):
            p = matcher.find(f, n, 0.82)
            if p: tap(*p); break
    time.sleep(1.3)
if not matcher.present(dev.last_frame(), "slide", 0.60):
    print("!! could not reach a run", flush=True); dev.stop(); raise SystemExit
print(">> RUN DETECTED", flush=True)

steps = [0]; hits = [0]; hp_hist = []
def on_step(now, f, decision):
    steps[0] += 1
    hp = hp_frac(f)
    hp_hist.append((now, hp))
    while hp_hist and now - hp_hist[0][0] > 1.1: hp_hist.pop(0)
    rmax = max(h for _, h in hp_hist)
    if rmax - hp > 0.06 and now > 4:
        hits[0] += 1
        print(f"HIT #{hits[0]} @ {now:.0f}s hp {rmax:.2f}->{hp:.2f} ({decision.reason})", flush=True)
        hp_hist[:] = [(now, hp)]

dur = farm.play_until_death(dev, cfg, agent, matcher, max_s=120, min_s=4.0, log=lambda *a: None, on_step=on_step)
fps = steps[0] / max(dur, 0.1)
print(f">> RUN OVER @ {dur:.0f}s | {hits[0]} hits | {hits[0]/max(dur/60,0.01):.1f} hits/min | "
      f"effective decision fps={fps:.1f} ({steps[0]} decisions)", flush=True)
dev.stop(); print("DONE", flush=True)
