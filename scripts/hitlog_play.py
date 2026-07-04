"""Decoupled hit-logger: NO navigation. Waits for the in-run HUD (slide) to appear, then
drives play_until_death with an on_step hook that logs every HP drop + decision timeline +
culprit frame. Start this, then manually start a run — it attaches when the run begins."""
import sys, time, os
from _runtime import CONFIG, DATA
import cv2, numpy as np
from cookierun_bot.config import load_config
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot.device import open_device
from cookierun_bot.policies.rule_based import StreamingRuleBasedAgent
from cookierun_bot.gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from cookierun_bot import farm

OUT = str(DATA / "hits")
os.makedirs(OUT, exist_ok=True)
for p in os.listdir(OUT):
    os.remove(os.path.join(OUT, p))
ACT = {ACTION_NOOP: ".", ACTION_JUMP: "JUMP", ACTION_SLIDE: "SLIDE"}


def hp_frac(f):
    strip = f[125:158, 240:820]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255])) > 0).mean())


cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg, log=print)
dev = open_device(cfg); dev.start()
matcher = TemplateMatcher(cfg.templates_dir)
agent = StreamingRuleBasedAgent(cfg)

print(">> waiting for a run (start one now)...", flush=True)
t_wait = time.time()
while time.time() - t_wait < 180:
    f = dev.last_frame()
    if f is not None and matcher.present(f, "slide", 0.60):
        break
    time.sleep(0.3)
else:
    print("no run started within 180s", flush=True); dev.stop(); raise SystemExit
print(">> RUN DETECTED; logging hits", flush=True)

ring = []
hp_hist = []
nhits = [0]


def on_step(now, f, decision):
    ring.append((now, f, decision.action, decision.reason))
    del ring[:-40]
    hp = hp_frac(f)
    hp_hist.append((now, hp))
    while hp_hist and now - hp_hist[0][0] > 1.1:
        hp_hist.pop(0)
    rmax = max(h for _, h in hp_hist)
    if rmax - hp > 0.06 and now > 4:
        recent = [r for r in ring if now - r[0] < 1.3]
        tl = " ".join(ACT[a] if a != ACTION_NOOP else "." for _, _, a, _ in recent)
        reasons = sorted({rs for _, _, a, rs in recent if a != ACTION_NOOP})
        print(f"HIT @ {now:.0f}s hp {rmax:.2f}->{hp:.2f} | acts: {tl} | reasons: {reasons}", flush=True)
        culprit = min(recent, key=lambda r: abs((now - 0.7) - r[0]))[1] if recent else f
        cv2.imwrite(OUT + f"\\hit_{nhits[0]:02d}_{now:.0f}s_culprit.png", culprit)
        cv2.imwrite(OUT + f"\\hit_{nhits[0]:02d}_{now:.0f}s_impact.png", f)
        nhits[0] += 1
        hp_hist[:] = [(now, hp)]


dur = farm.play_until_death(dev, cfg, agent, matcher, min_s=4.0, log=print, on_step=on_step)
print(f"RUN OVER @ {dur:.0f}s; {nhits[0]} hits", flush=True)
dev.stop(); print("DONE", flush=True)
