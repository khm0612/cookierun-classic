"""Record a human demo run for imitation learning: every scrcpy frame (timestamped) + every
KEY PRESS (W=jump, S=slide) via pynput, on the same PC clock. Bot sends NO input; human plays.
Arg 'test' = short 20s input-verify window; else records until Result screen / 12 min."""
import sys, time, os, threading, json, queue
from _runtime import CONFIG, DATA
import cv2
from pynput import keyboard
from cookierun_bot.config import load_config
from cookierun_bot.device import open_device
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot import farm

TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
# each recording gets its own folder (demo, demo2, demo3...) — pass a name as argv[1],
# else auto-pick the first unused. NEVER reuse a folder: the wipe below would destroy
# training data from an earlier session.
_base = str(DATA)
if not TEST and len(sys.argv) > 1:
    REC = os.path.join(_base, sys.argv[1])
elif TEST:
    REC = os.path.join(_base, "demo_test")
else:
    n = 2
    while os.path.exists(os.path.join(_base, f"demo{n}")):
        n += 1
    REC = os.path.join(_base, f"demo{n}")
print(f">> recording to {REC}", flush=True)
FRAMES = os.path.join(REC, "frames")
os.makedirs(FRAMES, exist_ok=True)
if not TEST:
    for p in os.listdir(FRAMES):
        os.remove(os.path.join(FRAMES, p))
SAVE_W = 960
KEYMAP = {"w": "jump", "s": "slide"}

keys = []            # {t, key, action}   (one per real press; auto-repeat de-duped)
_down = set()
stop = threading.Event()

def on_press(k):
    ch = getattr(k, "char", None)
    if ch is None:
        return
    ch = ch.lower()
    if ch in KEYMAP and ch not in _down:
        _down.add(ch)
        keys.append({"t": time.monotonic(), "key": ch, "action": KEYMAP[ch]})

def on_release(k):
    ch = getattr(k, "char", None)
    if ch:
        _down.discard(ch.lower())

kl = keyboard.Listener(on_press=on_press, on_release=on_release); kl.start()

cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg)
dev = open_device(cfg); dev.start()
matcher = TemplateMatcher(cfg.templates_dir)

wq = queue.Queue(maxsize=256)
def writer():
    while True:
        item = wq.get()
        if item is None:
            break
        idx, small = item
        cv2.imwrite(os.path.join(FRAMES, f"{idx:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
tw = threading.Thread(target=writer, daemon=True); tw.start()

dur_cap = 20 if TEST else 1800
print(f">> {'TEST (20s) — press W and S a few times' if TEST else 'RECORDING — play your full run (W=jump S=slide)'}", flush=True)

SAVE_FPS = 60            # dxcam delivers ~270 new fps; save at 60 (user request; ~17ms granularity,
_save_gap = 1.0 / SAVE_FPS   # 8min = ~29k frames / ~2-3GB — fine)
frames = []; idx = 0; t0 = time.monotonic(); result_seen = 0; confirmed = False
last_save = 0.0
wait_frame = getattr(dev, "wait_frame", None)
while time.monotonic() - t0 < dur_cap and not stop.is_set():
    f = wait_frame(0.5) if wait_frame else dev.last_frame()
    if f is None:
        continue
    now = time.monotonic()
    if now - last_save < _save_gap:
        continue                       # keys are captured by the pynput thread regardless
    last_save = now
    if not TEST:
        h, w = f.shape[:2]
        small = cv2.resize(f, (SAVE_W, int(h * SAVE_W / w)))
        try: wq.put_nowait((idx, small))
        except queue.Full: pass
        frames.append({"idx": idx, "t": now})
    idx += 1
    if not confirmed and keys:
        confirmed = True
        print(f">> INPUT CONFIRMED: {len(keys)} key presses (last: {keys[-1]['action']})", flush=True)
    if not TEST and idx % 300 == 0:
        jz = sum(1 for k in keys if k["action"] == "jump")
        print(f"   {idx} frames ({idx/(now-t0):.0f} fps), {len(keys)} keys (jump={jz} slide={len(keys)-jz}), {now-t0:.0f}s", flush=True)
    # Result-screen detection is a full-res template match (~56ms) — far too costly to run
    # every frame (it alone would cap capture at ~15fps). The Result screen persists for many
    # seconds, so checking a few times a second is plenty. Only touch result_seen on check
    # frames, else the intervening frames would reset the counter.
    if not TEST and idx % 6 == 0:
        if matcher.find(f, "ok", 0.82):
            result_seen += 1
            if result_seen > 6 and now - t0 > 20:
                print(">> Result screen — run over.", flush=True); break
        else:
            result_seen = 0

stop.set(); kl.stop()
wq.put(None); tw.join(timeout=5)
dur = time.monotonic() - t0
if not TEST:
    json.dump({"frames": frames, "save_w": SAVE_W, "duration_s": dur}, open(os.path.join(REC, "frames.json"), "w"))
    json.dump(keys, open(os.path.join(REC, "keys.json"), "w"))
dev.stop()
jz = sum(1 for k in keys if k["action"] == "jump")
print(f">> {'TEST' if TEST else 'DONE'}: {idx} frames @ {idx/dur:.0f} fps, {len(keys)} keys "
      f"(jump={jz} slide={len(keys)-jz}) over {dur:.0f}s", flush=True)
if keys:
    print(">> sample:", keys[:8], flush=True)
