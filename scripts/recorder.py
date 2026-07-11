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

# args: [name] [hifps]. `hifps` = 60fps experiment: emulator runs 60fps, so save 960px (not
# 1920) — the JPEG writer can't encode 1920px fast enough to keep up with 60fps (queue drops
# -> ~36fps), and 960px is still far above the <=240px training crop. Name goes to its own
# folder; 60fps demos use an `hf*` namespace so the normal `demo*` sweep never mixes fps.
_args = [a for a in sys.argv[1:] if a != "hifps"]
HIFPS = "hifps" in sys.argv
TEST = len(_args) > 0 and _args[0] == "test"
# each recording gets its own folder — pass a name as the first arg, else auto-pick unused.
# NEVER reuse a folder: the wipe below would destroy training data from an earlier session.
_base = str(DATA)
if not TEST and _args:
    REC = os.path.join(_base, _args[0])
elif TEST:
    REC = os.path.join(_base, "demo_test")
elif HIFPS:
    n = 1
    while os.path.exists(os.path.join(_base, f"hf{n}")):
        n += 1
    REC = os.path.join(_base, f"hf{n}")
else:
    n = 2
    while os.path.exists(os.path.join(_base, f"demo{n}")):
        n += 1
    REC = os.path.join(_base, f"demo{n}")
print(f">> recording to {REC}  ({'HIFPS 60fps/960px' if HIFPS else '35fps/1920px'})", flush=True)
FRAMES = os.path.join(REC, "frames")
os.makedirs(FRAMES, exist_ok=True)
if not TEST:
    for p in os.listdir(FRAMES):
        os.remove(os.path.join(FRAMES, p))
SAVE_W = 960 if HIFPS else 1920    # 960px encodes ~4x faster -> keeps up with 60fps capture;
                                   # still >> the <=240px training crop, so no quality loss
KEYMAP = {"w": "jump", "s": "slide"}

keys = []            # {t, key, action, dur}  one per real press; `dur` = HOLD length in
                     # seconds, filled on release (0.0 if still held at stop). Lets training
                     # label the whole slide/jump SPAN so the model can decide duration.
_down = set()
_open = {}           # key char -> index in `keys` of its currently-held press
stop = threading.Event()

def on_press(k):
    ch = getattr(k, "char", None)
    if ch is None:
        return
    ch = ch.lower()
    if ch in KEYMAP and ch not in _down:
        _down.add(ch)
        _open[ch] = len(keys)
        keys.append({"t": time.monotonic(), "key": ch, "action": KEYMAP[ch], "dur": 0.0})

def on_release(k):
    ch = getattr(k, "char", None)
    if ch is None:
        return
    ch = ch.lower()
    if ch in _down:
        _down.discard(ch)
        i = _open.pop(ch, None)
        if i is not None:                          # how long the key was HELD
            keys[i]["dur"] = round(time.monotonic() - keys[i]["t"], 4)

kl = keyboard.Listener(on_press=on_press, on_release=on_release); kl.start()

cfg = load_config(str(CONFIG))
cfg = farm._auto_serial_config(cfg)
dev = open_device(cfg); dev.start()
matcher = TemplateMatcher(cfg.templates_dir)

wq = queue.Queue(maxsize=512)
def writer():
    while True:
        item = wq.get()
        if item is None:
            break
        idx, small = item
        cv2.imwrite(os.path.join(FRAMES, f"{idx:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
# One JPEG-writer thread maxed out at ~47fps (queue filled, frames dropped). At 60fps the
# encode+disk-write must parallelize, so run a small pool. (HIFPS only needs the throughput.)
N_WRITERS = 4 if HIFPS else 1
_writers = [threading.Thread(target=writer, daemon=True) for _ in range(N_WRITERS)]
for _tw in _writers:
    _tw.start()

dur_cap = 20 if TEST else 1800
print(f">> {'TEST (20s) — press W and S a few times' if TEST else 'RECORDING — play your full run (W=jump S=slide)'}", flush=True)

SAVE_FPS = 60 if HIFPS else 35   # hifps experiment saves 60fps; normal stays 35 to match the
_save_gap = 1.0 / SAVE_FPS        # sweep's REC_FPS=35. dxcam delivers 240+fps, so this is a real cap.
frames = []; idx = 0; t0 = time.monotonic(); confirmed = False
next_save = t0
wait_frame = getattr(dev, "wait_frame", None)

# Result-screen auto-stop runs in a BACKGROUND thread: the "ok" template ONLY matches at
# native 2560px (every downscale drops it below threshold), and a 57-90ms full-res match on
# the hot path would cap the save loop well under 60fps. So the loop just stashes the latest
# full-res frame and this watcher matches it a few times a second. It also honours an external
# STOP file (`<rec>/STOP`) for a guaranteed clean manual finalize when a run won't end itself.
_latest = [None]
STOP_FILE = os.path.join(REC, "STOP")
def result_watcher():
    seen = 0
    while not stop.is_set():
        time.sleep(0.4)
        if os.path.exists(STOP_FILE):
            print(">> STOP file — finalizing.", flush=True); stop.set(); break
        lf = _latest[0]
        if TEST or lf is None or time.monotonic() - t0 < 20:
            continue
        if matcher.find(lf, "ok", 0.80) is not None:
            seen += 1
            if seen > 4:
                print(">> Result screen — run over.", flush=True); stop.set(); break
        else:
            seen = 0
rw = threading.Thread(target=result_watcher, daemon=True); rw.start()

while time.monotonic() - t0 < dur_cap and not stop.is_set():
    f = wait_frame(0.5) if wait_frame else dev.last_frame()
    if f is None:
        continue
    _latest[0] = f                     # hand the freshest full-res frame to the result watcher
    now = time.monotonic()
    if now < next_save:
        continue                       # keys are captured by the pynput thread regardless
    # advance the target on a FIXED cadence grid, not "gap after the last actual save" — the
    # latter adds the ~4ms save-work to every period (60fps target -> 48fps actual).
    next_save += _save_gap
    if next_save < now:                # fell far behind -> resync (don't burst-catch-up)
        next_save = now + _save_gap
    if not TEST:
        h, w = f.shape[:2]
        small = cv2.resize(f, (SAVE_W, int(h * SAVE_W / w)))
        try:
            wq.put_nowait((idx, small))
            frames.append({"idx": idx, "t": now})   # only list frames that WILL be written;
        except queue.Full:                           # a dropped frame has no JPEG on disk, so
            pass                                     # recording its idx would break training loads
    idx += 1
    if not confirmed and keys:
        confirmed = True
        print(f">> INPUT CONFIRMED: {len(keys)} key presses (last: {keys[-1]['action']})", flush=True)
    if not TEST and idx % 300 == 0:
        jz = sum(1 for k in keys if k["action"] == "jump")
        print(f"   {idx} frames ({idx/(now-t0):.0f} fps), {len(keys)} keys (jump={jz} slide={len(keys)-jz}), {now-t0:.0f}s", flush=True)
    # (Result-screen auto-stop is handled by the background result_watcher thread above.)

stop.set(); kl.stop()
for _tw in _writers:                 # one sentinel per writer so the pool drains + joins
    wq.put(None)
for _tw in _writers:
    _tw.join(timeout=5)
dur = time.monotonic() - t0
if not TEST:
    json.dump({"frames": frames, "save_w": SAVE_W, "duration_s": dur,
               "actual_fps": round(idx / max(dur, 0.1), 1), "hifps": HIFPS},
              open(os.path.join(REC, "frames.json"), "w"))
    json.dump(keys, open(os.path.join(REC, "keys.json"), "w"))
dev.stop()
jz = sum(1 for k in keys if k["action"] == "jump")
print(f">> {'TEST' if TEST else 'DONE'}: {idx} frames @ {idx/dur:.0f} fps, {len(keys)} keys "
      f"(jump={jz} slide={len(keys)-jz}) over {dur:.0f}s", flush=True)
if keys:
    print(">> sample:", keys[:8], flush=True)
