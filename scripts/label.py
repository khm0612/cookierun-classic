"""Turn the recorded demo (frames.json + clicks.json) into per-frame labels for imitation
learning. A frame is labelled jump/slide if a click of that zone falls within a small window
around the frame's timestamp, else none. Prints the class distribution + a click histogram."""
import sys, os, json, csv
from _runtime import DATA

REC = str(DATA / "demo")
WIN_BEFORE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12   # label frames up to 120ms BEFORE a click
WIN_AFTER = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10    # ...and 100ms after

fm = json.load(open(os.path.join(REC, "frames.json")))
frames = fm["frames"]
keys = json.load(open(os.path.join(REC, "keys.json")))
print(f"{len(frames)} frames over {fm['duration_s']:.0f}s ({len(frames)/max(fm['duration_s'],1):.0f} fps); "
      f"{len(keys)} key presses", flush=True)
if not keys:
    print("NO KEYS — cannot label. Check the recording."); raise SystemExit

# per-frame label: assign a keypress's action (jump/slide) to frames in
# [t_press - WIN_BEFORE, t_press + WIN_AFTER]; everything else = none.
labels = {f["idx"]: "none" for f in frames}
ft = [(f["idx"], f["t"]) for f in frames]
for c in keys:
    tc, act = c["t"], c["action"]
    for idx, t in ft:
        if tc - WIN_BEFORE <= t <= tc + WIN_AFTER:
            labels[idx] = act          # jump/slide overrides none

from collections import Counter
dist = Counter(labels.values())
print("label distribution:", dict(dist), flush=True)
print("keys by action:", dict(Counter(c["action"] for c in keys)), flush=True)

with open(os.path.join(REC, "labels.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["idx", "file", "t", "label"])
    for f in frames:
        w.writerow([f["idx"], f"{f['idx']:06d}.jpg", f["t"], labels[f["idx"]]])
print(f">> wrote labels.csv ({len(frames)} rows)", flush=True)
