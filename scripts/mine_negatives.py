"""Mine AUTOMATIC negative examples from bot self-runs for the "hit cam" training signal.

Idea (user's): the human demo is the POSITIVE signal (imitate correct play); the bot's own
runs are the NEGATIVE signal — specifically the moments it got HIT (HP bar drops) or produced a
detector-confirmed PIT revive prompt. At those pre-hit frames the bot was passive ("none") and paid
for it, so we teach the model to be LESS passive there. This script detects those events from
the recorded frames and dumps the pre-hit K-stacks, built with the SAME crop + consecutive-frame
geometry train2.py uses for positives (so the negatives are in-distribution).

Output: an .npz with `stacks` (M,K,H,W uint8, oldest->newest), `kind` ('hit'|'pit'), `bot_action`
(what the bot did at the decision frame), `run`, `t`. train2.py --neg-npz consumes it and adds an
unlikelihood loss that pushes p(none) DOWN on these frames.

HP bar: read as the orange-saturated pixel fraction in a fixed fractional strip (works at any
save resolution). A HIT = a >HP_DROP fall within ~1.1s (the slow continuous health decay never
trips this; only a sudden knock does). PIT labels require recorder evidence; uncertain terminal
frames are deliberately not guessed.
"""
import os, sys, json, glob
from _runtime import DATA
import numpy as np, cv2

BASE = str(DATA)


def _farg(flag, default):
    a = sys.argv[1:]
    for i, tok in enumerate(a):
        if tok == flag and i + 1 < len(a):
            return a[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


RUN_NAMES = [r.strip() for r in _farg("--runs", "").split(",") if r.strip()]
META_FROM = _farg("--meta-from", os.path.join(BASE, "demo", "model_meta.json"))
OUT = _farg("--out", os.path.join(BASE, "ai_hits", "auto_negatives.npz"))
HP_DROP = float(_farg("--hp-drop", "0.06"))     # min HP fall within the window to count a hit
PRE_S = float(_farg("--pre", "0.30"))           # stack ends this far BEFORE impact (the decision)
PIT_PROMPT_LAG_S = float(_farg("--pit-prompt-lag", "0.75"))
HITWIN_S = 1.1                                   # window over which the drop is measured
MIN_GAP_S = 0.6                                  # dedupe: >= this many seconds between hits
HP_LO_T = float(_farg("--hp-lo", "0.06"))        # below this = bar basically empty (skip: refills/noise)


def _pit_times(metadata):
    """Return only detector-confirmed pit timestamps stored by the recorder."""
    if metadata.get("complete") is False:
        return []
    duration = float(metadata.get("duration_s", 0.0) or 0.0)
    frame_times = []
    for frame in metadata.get("frames", []):
        try:
            frame_times.append(float(frame["t"]))
        except (KeyError, TypeError, ValueError):
            continue
    latest_frame = min(duration, max(frame_times)) if frame_times else duration
    result = []
    for value in metadata.get("pit_times", []):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= latest_frame:
            result.append(value)
    return result


def _decision_time(event_time, kind):
    """The revive prompt appears after the miss; move pit stacks back before the ledge."""
    return event_time - PRE_S - (PIT_PROMPT_LAG_S if kind == "pit" else 0.0)


def _default_run_names(base):
    dirs = glob.glob(os.path.join(base, "demo_self_*")) + glob.glob(os.path.join(base, "botrun_*"))
    return sorted(os.path.basename(d) for d in dirs
                  if os.path.exists(os.path.join(d, "frames.json")))


meta = json.load(open(META_FROM))
K, H, W = int(meta["K"]), int(meta["H"]), int(meta["W"])
x0f, y0f, x1f, y1f = meta["crop"]
print(f"meta: K{K} H{H} W{W} crop{meta['crop']} from {META_FROM}", flush=True)

if not RUN_NAMES:
    RUN_NAMES = _default_run_names(BASE)
print(f"neg runs ({len(RUN_NAMES)}): {RUN_NAMES}", flush=True)


def hp_frac(bgr):
    """Orange-saturated fraction in the HP-bar strip (fractional ROI => resolution-independent)."""
    h, w = bgr.shape[:2]
    strip = bgr[int(0.096 * h):int(0.141 * h), int(0.083 * w):int(0.823 * w)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255]))
    return float((m > 0).mean())


def band_gray(idx, fdir):
    im = cv2.imread(os.path.join(fdir, f"{idx:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape
    b = im[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]
    return cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)


def bot_action_at(keys, t):
    """What the bot was doing at time t: an active slide-hold, an ~instant jump, else 'none'."""
    for k in keys:
        tp = k["t"]; dur = float(k.get("dur", 0.0) or 0.0)
        if k["action"] == "slide" and dur > 0 and tp <= t <= tp + dur:
            return "slide"
        if abs(t - tp) <= 0.05:
            return k["action"]
    return "none"


stacks, kinds, bacts, sruns, stimes = [], [], [], [], []
for rn in RUN_NAMES:
    rdir = os.path.join(BASE, rn)
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    if fm.get("complete") is False:
        print(f"  {rn}: incomplete recording, skipped", flush=True)
        continue
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    idxs = np.array([f["idx"] for f in frames])
    ts = np.array([f["t"] for f in frames])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    fdir = os.path.join(rdir, "frames")

    # pass 1: HP over the run (sample every 3rd frame for speed; hits span ~150ms so this is ample)
    step = 3
    hp_t, hp_v = [], []
    for j in range(0, len(frames), step):
        im = cv2.imread(os.path.join(fdir, f"{frames[j]['idx']:06d}.jpg"))
        if im is None:
            continue
        hp_t.append(ts[j]); hp_v.append(hp_frac(im))
    hp_t = np.array(hp_t); hp_v = np.array(hp_v)
    if len(hp_v) < 5:
        print(f"  {rn}: unreadable, skipped", flush=True); continue

    # detect hits: a sudden fall vs the max of the trailing window
    hit_times, last = [], -9.0
    for j in range(len(hp_v)):
        lo = np.searchsorted(hp_t, hp_t[j] - HITWIN_S)
        rmax = hp_v[lo:j + 1].max()
        if (rmax - hp_v[j] > HP_DROP and hp_v[j] > HP_LO_T
                and hp_t[j] > 4.0 and hp_t[j] - last > MIN_GAP_S):
            last = hp_t[j]; hit_times.append(float(hp_t[j]))

    # A long in-play ending is not evidence of a pit: HP deaths, time caps, and manual stops
    # look the same here. Only use explicit detector events written by ai_farm.
    pit_times = _pit_times(fm)

    def add_event(t_impact, kind):
        t_dec = _decision_time(t_impact, kind)
        end = int(np.searchsorted(ts, t_dec))            # frame nearest the decision moment
        end = min(max(end, 0), len(frames) - 1)
        # K consecutive frames ending at `end`, clamped to run start (matches train2 idx_mat)
        sel = [max(end - (K - 1 - i), 0) for i in range(K)]   # oldest -> newest
        st = []
        for s in sel:
            g = band_gray(idxs[s], fdir)
            if g is None:
                return False
            st.append(g)
        stacks.append(np.stack(st).astype(np.uint8))
        kinds.append(kind)
        bacts.append(bot_action_at(keys, t_dec))
        sruns.append(rn); stimes.append(round(t_impact, 1))
        return True

    nh = sum(add_event(t, "hit") for t in hit_times)
    npit = sum(add_event(t, "pit") for t in pit_times)
    print(f"  {rn}: {nh} hits + {npit} pit (from {len(hp_v)} hp samples)", flush=True)

if not stacks:
    raise SystemExit("no negatives mined — check HP ROI / thresholds")

X = np.stack(stacks)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, stacks=X, kind=np.array(kinds), bot_action=np.array(bacts),
                    run=np.array(sruns), t=np.array(stimes))
from collections import Counter
print(f">> mined {len(X)} negatives -> {OUT}", flush=True)
print(f"   kinds: {dict(Counter(kinds))} | bot action at decision: {dict(Counter(bacts))} | "
      f"stack {X.shape}", flush=True)
