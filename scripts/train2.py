"""Behavioral cloning v3 — multi-run training over all recorded demos.
Improvements over v2 (single-run):
  * trains on EVERY data/demo* recording (frame stacks never cross run boundaries)
  * "not-yet" weighting: none-frames BEFORE a press are upsampled — live analysis showed
    the model fires ~0.78s early, i.e. exactly the frames where it must learn "obstacle
    visible but do NOT act yet"
  * DAgger corrections: labels from scripts/correct.py (the bot's own logged failure
    moments, human-corrected) are mixed into training at high weight — they sit on the
    model's OWN failure distribution, which demo frames never cover
  * validation = the last 15% of EACH run (time-ordered, unseen; corrections never enter val)
Saves model.pt + model_meta.json to data/demo (the LearnedAgent load path).

Usage: python scripts/train2.py [EPOCHS]     |     train2.py --check-corr  (dry-run loader)"""
import os, json, sys, time, glob
from _runtime import DATA
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from cookierun_bot.policies.learned import build_net_from_meta

BASE = str(DATA)
OUT = os.path.join(BASE, "demo")               # model.pt destination (LearnedAgent path)
HITS = os.path.join(BASE, "ai_hits")
CLASSES = ["none", "jump", "slide"]
CHECK_CORR = "--check-corr" in sys.argv[1:]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
torch.manual_seed(0)

META = {
    "classes": CLASSES,
    "K": 4, "H": 96, "W": 224,
    "crop": [0.10, 0.20, 1.00, 0.90],
    "fps": 35.0,
    "conv": [[24, 5, 2], [48, 3, 2], [64, 3, 2], [64, 3, 2]],
    "fc": 256,
    # label window / not-yet weighting = the SWEEP WINNER (J_win25_ny4, 2026-07-04):
    # win 0.25 + not-yet 4.0 scored 0.366 vs 0.152 for the old 0.15/2.5 defaults. Keep
    # train2 in sync with the deployed winner so a plain retrain can't silently regress.
    "win_pre": 0.25, "win_post": 0.03,
    "notyet_lo": 0.25, "notyet_hi": 0.70, "notyet_w": 4.0,
    "corr_w": 10.0,        # sampling boost for DAgger correction samples
}


def load_corrections(meta) -> "tuple[np.ndarray, np.ndarray] | tuple[None, None]":
    """Load correction labels written by scripts/correct.py into (stacks, labels).
    Each record labels the pre03 frame (~0.3s before impact — human dodge timing).
    If the record has >=K k-frame snapshots (newer batches), a true K-stack at training
    fps spacing is built; otherwise the single frame is replicated K times (stationary
    stack — degraded but still teaches WHICH obstacle/position wants WHICH action).
    'skip' labels are ignored. Returns (None, None) when there are no usable records."""
    fp = os.path.join(HITS, "corrections.jsonl")
    if not os.path.exists(fp):
        return None, None
    x0f, y0f, x1f, y1f = meta["crop"]
    Hh, Ww, Kk = meta["H"], meta["W"], meta["K"]

    def band(path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        h, w = im.shape
        return cv2.resize(im[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)],
                          (Ww, Hh), interpolation=cv2.INTER_AREA)

    stacks, labels, dropped = [], [], 0
    for line in open(fp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            dropped += 1
            continue
        if r.get("label") not in ("jump", "slide", "none"):
            continue
        kpaths = [os.path.join(HITS, k) for k in r.get("kimgs", [])]
        kimgs = [band(p) for p in kpaths]
        kimgs = [k for k in kimgs if k is not None]
        if len(kimgs) >= Kk:
            st = np.stack(kimgs[-Kk:])             # oldest->newest, ends at labeled frame
        else:
            base_img = band(os.path.join(HITS, r["img"]))
            if base_img is None:
                dropped += 1
                continue
            st = np.stack([base_img] * Kk)
        stacks.append(st.astype(np.uint8))
        labels.append(CLASSES.index(r["label"]))
    if dropped:
        print(f"corrections: dropped {dropped} unreadable record(s)/image(s)", flush=True)
    if not stacks:
        return None, None
    return np.stack(stacks), np.array(labels, np.int64)


if CHECK_CORR:
    cs, cy = load_corrections(META)
    if cs is None:
        print("corrections: none found (label some with scripts/correct.py first)")
    else:
        print(f"corrections: {len(cy)} usable | per class "
              f"{dict(zip(CLASSES, np.bincount(cy, minlength=3).tolist()))} "
              f"| stack shape {cs.shape}")
    sys.exit(0)

runs = sorted(d for d in glob.glob(os.path.join(BASE, "demo*"))
              if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json"))
              and "test" not in os.path.basename(d))
print("runs:", [os.path.basename(r) for r in runs], flush=True)

x0f, y0f, x1f, y1f = META["crop"]
H, W, K = META["H"], META["W"], META["K"]
imgs_all, y_all, notyet_all, run_id, run_start = [], [], [], [], []
tr_ids, va_ids = [], []
offset = 0
for ri, rdir in enumerate(runs):
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    ts = np.array([f["t"] for f in frames])
    y = np.zeros(len(frames), np.int64)
    notyet = np.zeros(len(frames), bool)
    for k in keys:
        cls = CLASSES.index(k["action"])
        lo = np.searchsorted(ts, k["t"] - META["win_pre"])
        hi = np.searchsorted(ts, k["t"] + META["win_post"])
        y[lo:hi] = cls
        nlo = np.searchsorted(ts, k["t"] - META["notyet_hi"])
        nhi = np.searchsorted(ts, k["t"] - META["notyet_lo"])
        notyet[nlo:nhi] = True
    notyet &= (y == 0)                        # only none-frames get the boost
    print(f"  {os.path.basename(rdir)}: {len(frames)} frames, {len(keys)} keys, "
          f"labels {dict(zip(CLASSES, np.bincount(y, minlength=3).tolist()))}, "
          f"not-yet {int(notyet.sum())}", flush=True)
    t0 = time.time()
    imgs = np.zeros((len(frames), H, W), np.uint8)
    fdir = os.path.join(rdir, "frames")
    for i, fr in enumerate(frames):
        im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        h, w = im.shape
        band = im[int(h*y0f):int(h*y1f), int(w*x0f):int(w*x1f)]
        imgs[i] = cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)
    print(f"    loaded in {time.time()-t0:.0f}s", flush=True)
    imgs_all.append(imgs); y_all.append(y); notyet_all.append(notyet)
    run_id.extend([ri] * len(frames)); run_start.extend([offset] * len(frames))
    cut = offset + int(len(frames) * 0.85)
    tr_ids.extend(range(offset, cut)); va_ids.extend(range(cut, offset + len(frames)))
    offset += len(frames)

imgs = np.concatenate(imgs_all); y = np.concatenate(y_all)
notyet = np.concatenate(notyet_all)
run_start = np.array(run_start); n = len(y)
print(f"total {n} frames | train {len(tr_ids)} | val {len(va_ids)}", flush=True)

def stack(i):
    lo = run_start[i]                          # never stack across a run boundary
    idxs = [max(lo, i - k) for k in range(K - 1, -1, -1)]
    return imgs[idxs].astype(np.float32) / 255.0

# DAgger corrections (scripts/correct.py) — train-only extra samples, never in val
corr_x, corr_y = load_corrections(META)
n_corr = 0 if corr_y is None else len(corr_y)
if n_corr:
    print(f"corrections: {n_corr} mixed in at weight x{META['corr_w']} | per class "
          f"{dict(zip(CLASSES, np.bincount(corr_y, minlength=3).tolist()))}", flush=True)

class DS(Dataset):
    """Demo frames by index, then correction stacks appended at the tail (train only)."""
    def __init__(self, ids, with_corr=False):
        self.ids = ids
        self.n_corr = n_corr if with_corr else 0
    def __len__(self): return len(self.ids) + self.n_corr
    def __getitem__(self, j):
        if j < len(self.ids):
            i = self.ids[j]
            return torch.from_numpy(stack(i)), int(y[i])
        k = j - len(self.ids)
        return torch.from_numpy(corr_x[k].astype(np.float32) / 255.0), int(corr_y[k])

counts = np.bincount(y[tr_ids], minlength=3)
print("train class counts:", dict(zip(CLASSES, counts.tolist())), flush=True)
w_cls = 1.0 / np.sqrt(np.maximum(counts, 1))
w = w_cls[y[tr_ids]].astype(np.float64)
w[notyet[tr_ids]] *= META["notyet_w"]          # "obstacle coming but do NOT act yet"
if n_corr:                                     # corrections: class weight x corr boost
    w = np.concatenate([w, w_cls[corr_y] * META["corr_w"]])
samp = WeightedRandomSampler(torch.tensor(w), num_samples=len(w), replacement=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
net = build_net_from_meta(torch, META).to(dev)
opt = torch.optim.Adam(net.parameters(), 1e-3)
lossf = nn.CrossEntropyLoss()
trl = DataLoader(DS(tr_ids, with_corr=True), batch_size=128, sampler=samp)
val = DataLoader(DS(va_ids), batch_size=256)      # val = pure demo frames, no corrections

def predict_val():
    net.eval(); pr = []
    with torch.no_grad():
        for xb, _ in val:
            pr.append(torch.softmax(net(xb.to(dev)), 1).cpu().numpy())
    return np.concatenate(pr)

def event_eval(conf=0.60):
    p = predict_val(); pred = p.argmax(1); fire = (pred != 0) & (p.max(1) > conf)
    yv = y[va_ids]
    events, i = [], 0
    while i < len(yv):
        if yv[i] != 0:
            j = i
            while j + 1 < len(yv) and yv[j + 1] == yv[i]: j += 1
            events.append((i, j, yv[i])); i = j + 1
        else: i += 1
    hits = sum(1 for a, b, c in events if np.any(fire[a:b+1] & (pred[a:b+1] == c)))
    fam = (fire & (yv == 0)).mean() * 35 * 60   # false-fire frames/min at 35fps
    return len(events), hits, fam

for ep in range(EPOCHS):
    net.train(); tot = 0
    for xb, yb in trl:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step(); tot += l.item()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        ne, hits, fam = event_eval()
        print(f"ep{ep+1} loss={tot/len(trl):.3f} events {hits}/{ne} hit, "
              f"false-fires/min={fam:.0f}", flush=True)

torch.save(net.state_dict(), os.path.join(OUT, "model.pt"))
json.dump(META, open(os.path.join(OUT, "model_meta.json"), "w"))
ne, hits, fam = event_eval()
p = predict_val(); pred = p.argmax(1)
cm = np.zeros((3, 3), int)
for t_, p_ in zip(y[va_ids], pred): cm[t_, p_] += 1
print("val confusion (rows=true):", cm.tolist(), flush=True)
print(f">> saved model.pt | events {hits}/{ne} | false-fires/min {fam:.0f}", flush=True)
