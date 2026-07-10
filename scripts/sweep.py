"""Hyperparameter sweep for the imitation model on the two recorded demos (fixed data —
user won't record more for now, so squeeze the knobs). Loads + preprocesses each run ONCE
at the largest resolution, then trains each config with best-epoch checkpointing (the last
epoch is often not the best) and reports event-hit rate + false-fires/min on the held-out
tails at several confidence gates. Winner is saved to data/demo/model.pt (+meta).

Spacing note: frame-stack spacing is expressed through meta['fps'] — LearnedAgent stacks
at 1/fps seconds, so a config's fps = REC_FPS/stride == stride-N over the recording with NO
inference-code changes needed. REC_FPS is MEASURED from the demos' real timestamps (below),
so it tracks the recorder cadence instead of a stale hardcoded rate.
"""
import os, json, sys, time, glob, copy
from _runtime import DATA
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from cookierun_bot.policies.learned import build_net_from_meta

BASE = str(DATA)
OUT = os.path.join(BASE, "demo")
CLASSES = ["none", "jump", "slide"]
CROP = [0.10, 0.20, 1.00, 0.90]
BIG_H, BIG_W = 120, 280            # cache at the largest swept res; downscale per config
REC_FPS = 35.0
EPOCHS = 30
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- load all runs once ------------------------------------------------
runs = sorted(d for d in glob.glob(os.path.join(BASE, "demo*"))
              if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json"))
              and "test" not in os.path.basename(d))
print("runs:", [os.path.basename(r) for r in runs], flush=True)
x0f, y0f, x1f, y1f = CROP
IMGS, TS, KEYS, run_start, tr_ids, va_ids = [], [], [], [], [], []
offset = 0
for rdir in runs:
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    ts = np.array([f["t"] for f in frames])
    cache = os.path.join(rdir, f"cache_{BIG_H}x{BIG_W}.npy")
    if os.path.exists(cache):
        imgs = np.load(cache)
    else:
        t0 = time.time()
        imgs = np.zeros((len(frames), BIG_H, BIG_W), np.uint8)
        fdir = os.path.join(rdir, "frames")
        for i, fr in enumerate(frames):
            im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            h, w = im.shape
            band = im[int(h*y0f):int(h*y1f), int(w*x0f):int(w*x1f)]
            imgs[i] = cv2.resize(band, (BIG_W, BIG_H), interpolation=cv2.INTER_AREA)
        np.save(cache, imgs)
        print(f"  cached {os.path.basename(rdir)} in {time.time()-t0:.0f}s", flush=True)
    IMGS.append(imgs); TS.append(ts); KEYS.append(keys)
    run_start.extend([offset] * len(frames))
    cut = offset + int(len(frames) * 0.85)
    tr_ids.extend(range(offset, cut)); va_ids.extend(range(cut, offset + len(frames)))
    offset += len(frames)
IMGS_BIG = np.concatenate(IMGS)
run_start = np.array(run_start)
tr_ids = np.array(tr_ids); va_ids = np.array(va_ids)
print(f"total {len(IMGS_BIG)} frames", flush=True)
# Recording cadence is the single source of truth for frame-stack spacing: measure it from
# the demos' real timestamps rather than assuming 35fps (the recorder now saves ~60fps). Each
# config's meta['fps'] = REC_FPS/stride then matches its index-strided training spacing, so
# LearnedAgent gates the live stack at the same span it trained on (no OOD drift).
_dts = np.concatenate([np.diff(ts) for ts in TS if len(ts) > 1]) if TS else np.array([])
if len(_dts):
    REC_FPS = round(1.0 / float(np.median(_dts)), 1)
    print(f"measured recording fps: {REC_FPS}", flush=True)

def build_labels(win_pre, win_post, ny_lo, ny_hi):
    y = np.zeros(len(IMGS_BIG), np.int64)
    ny = np.zeros(len(IMGS_BIG), bool)
    off = 0
    for ts, keys in zip(TS, KEYS):
        for k in keys:
            cls = CLASSES.index(k["action"])
            lo = off + np.searchsorted(ts, k["t"] - win_pre)
            hi = off + np.searchsorted(ts, k["t"] + win_post)
            y[lo:hi] = cls
            nlo = off + np.searchsorted(ts, k["t"] - ny_hi)
            nhi = off + np.searchsorted(ts, k["t"] - ny_lo)
            ny[nlo:nhi] = True
        off += len(ts)
    ny &= (y == 0)
    return y, ny

def run_config(name, H, W, K, stride, win_pre, ny_w, ny_zone, epochs=EPOCHS, aug=False):
    y, ny = build_labels(win_pre, 0.03, ny_zone[0], ny_zone[1])
    if (H, W) == (BIG_H, BIG_W):
        imgs = IMGS_BIG
    else:
        imgs = np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in IMGS_BIG])
    meta = {"classes": CLASSES, "K": K, "H": H, "W": W, "crop": CROP,
            "fps": REC_FPS / stride, "conv": [[24, 5, 2], [48, 3, 2], [64, 3, 2], [64, 3, 2]],
            "fc": 256, "win_pre": win_pre, "win_post": 0.03,
            "notyet_lo": ny_zone[0], "notyet_hi": ny_zone[1], "notyet_w": ny_w}

    def stack(i):
        lo = run_start[i]
        idxs = [max(lo, i - k * stride) for k in range(K - 1, -1, -1)]
        return imgs[idxs].astype(np.float32) / 255.0

    class DS(Dataset):
        def __init__(self, ids, train=False): self.ids, self.train = ids, train
        def __len__(self): return len(self.ids)
        def __getitem__(self, j):
            i = self.ids[j]
            x = stack(i)
            if aug and self.train:
                # zone-robustness: the game's zones swing bright<->dark; random gain/bias
                # teaches brightness invariance (the old CV detectors' weakest axis)
                g = np.float32(np.random.uniform(0.7, 1.3))
                b = np.float32(np.random.uniform(-0.10, 0.10))
                x = np.clip(x * g + b, 0.0, 1.0)
            return torch.from_numpy(x), int(y[i])

    counts = np.bincount(y[tr_ids], minlength=3)
    w = (1.0 / np.sqrt(np.maximum(counts, 1)))[y[tr_ids]].astype(np.float64)
    w[ny[tr_ids]] *= ny_w
    samp = WeightedRandomSampler(torch.tensor(w), num_samples=len(tr_ids), replacement=True)
    net = build_net_from_meta(torch, meta).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    trl = DataLoader(DS(tr_ids, train=True), batch_size=128, sampler=samp)
    val = DataLoader(DS(va_ids), batch_size=256)
    yv = y[va_ids]
    events, i = [], 0
    while i < len(yv):
        if yv[i] != 0:
            j = i
            while j + 1 < len(yv) and yv[j + 1] == yv[i]: j += 1
            events.append((i, j, yv[i])); i = j + 1
        else: i += 1

    def evaluate():
        net.eval(); pr = []
        with torch.no_grad():
            for xb, _ in val:
                pr.append(torch.softmax(net(xb.to(dev)), 1).cpu().numpy())
        p = np.concatenate(pr)
        best = None
        for conf in (0.5, 0.6, 0.7):
            pred = p.argmax(1); fire = (pred != 0) & (p.max(1) > conf)
            hits = sum(1 for a, b, c in events if np.any(fire[a:b+1] & (pred[a:b+1] == c)))
            fam = (fire & (yv == 0)).mean() * REC_FPS * 60
            score = hits / max(len(events), 1) - fam / 400.0
            if best is None or score > best[0]:
                best = (score, conf, hits, fam)
        return best

    best_ck = None
    for ep in range(epochs):
        net.train()
        for xb, yb in trl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
        if ep >= 9 and (ep % 5 == 4 or ep == epochs - 1):
            score, conf, hits, fam = evaluate()
            if best_ck is None or score > best_ck[0][0]:
                best_ck = ((score, conf, hits, fam), ep + 1,
                           copy.deepcopy(net.state_dict()))
    (score, conf, hits, fam), ep, sd = best_ck
    print(f"[{name}] BEST ep{ep} conf={conf}: events {hits}/{len(events)} "
          f"({hits/len(events)*100:.0f}%), false/min {fam:.0f}, score {score:.3f}", flush=True)
    return {"name": name, "score": score, "conf": conf, "hits": hits,
            "events": len(events), "fam": fam, "ep": ep, "meta": meta, "state": sd}

CONFIGS_R1 = [
    ("A_base",        96, 224, 4, 1, 0.15, 2.5, (0.15, 0.60)),
    ("B_win25",       96, 224, 4, 1, 0.25, 2.5, (0.25, 0.70)),
    ("C_win35",       96, 224, 4, 1, 0.35, 2.5, (0.35, 0.80)),
    ("D_stride2",     96, 224, 4, 2, 0.15, 2.5, (0.15, 0.60)),
    ("E_K6",          96, 224, 6, 1, 0.15, 2.5, (0.15, 0.60)),
    ("F_ny4",         96, 224, 4, 1, 0.15, 4.0, (0.15, 0.60)),
    ("G_nywide",      96, 224, 4, 1, 0.15, 2.5, (0.20, 0.80)),
    ("H_res120",     120, 280, 4, 1, 0.15, 2.5, (0.15, 0.60)),
    ("I_win25_str2",  96, 224, 4, 2, 0.25, 2.5, (0.25, 0.70)),
    ("J_win25_ny4",   96, 224, 4, 1, 0.25, 4.0, (0.25, 0.70)),
    ("K_K6_win25",    96, 224, 6, 1, 0.25, 2.5, (0.25, 0.70)),
    ("L_res120_win25",120, 280, 4, 1, 0.25, 2.5, (0.25, 0.70)),
]
# round 2: fine grid around round-1 winner J (win25/ny4) + augmentation + capacity combos
CONFIGS_R2 = [
    ("R2_J_repro",    96, 224, 4, 1, 0.25, 4.0, (0.25, 0.70), EPOCHS, False),
    ("R2_win20_ny4",  96, 224, 4, 1, 0.20, 4.0, (0.20, 0.65), EPOCHS, False),
    ("R2_win30_ny4",  96, 224, 4, 1, 0.30, 4.0, (0.30, 0.75), EPOCHS, False),
    ("R2_ny6",        96, 224, 4, 1, 0.25, 6.0, (0.25, 0.70), EPOCHS, False),
    ("R2_J_aug",      96, 224, 4, 1, 0.25, 4.0, (0.25, 0.70), EPOCHS, True),
    ("R2_J_aug_K6",   96, 224, 6, 1, 0.25, 4.0, (0.25, 0.70), EPOCHS, True),
    ("R2_J_aug_r120",120, 280, 4, 1, 0.25, 4.0, (0.25, 0.70), EPOCHS, True),
]
CONFIGS = CONFIGS_R2 if (len(sys.argv) > 1 and sys.argv[1] == "r2") else CONFIGS_R1

results = []
for cfg in CONFIGS:
    t0 = time.time()
    r = run_config(*cfg)
    r["secs"] = time.time() - t0
    results.append(r)

results.sort(key=lambda r: -r["score"])
print("\n===== LEADERBOARD =====", flush=True)
for r in results:
    print(f"{r['name']:>16}: score {r['score']:.3f} | {r['hits']}/{r['events']} events "
          f"| {r['fam']:.0f} false/min | conf {r['conf']} | ep{r['ep']}", flush=True)
win = results[0]
runset = sorted(os.path.basename(r) for r in runs)
# Never overwrite a better deployed model — BUT only when the comparison is valid. score
# depends on the demo set (val tail + event set = the last 15% of whatever demos exist), so a
# score from a DIFFERENT run set isn't comparable. If the demos changed (e.g. you added some —
# the #1 lever), deploy this round's winner instead of gating on a stale, incomparable number.
prev_path = os.path.join(OUT, "sweep_results.json")
prev_score = -1e9
prev_runset = None
if os.path.exists(prev_path):
    prev = json.load(open(prev_path))
    prev_runset = prev.get("runs")
    if prev.get("board"):
        prev_score = prev["board"][0].get("score", -1e9)
comparable = prev_runset is not None and sorted(prev_runset) == runset
if comparable and win["score"] <= prev_score:
    print(f"\n>> round winner {win['name']} (score {win['score']:.3f}) does NOT beat the "
          f"deployed model (score {prev_score:.3f}) on the same demo set — keeping it.", flush=True)
else:
    if prev_runset is not None and not comparable:
        print(f"\n>> demo set changed since last sweep (was {sorted(prev_runset)}, now "
              f"{runset}); scores aren't comparable — deploying this round's winner.", flush=True)
    torch.save(win["state"], os.path.join(OUT, "model.pt"))
    json.dump(win["meta"], open(os.path.join(OUT, "model_meta.json"), "w"))
    json.dump({"winner": win["name"], "conf": win["conf"], "runs": runset,
               "board": [{k: r[k] for k in ("name", "score", "hits", "events", "fam", "conf", "ep")}
                         for r in results]},
              open(os.path.join(OUT, "sweep_results.json"), "w"), indent=1)
    print(f"\n>> WINNER {win['name']} saved to model.pt (deploy conf={win['conf']})", flush=True)
