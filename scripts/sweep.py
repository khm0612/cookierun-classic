"""Hyperparameter sweep for the imitation model on the recorded demos (fixed data —
squeeze the knobs). Loads + preprocesses each run ONCE at the largest resolution, then
trains each config with best-epoch checkpointing (the last epoch is often not the best)
and reports event-hit rate + false-fires/min on the held-out tails at several confidence
gates. Winner is saved to data/demo/model.pt (+meta).

Spacing note: frame-stack spacing is expressed through meta['fps'] — LearnedAgent stacks
at 1/fps seconds, so a config's fps = REC_FPS/stride == stride-N over the recording with NO
inference-code changes needed. REC_FPS is MEASURED from the demos' real timestamps (below),
so it tracks the recorder cadence instead of a stale hardcoded rate.
"""
import os, json, sys, time, glob, copy
from _runtime import DATA, ROOT

# live per-epoch training log for the dashboard (sweep_dash.py tails this)
PROG = str(ROOT / "sweep_progress.jsonl")
def _prog(o):
    try:
        with open(PROG, "a") as _f:
            _f.write(json.dumps(o) + "\n")
    except Exception:
        pass

# optional Weights & Biases logging: `--wandb` -> one W&B run per config, grouped as one sweep
# (`--wandb-project=`, `--wandb-mode=offline`, `--wandb-group=` to override). Needs `wandb login`.
USE_WANDB = "--wandb" in sys.argv
WANDB_PROJECT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--wandb-project=")), "cookierun-sweep")
WANDB_MODE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--wandb-mode=")), None)
WANDB_GROUP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--wandb-group=")), None)
wb = None
if USE_WANDB:
    import wandb as wb
    WANDB_GROUP = WANDB_GROUP or f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from cookierun_bot.policies.learned import build_net_from_meta

BASE = str(DATA)
HR = len(sys.argv) > 1 and sys.argv[1] == "hr"        # high-res mode: train at 240x560 (VRAM max)
HIFPS = len(sys.argv) > 1 and sys.argv[1] == "hifps"  # 60fps / high-K experiment on hf* demos
OUT = os.path.join(BASE, "_hifps_model" if HIFPS else "demo")   # keep the deployed 35fps model untouched
os.makedirs(OUT, exist_ok=True)
CLASSES = ["none", "jump", "slide"]
CROP = [0.10, 0.20, 1.00, 0.90]
BIG_H, BIG_W = (240, 560) if HR else (120, 280)  # cache/bank at the largest swept res
REC_FPS = 60.0 if HIFPS else 35.0                # hf* demos are recorded at 60fps
EPOCHS = 30
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- load all runs once ------------------------------------------------
# 60fps demos live in an `hf*` namespace so the normal `demo*` sweep never mixes frame rates
_prefix = "hf" if HIFPS else "demo"
runs = sorted(d for d in glob.glob(os.path.join(BASE, _prefix + "*"))
              if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json"))
              and "test" not in os.path.basename(d)
              and "self" not in os.path.basename(d))   # self-farm runs are for self_farm.py only
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
# Move the ENTIRE frame bank into VRAM and free system RAM (was ~1.6GB of numpy). Every config
# gathers its K-stacks straight from this GPU tensor; other resolutions are GPU-resized once.
IMGS_BIG_G = torch.from_numpy(IMGS_BIG).to(dev)
del IMGS_BIG, IMGS
import gc as _gc; _gc.collect()
print(f"total {IMGS_BIG_G.shape[0]} frames | frame bank on GPU: "
      f"{IMGS_BIG_G.element_size() * IMGS_BIG_G.nelement() / 1e6:.0f} MB VRAM, 0 in system RAM",
      flush=True)

# Recording cadence is the single source of truth for frame-stack spacing: measure it from
# the demos' real timestamps rather than assuming 35fps (the recorder now saves ~60fps). Each
# config's meta['fps'] = REC_FPS/stride then matches its index-strided training spacing, so
# LearnedAgent gates the live stack at the same span it trained on (no OOD drift).
_dts = np.concatenate([np.diff(ts) for ts in TS if len(ts) > 1]) if TS else np.array([])
if len(_dts):
    REC_FPS = round(1.0 / float(np.median(_dts)), 1)
    print(f"measured recording fps: {REC_FPS}", flush=True)

import torch.nn.functional as _F
_resize_cache: dict = {}


def get_imgs_g(H, W):
    """The frame bank at (H,W) as a uint8 GPU tensor. BIG res is the bank itself; other sizes are
    GPU-resized once (area interp ~ cv2 INTER_AREA) and cached, in chunks to avoid a huge float
    temporary. Nothing ever lands in system RAM."""
    if (H, W) == (BIG_H, BIG_W):
        return IMGS_BIG_G
    if (H, W) not in _resize_cache:
        out = torch.empty((IMGS_BIG_G.shape[0], H, W), dtype=torch.uint8, device=dev)
        for c in range(0, IMGS_BIG_G.shape[0], 4096):
            chunk = IMGS_BIG_G[c:c + 4096].unsqueeze(1).float()
            r = _F.interpolate(chunk, size=(H, W), mode="area")
            out[c:c + 4096] = r.squeeze(1).round_().clamp_(0, 255).to(torch.uint8)
        _resize_cache[(H, W)] = out
    return _resize_cache[(H, W)]

def build_labels(win_pre, win_post, ny_lo, ny_hi):
    y = np.zeros(IMGS_BIG_G.shape[0], np.int64)
    ny = np.zeros(IMGS_BIG_G.shape[0], bool)
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

BATCH = 128 if HR else 256   # smaller batch at high-res (240x560) to stay within 16GB VRAM


def run_config(name, H, W, K, stride, win_pre, ny_w, ny_zone, epochs=EPOCHS, aug=False):
    y, ny = build_labels(win_pre, 0.03, ny_zone[0], ny_zone[1])
    meta = {"classes": CLASSES, "K": K, "H": H, "W": W, "crop": CROP,
            "fps": REC_FPS / stride, "conv": [[24, 5, 2], [48, 3, 2], [64, 3, 2], [64, 3, 2]],
            "fc": 256, "win_pre": win_pre, "win_post": 0.03,
            "notyet_lo": ny_zone[0], "notyet_hi": ny_zone[1], "notyet_w": ny_w}

    # ---- GPU-RESIDENT data path ----------------------------------------------------------
    # The frame bank already lives in VRAM (get_imgs_g); the K-stack index matrix and labels go
    # there too, so every train/eval batch is a pure GPU gather + forward -- no CPU DataLoader,
    # no host->device copy, and NOTHING in system RAM. Result math is unchanged.
    imgs_g = get_imgs_g(H, W)                                               # (N,H,W) uint8, on GPU
    N = imgs_g.shape[0]
    ks = np.arange(K - 1, -1, -1)                                           # oldest..newest
    idx_mat = np.maximum(np.arange(N)[:, None] - ks[None, :] * stride, run_start[:, None])
    idx_g = torch.from_numpy(idx_mat).to(dev)                               # (N,K) long
    y_g = torch.from_numpy(y).to(dev)                                       # (N,) long
    tr_g = torch.from_numpy(np.asarray(tr_ids)).to(dev)
    va_g = torch.from_numpy(np.asarray(va_ids)).to(dev)

    def batch_stacks(frame_ids):                                           # -> (B,K,H,W) float
        return imgs_g[idx_g[frame_ids]].float().div_(255.0)

    counts = np.bincount(y[tr_ids], minlength=3)
    w = (1.0 / np.sqrt(np.maximum(counts, 1)))[y[tr_ids]].astype(np.float64)
    w[ny[tr_ids]] *= ny_w
    w_t = torch.tensor(w, dtype=torch.float32, device=dev)                 # weights over tr_ids

    net = build_net_from_meta(torch, meta).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()

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
            for b in range(0, len(va_ids), 512):
                pr.append(torch.softmax(net(batch_stacks(va_g[b:b + 512])), 1).cpu().numpy())
        p = np.concatenate(pr)
        best = None
        # search HIGHER confidences too (0.8-0.9): at 60fps the false-fire penalty scales with fps,
        # so a higher deploy conf that trades a few missed events for far fewer spurious fires wins.
        for conf in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
            pred = p.argmax(1); fire = (pred != 0) & (p.max(1) > conf)
            hits = sum(1 for a, b, c in events if np.any(fire[a:b+1] & (pred[a:b+1] == c)))
            fam = (fire & (yv == 0)).mean() * REC_FPS * 60
            score = hits / max(len(events), 1) - fam / 400.0
            if best is None or score > best[0]:
                best = (score, conf, hits, fam)
        return best

    wrun = None
    if wb is not None:
        wrun = wb.init(project=WANDB_PROJECT, group=WANDB_GROUP, name=name, reinit=True,
                       mode=WANDB_MODE, tags=[f"K{K}", f"win{win_pre}", ("aug" if aug else "noaug")],
                       config={"K": K, "H": H, "W": W, "stride": stride, "win_pre": win_pre,
                               "ny_w": ny_w, "aug": aug, "epochs": epochs, "fps": REC_FPS,
                               "window_ms": round(1000 * K * stride / REC_FPS),
                               "runs": [os.path.basename(r) for r in runs]})

    n_tr = len(tr_ids)
    best_ck = None
    for ep in range(epochs):
        net.train()
        sampled = tr_g[torch.multinomial(w_t, n_tr, replacement=True)]     # weighted w/ replacement
        last_l = None
        for b in range(0, n_tr, BATCH):
            bi = sampled[b:b + BATCH]
            xb = batch_stacks(bi)
            if aug:
                # per-sample brightness gain/bias on GPU (zone bright<->dark invariance)
                g = torch.empty(bi.shape[0], 1, 1, 1, device=dev).uniform_(0.7, 1.3)
                bs = torch.empty(bi.shape[0], 1, 1, 1, device=dev).uniform_(-0.10, 0.10)
                xb = (xb * g + bs).clamp_(0.0, 1.0)
            opt.zero_grad(); l = lossf(net(xb), y_g[bi]); l.backward(); opt.step()
            last_l = l
        # one GPU->CPU sync per epoch (not per batch) for the live loss curve
        eloss = round(float(last_l), 4)
        _prog({"t": "epoch", "config": name, "K": K, "epoch": ep + 1, "epochs": epochs, "loss": eloss})
        wlog = {"train_loss": eloss}
        if ep >= 9 and (ep % 5 == 4 or ep == epochs - 1):
            score, conf, hits, fam = evaluate()
            _prog({"t": "eval", "config": name, "K": K, "epoch": ep + 1,
                   "score": round(float(score), 4), "hits": int(hits),
                   "events": len(events), "fam": round(float(fam), 1)})
            wlog.update({"val_score": round(float(score), 4), "eval_conf": conf,
                         "hit_pct": round(hits / max(len(events), 1), 3), "false_per_min": round(float(fam), 1)})
            if best_ck is None or score > best_ck[0][0]:
                sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                best_ck = ((score, conf, hits, fam), ep + 1, sd)
        if wrun is not None:
            wrun.log(wlog, step=ep + 1)
    (score, conf, hits, fam), ep, sd = best_ck
    print(f"[{name}] BEST ep{ep} conf={conf}: events {hits}/{len(events)} "
          f"({hits/len(events)*100:.0f}%), false/min {fam:.0f}, score {score:.3f}", flush=True)
    if wrun is not None:
        wrun.summary.update({"best_score": score, "best_epoch": ep, "best_conf": conf,
                             "best_hit_pct": round(hits / max(len(events), 1), 3),
                             "best_false_per_min": round(float(fam), 1)})
        wrun.finish()
    del idx_g, y_g, tr_g, va_g, w_t, net       # per-config tensors (imgs_g is a shared VRAM cache)
    torch.cuda.empty_cache()
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
# HIGH-RES set: all 240x560 (== BIG, so no GPU-resize), sweeping the tuning knobs at max res
CONFIGS_HR = [
    ("HR_base",      240, 560, 4, 1, 0.15, 2.5, (0.15, 0.60)),
    ("HR_win25",     240, 560, 4, 1, 0.25, 2.5, (0.25, 0.70)),
    ("HR_ny4",       240, 560, 4, 1, 0.15, 4.0, (0.15, 0.60)),
    ("HR_K6",        240, 560, 6, 1, 0.15, 2.5, (0.15, 0.60)),
    ("HR_win25_ny4", 240, 560, 4, 1, 0.25, 4.0, (0.25, 0.70)),
]
# 60fps / high-K experiment. At 60fps the K-stack window = K/60 s, so this sweeps K to test
# the user's hypothesis directly ON THE SAME 60fps data: K4=67ms (control), K7=117ms (matches
# the winning 35fps/K4 window of 114ms), K10=167ms, plus win/ny/stride variants around it.
# TUNED re-sweep (2026-07-06): the first hifps sweep's best was the LONGEST window (K10) but all
# configs had high false-fires; ny4 was WORST (more spam). So push the window LONGER (K8-K14 @60fps
# = 133-233ms) at the default ny (2.5), keep win25/win30, and let evaluate() pick a HIGH deploy conf
# (up to 0.9) to cut the false-fire rate. No ny4 (it increased spam).
# NEW-SETUP, hf2-ONLY optimization (2026-07-06): the user won't record more, so squeeze ONE demo.
# With a single demo AUGMENTATION is the decisive lever — without it the net memorizes hf2 (the
# no-aug K8 scored 0.087 w/ 199 false/min). Sweep K8/K10/K12 windows WITH aug (+one no-aug baseline);
# evaluate() picks a high deploy conf (<=0.9) to hold false-fires down. FINAL ranking is live SURVIVAL.
# hf2-only, WIN_PRE sweep (2026-07-06 ~04:00): the win30 winner over-fired (137 jumps+116 slides/117s
# = high false/min) and jumped EARLY. win_pre is the label window BEFORE the human press: SMALLER =
# tighter labels = fewer/cleaner fires AND fires LATER (closer to the obstacle). So sweep win_pre
# 0.10-0.20 at the winning K8/K10 windows, all augmented. Ranked FINALLY on live SURVIVAL.
CONFIGS_HIFPS = [
    ("HF2_K10_win20_aug", 96, 224, 10, 1, 0.20, 2.5, (0.20, 0.65), EPOCHS, True),
    ("HF2_K10_win15_aug", 96, 224, 10, 1, 0.15, 2.5, (0.15, 0.60), EPOCHS, True),
    ("HF2_K10_win10_aug", 96, 224, 10, 1, 0.10, 2.5, (0.10, 0.55), EPOCHS, True),
    ("HF2_K8_win20_aug",  96, 224,  8, 1, 0.20, 2.5, (0.20, 0.65), EPOCHS, True),
    ("HF2_K8_win15_aug",  96, 224,  8, 1, 0.15, 2.5, (0.15, 0.60), EPOCHS, True),
    ("HF2_K10_win30_aug", 96, 224, 10, 1, 0.30, 2.5, (0.30, 0.75), EPOCHS, True),  # prev winner (control)
]
# round 3 (2026-07-05): "optimize what we have" — trains on all THREE 35fps demos (demo2+3+4,
# incl. the 19-min/82-slide demo4 = 2.6x B_win25's data) and tests the 60fps-sweep lead that the
# model wants a LONGER temporal window than the K4=114ms champ. K5/K6/K7 @35fps = 143/171/200ms,
# at the proven win25/ny2.5 recipe, + augmentation on the top-K candidates. Far less noisy than
# the 1-demo 60fps test. (window ms = K / 35fps.)
CONFIGS_R3 = [
    ("R3_K4_win25",  96, 224, 4, 1, 0.25, 2.5, (0.25, 0.70)),               # 114ms — current champ (control)
    ("R3_K5_win25",  96, 224, 5, 1, 0.25, 2.5, (0.25, 0.70)),               # 143ms
    ("R3_K6_win25",  96, 224, 6, 1, 0.25, 2.5, (0.25, 0.70)),               # 171ms — the 60fps lead
    ("R3_K7_win25",  96, 224, 7, 1, 0.25, 2.5, (0.25, 0.70)),               # 200ms (longest)
    ("R3_K5_win30",  96, 224, 5, 1, 0.30, 2.5, (0.30, 0.75)),               # longer window + wider label
    ("R3_K4_aug",    96, 224, 4, 1, 0.25, 2.5, (0.25, 0.70), EPOCHS, True), # champ + augmentation
    ("R3_K6_aug",    96, 224, 6, 1, 0.25, 2.5, (0.25, 0.70), EPOCHS, True), # longer window + aug
]
if HIFPS:
    CONFIGS = CONFIGS_HIFPS
elif HR:
    CONFIGS = CONFIGS_HR
elif len(sys.argv) > 1 and sys.argv[1] == "r2":
    CONFIGS = CONFIGS_R2
elif len(sys.argv) > 1 and sys.argv[1] == "r3":
    CONFIGS = CONFIGS_R3
else:
    CONFIGS = CONFIGS_R1

# optional GPU-parallel sharding: `sweep.py <mode> shard <i> <n>` trains CONFIGS[i::n] so N
# shards fill VRAM concurrently. Sharded runs skip deploy (save best to data/_shard_i.pt);
# scripts/sweep_par.py launches the shards, then merges + deploys the overall winner.
SHARD = None
if "shard" in sys.argv:
    _si = sys.argv.index("shard")
    SHARD = (int(sys.argv[_si + 1]), int(sys.argv[_si + 2]))

CONFIGS_FULL = CONFIGS
if SHARD is not None:
    CONFIGS = CONFIGS[SHARD[0]::SHARD[1]]      # this shard's slice of the grid
else:
    open(PROG, "w").close()                    # single-process: fresh log (sweep_par truncates for shards)
_prog({"t": "start", "total": len(CONFIGS_FULL), "configs": [c[0] for c in CONFIGS_FULL],
       "runs": [os.path.basename(r) for r in runs], "epochs": EPOCHS})
results = []
for cfg in CONFIGS:
    t0 = time.time()
    r = run_config(*cfg)
    r["secs"] = time.time() - t0
    results.append(r)

if not results:                        # empty shard slice (more shards than configs)
    if SHARD is not None:
        torch.save({"win": None, "board": []}, os.path.join(BASE, f"_shard_{SHARD[0]}.pt"))
    print(">> no configs for this shard — done", flush=True)
    raise SystemExit
results.sort(key=lambda r: -r["score"])
print("\n===== LEADERBOARD =====", flush=True)
for r in results:
    print(f"{r['name']:>16}: score {r['score']:.3f} | {r['hits']}/{r['events']} events "
          f"| {r['fam']:.0f} false/min | conf {r['conf']} | ep{r['ep']}", flush=True)
win = results[0]
_board = [{k: r[k] for k in ("name", "score", "hits", "events", "fam", "conf", "ep")} for r in results]
if SHARD is not None:
    # a shard doesn't deploy — it saves its slice's best (+ weights) for sweep_par to merge
    torch.save({"win": {k: win[k] for k in ("name", "score", "conf", "hits", "events", "fam", "ep")},
                "state": win["state"], "meta": win["meta"], "board": _board},
               os.path.join(BASE, f"_shard_{SHARD[0]}.pt"))
    print(f"\n>> SHARD {SHARD[0]}/{SHARD[1]} done — slice best {win['name']} ({win['score']:.3f}); "
          f"sweep_par.py merges + deploys", flush=True)
else:
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
        json.dump({"winner": win["name"], "conf": win["conf"], "runs": runset, "board": _board},
                  open(os.path.join(OUT, "sweep_results.json"), "w"), indent=1)
        print(f"\n>> WINNER {win['name']} saved to model.pt (deploy conf={win['conf']})", flush=True)
