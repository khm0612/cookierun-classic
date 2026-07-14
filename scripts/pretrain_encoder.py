"""Self-supervised pretraining of the dodge-policy conv encoder on ALL recorded frames.

WHY: the policy head only has ~2 human demos to learn from, and a from-scratch encoder
overfits them fast (proven: winners peak ~ep25 and decline; deep-zone obstacles the bot
never survived to see are out-of-distribution). But the corpus of UNLABELED frames is
huge — human demos + every self-farm run. Pretraining the encoder on all of it teaches
robust obstacle/motion features without a single label; the thin demos then only have to
teach TIMING.

Pretext task: predict the frame ~100ms AHEAD from a masked K-stack (denoising + temporal
prediction). Predicting the future forces the encoder to represent scroll speed, obstacle
positions and object permanence — exactly the features the dodge head needs.

The encoder is the EXACT small_cnn conv trunk (geometry from --meta-from, default the
deployed model_meta.json), built via policies.learned.build_convs, so the pretrained
state_dict loads 1:1 into small_cnn or small_cnn_film via train2.py --encoder-init.

Also calibrates the scroll-speed normaliser (median/p90 px/sec over the corpus) that
train2.py writes into meta["cond"]["speed_norm"] for the FiLM models.

Usage:
  python scripts/pretrain_encoder.py [epochs] [--runs hf2,hf3,...] [--meta-from PATH]
                                     [--out PATH]
Defaults: 20 epochs, every data/<run>/ with frames.json (excluding *test*), geometry from
data/demo/model_meta.json, output data/demo/encoder_ssl.pt.
"""
import os, json, sys, glob, time
from _runtime import DATA, recording_is_complete
import numpy as np, cv2
import torch, torch.nn as nn
from cookierun_bot.policies.learned import build_convs
from cookierun_bot.policies import condition

BASE = str(DATA)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20


def _farg(flag, default):
    a = sys.argv[1:]
    for i, tok in enumerate(a):
        if tok == flag and i + 1 < len(a):
            return a[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


META_FROM = _farg("--meta-from", os.path.join(BASE, "demo", "model_meta.json"))
OUT_PATH = _farg("--out", os.path.join(BASE, "demo", "encoder_ssl.pt"))
RUN_NAMES = _farg("--runs", None)
HORIZON_S = float(_farg("--horizon", 0.10))     # predict this far ahead
BATCH = int(_farg("--batch", 256))
torch.manual_seed(0)

meta = json.load(open(META_FROM))
K, H, W = meta["K"], meta["H"], meta["W"]
CROP = meta.get("crop", [0.10, 0.20, 1.00, 0.90])
# ponytail: decoder below hard-mirrors 4x stride-2 stages; guard instead of generalising
assert len(meta["conv"]) == 4 and all(s == 2 for _, _, s in meta["conv"]) \
    and H % 16 == 0 and W % 16 == 0, "pretrainer expects the 4x stride-2 small_cnn trunk"

if RUN_NAMES:
    runs = [os.path.join(BASE, r.strip()) for r in RUN_NAMES.split(",") if r.strip()]
else:
    runs = sorted(d for d in glob.glob(os.path.join(BASE, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json"))
                  and "test" not in os.path.basename(d)
                  and not os.path.basename(d).startswith("_"))
missing = [r for r in runs if not os.path.exists(os.path.join(r, "frames.json"))]
if missing:
    raise SystemExit(f"missing frames.json: {missing}")
print(f"SSL corpus: {[os.path.basename(r) for r in runs]}", flush=True)

x0f, y0f, x1f, y1f = CROP
IMGS, TS = [], []
for rdir in runs:
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    if not recording_is_complete(fm):
        print(f"  {os.path.basename(rdir)}: incomplete recording, skipped", flush=True)
        continue
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    ts = np.array([f["t"] for f in frames])
    # crop is part of the key: same H/W with different crop fractions = different pixels,
    # and the length-only staleness check below cannot tell them apart
    _ctag = "-".join(f"{v:g}" for v in CROP)
    cache = os.path.join(rdir, f"cache_ssl_{H}x{W}_{_ctag}.npy")
    if os.path.exists(cache):
        imgs = np.load(cache)
        if len(imgs) != len(frames):               # stale cache (run dir changed)
            imgs = None
    else:
        imgs = None
    if imgs is None:
        t0 = time.time()
        imgs = np.zeros((len(frames), H, W), np.uint8)
        fdir = os.path.join(rdir, "frames")
        for i, fr in enumerate(frames):
            im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            h, w = im.shape
            band = im[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]
            imgs[i] = cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)
        np.save(cache, imgs)
        print(f"  cached {os.path.basename(rdir)}: {len(frames)} frames in "
              f"{time.time() - t0:.0f}s", flush=True)
    IMGS.append(imgs)
    TS.append(ts)

if not IMGS:
    raise SystemExit("no complete recordings to pretrain")

# ---- scroll-speed calibration over the whole corpus (px/sec at model res) --------------
t0 = time.time()
all_speeds = []
for imgs, ts in zip(IMGS, TS):
    sp = condition.run_speeds(ts, imgs, scroll_v=condition.SCROLL_V)
    all_speeds.append(sp[sp > 0])
all_speeds = np.concatenate(all_speeds) if all_speeds else np.array([1.0])
SPEED_MED = float(np.median(all_speeds))
SPEED_P90 = float(np.percentile(all_speeds, 90))
print(f"scroll speed px/sec @ {H}x{W}: median {SPEED_MED:.1f} | p90 {SPEED_P90:.1f} "
      f"({time.time() - t0:.0f}s)", flush=True)

# ---- anchors: i has a full K-history AND a ~HORIZON_S-ahead target within its run -------
anchors, targets, tr_anchor_mask = [], [], []
offset = 0
for imgs, ts in zip(IMGS, TS):
    n = len(ts)
    tgt = np.searchsorted(ts, ts + HORIZON_S)
    ii = np.arange(n)
    ok = (ii >= K - 1) & (tgt < n)
    dt = np.where(tgt < n, ts[np.minimum(tgt, n - 1)] - ts, 0)
    ok &= (dt >= HORIZON_S * 0.5) & (dt <= HORIZON_S * 2.5)
    ai = ii[ok]
    anchors.append(ai + offset)
    targets.append(tgt[ok] + offset)
    cut = int(n * 0.9)                             # last 10% of each run = val
    tr_anchor_mask.append(ai < cut)
    offset += n
anchors = np.concatenate(anchors)
targets = np.concatenate(targets)
tr_mask = np.concatenate(tr_anchor_mask)
tr_a, tr_t = anchors[tr_mask], targets[tr_mask]
va_a, va_t = anchors[~tr_mask], targets[~tr_mask]
print(f"anchors: {len(tr_a)} train | {len(va_a)} val", flush=True)

if not torch.cuda.is_available():
    raise SystemExit("CUDA required (the frame bank lives in VRAM).")
dev = torch.device("cuda")
bank_np = np.concatenate(IMGS)
del IMGS
bank_gb = bank_np.nbytes / 1e9
BANK_ON_GPU = bank_gb < 9.0
bank = torch.from_numpy(bank_np).to(dev) if BANK_ON_GPU else torch.from_numpy(bank_np)
del bank_np
print(f"frame bank: {bank.shape[0]} frames, {bank_gb:.1f} GB "
      f"({'VRAM' if BANK_ON_GPU else 'system RAM — gathered per batch'})", flush=True)
ks = torch.arange(K - 1, -1, -1, dtype=torch.long)


def gather_stacks(anchor_idx, target_idx):
    """(B,K,H,W) input float stacks + (B,1,H,W) target frames on the GPU."""
    idx = anchor_idx[:, None] - ks[None, :].to(anchor_idx.device)
    if BANK_ON_GPU:
        x = bank[idx].float().div_(255.0)
        yt = bank[target_idx].float().div_(255.0).unsqueeze(1)
    else:
        x = bank[idx.cpu()].to(dev, non_blocking=True).float().div_(255.0)
        yt = bank[target_idx.cpu()].to(dev, non_blocking=True).float().div_(255.0).unsqueeze(1)
    return x, yt


def mask_inputs(x):
    """Cutout 3 random rectangles (~H/5 x W/5) per batch, SAME region across the K
    channels (temporally consistent occlusion -> the net must use motion + context)."""
    B = x.shape[0]
    for _ in range(3):
        ch, cw = H // 5, W // 5
        yy = int(torch.randint(0, H - ch, (1,)).item())
        xx = int(torch.randint(0, W - cw, (1,)).item())
        x[:, :, yy:yy + ch, xx:xx + cw] = 0.0
    return x


convs, c_out, h_out, w_out = build_convs(nn, {"K": K, "H": H, "W": W, "conv": meta["conv"]})
decoder = nn.Sequential(
    nn.ConvTranspose2d(c_out, 64, 4, 2, 1), nn.ReLU(),
    nn.ConvTranspose2d(64, 48, 4, 2, 1), nn.ReLU(),
    nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.ReLU(),
    nn.ConvTranspose2d(24, 1, 4, 2, 1), nn.Sigmoid(),
)
net = nn.Sequential()  # container so one optimizer covers both
net.add_module("convs", convs)
net.add_module("decoder", decoder)
net.to(dev)
opt = torch.optim.Adam(net.parameters(), 1e-3)
print(f"encoder: conv{meta['conv']} -> ({c_out},{h_out},{w_out}) | horizon {HORIZON_S}s "
      f"| epochs {EPOCHS} | batch {BATCH}", flush=True)

tr_a_g = torch.from_numpy(tr_a).to(dev)
tr_t_g = torch.from_numpy(tr_t).to(dev)
va_a_g = torch.from_numpy(va_a).to(dev)
va_t_g = torch.from_numpy(va_t).to(dev)

best_val = float("inf")
best_state = None
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(tr_a), device=dev)
    tot, nb = 0.0, 0
    for b in range(0, len(tr_a), BATCH):
        sel = perm[b:b + BATCH]
        x, yt = gather_stacks(tr_a_g[sel], tr_t_g[sel])
        x = mask_inputs(x)
        opt.zero_grad()
        pred = decoder(convs(x))
        loss = torch.nn.functional.l1_loss(pred, yt)
        loss.backward()
        opt.step()
        tot += loss.item(); nb += 1
    net.eval()
    vtot, vnb = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(va_a), 512):
            x, yt = gather_stacks(va_a_g[b:b + 512], va_t_g[b:b + 512])
            vtot += torch.nn.functional.l1_loss(decoder(convs(x)), yt).item(); vnb += 1
    vl = vtot / max(vnb, 1)
    print(f"ep{ep + 1} train L1={tot / max(nb, 1):.4f} val L1={vl:.4f}", flush=True)
    if vl < best_val:
        best_val = vl
        best_state = {k: v.detach().cpu().clone() for k, v in convs.state_dict().items()}

enc_meta = {
    "conv": meta["conv"], "K": K, "H": H, "W": W, "crop": CROP,
    "horizon_s": HORIZON_S, "epochs": EPOCHS, "best_val_l1": best_val,
    "speed_med": SPEED_MED, "speed_p90": SPEED_P90,
    "runs": [os.path.basename(r) for r in runs],
}
torch.save({"convs": best_state, "meta": enc_meta}, OUT_PATH)
print(f">> saved encoder (best val L1 {best_val:.4f}) + speed stats to {OUT_PATH}", flush=True)
