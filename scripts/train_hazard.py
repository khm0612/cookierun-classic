"""M4 hazard head — learn to SEE a pit before the fall, the one thing gating/imitation can't add.

M1.1 forensics proved the dominant failure is BLINDNESS: 41/46 no-jump falls have the policy's
jump-confidence at ~0 over the whole pre-fall window. So the no-human path to fewer falls is a
detector: a small binary head on the SAME conv trunk as the policy, supervised by the mined
falls ("a pit-lift prompt occurs within HAZARD_S"), warm-started from iql3's learned obstacle
features. This script only MEASURES whether the pixels carry the signal (held-out precision/
recall on a run-level split). If they don't separate, only human demos remain; if they do, the
head gets wired as a jump trigger (see docs/MILESTONES.md M4).

Usage:
    python scripts/train_hazard.py [epochs] [--hazard-s 1.5] [--enc-init iql3]
                                   [--val-frac 0.25] [--out hazard]
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from _runtime import DATA
from cookierun_bot.policies.learned import build_convs

BASE = str(DATA)
_RUN_FPS = {}                                # per-run measured capture fps (filled by load_run)


def _farg(flag, default):
    a = sys.argv[1:]
    for i, t in enumerate(a):
        if t == flag and i + 1 < len(a):
            return a[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return default


EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15
HAZARD_S = float(_farg("--hazard-s", 1.5))
ENC_INIT = _farg("--enc-init", "iql3")
VAL_FRAC = float(_farg("--val-frac", 0.25))
OUT = _farg("--out", "hazard")
# --focal G (default 0.0): focal-loss modulation of the per-sample BCE; 0.0 == plain BCE.
FOCAL = float(_farg("--focal", 0.0))
# --augment (flag, default off): train-only brightness jitter + 1-in-K temporal drop.
AUGMENT = "--augment" in sys.argv[1:]
torch.manual_seed(0)
np.random.seed(0)

meta = json.load(open(os.path.join(BASE, "demo", "model_meta.json")))
# force the standard policy geometry so the on-disk SSL caches line up + enc transfer works
gmeta = json.load(open(os.path.join(BASE, "demo", f"{ENC_INIT}_meta.json")))
K, H, W = int(gmeta["K"]), int(gmeta["H"]), int(gmeta["W"])
CROP = gmeta.get("crop", [0.1, 0.2, 1.0, 0.9])
CTAG = "-".join(f"{v:g}" for v in CROP)


def load_run(rdir):
    cache = os.path.join(rdir, f"cache_ssl_{H}x{W}_{CTAG}.npy")
    fj = os.path.join(rdir, "frames.json")
    cp = os.path.join(rdir, "cache_pits.npy")
    if not (os.path.exists(cache) and os.path.exists(fj) and os.path.exists(cp)):
        return None
    fm = json.load(open(fj))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    imgs = np.load(cache)
    if len(imgs) != len(frames):
        return None
    ts = np.array([f["t"] for f in frames], np.float64)
    if len(ts) > 1:                          # true capture fps for the false-fire-rate metric (#9)
        _RUN_FPS[os.path.basename(rdir)] = float((len(ts) - 1) / max(ts[-1] - ts[0], 1e-6))
    pits = np.load(cp).astype(int)
    # label = 1 for frames within HAZARD_S BEFORE a pit-lift prompt (the approach window)
    y = np.zeros(len(frames), np.float32)
    for p in pits:
        lo = np.searchsorted(ts, ts[p] - HAZARD_S)
        y[lo:p + 1] = 1.0
    return imgs, y, len(pits)


def build_dataset(dirs):
    """Return per-run (imgs, y). K-stacks are built on the fly (stride 1 = live stacking)."""
    runs = []
    for d in dirs:
        r = load_run(d)
        if r is None:
            continue
        imgs, y, npit = r
        if npit == 0:
            # keep a few pit-free runs as pure negatives, but not the whole 65k-frame demos
            if imgs.shape[0] > 20000:
                continue
        runs.append((os.path.basename(d), imgs, y))
    return runs


class HazardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs, c, h, w = build_convs(nn, gmeta)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * h * w, 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.convs(x)).squeeze(1)


def warm_start(net):
    """Load iql3's conv trunk into the head's encoder (obstacle features transfer 1:1)."""
    try:
        sd = torch.load(os.path.join(BASE, "demo", f"{ENC_INIT}.pt"), map_location="cpu")
        conv_sd = {k: v for k, v in sd.items() if k[0].isdigit() and int(k.split(".")[0]) <
                   len(net.convs)}
        net.convs.load_state_dict({k: v for k, v in conv_sd.items()}, strict=False)
        print(f"warm-started encoder from {ENC_INIT}.pt ({len(conv_sd)} tensors)", flush=True)
    except Exception as e:
        print(f"warm-start skipped ({e}) — training encoder from scratch", flush=True)


def stacks(imgs, idx):
    ks = np.arange(K - 1, -1, -1)
    ind = np.maximum(idx[:, None] - ks[None, :], 0)
    return imgs[ind]


def augment_stacks(xb, rng):
    """Train-only, shape-preserving aug on a (B,K,H,W) float tensor already in [0,1].

    ONLY called under `if AUGMENT` so the default path draws ZERO rng and returns the
    exact input tensor. Consumes rng in a fixed order: temporal-drop mask, drop slots,
    brightness factors — all from the script's shared rng so runs stay deterministic.
    """
    B, Kc = int(xb.shape[0]), int(xb.shape[1])
    # 1-in-K temporal drop: repeat the previous frame into one random slot (per sample).
    if Kc >= 2:
        drop = rng.random(B) < (1.0 / Kc)
        slots = rng.integers(1, Kc, size=B)  # slot in [1, K-1] -> copies slot s-1
        for bi in range(B):
            if drop[bi]:
                s = int(slots[bi])
                xb[bi, s] = xb[bi, s - 1]
    # brightness jitter: per-sample x * U(0.9, 1.1), clamped back to [0, 1].
    fac = torch.from_numpy(rng.uniform(0.9, 1.1, size=B).astype(np.float32)).to(xb.device)
    return (xb * fac[:, None, None, None]).clamp_(0.0, 1.0)


def main():
    print(f"[hazard] epochs={EPOCHS} hazard_s={HAZARD_S} enc_init={ENC_INIT} "
          f"val_frac={VAL_FRAC} out={OUT} focal={FOCAL} augment={AUGMENT}", flush=True)
    dirs = sorted(glob.glob(os.path.join(BASE, "botrun_*")) +
                  glob.glob(os.path.join(BASE, "demo_self_*")) +
                  [os.path.join(BASE, r) for r in ("hf2", "hf3", "hf4")])
    runs = build_dataset(dirs)
    # run-level split (no frame leakage): hold out whole runs, ensuring val has pits
    pit_runs = [r for r in runs if r[2].sum() > 0]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pit_runs))
    n_val = max(2, int(len(pit_runs) * VAL_FRAC))
    val_names = {pit_runs[i][0] for i in order[:n_val]}
    train = [r for r in runs if r[0] not in val_names]
    val = [r for r in runs if r[0] in val_names]
    tot_pos = int(sum(r[2].sum() for r in runs))
    print(f"runs: {len(runs)} ({len(train)} train / {len(val)} val)  "
          f"positive frames: {tot_pos}  val-run pits: "
          f"{int(sum(r[2].sum() for r in val))}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HazardNet().to(dev)
    warm_start(net)

    # build flat index pools (run_id, frame_i); subsample negatives to 4:1
    def pool(subset, neg_ratio=4):
        pos, neg = [], []
        for ri, (_, imgs, y) in enumerate(subset):
            p = np.where(y > 0.5)[0]
            n = np.where(y < 0.5)[0]
            pos += [(ri, i) for i in p]
            neg += [(ri, i) for i in n]
        neg = [neg[i] for i in rng.permutation(len(neg))[:max(1, len(pos) * neg_ratio)]]
        return pos, neg

    tr_pos, tr_neg = pool(train)
    print(f"train frames: {len(tr_pos)} pos / {len(tr_neg)} neg", flush=True)
    pos_weight = torch.tensor([len(tr_neg) / max(len(tr_pos), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)

    tr_idx = tr_pos + tr_neg
    for ep in range(EPOCHS):
        net.train()
        perm = rng.permutation(len(tr_idx))
        tot = 0.0
        for b in range(0, len(perm), 256):
            bi = [tr_idx[j] for j in perm[b:b + 256]]
            xs = np.concatenate([stacks(train[ri][1], np.array([i])) for ri, i in bi])
            ys = np.array([train[ri][2][i] for ri, i in bi], np.float32)
            xb = torch.from_numpy(xs).to(dev).float().div_(255.0)
            if AUGMENT:  # default off -> xb untouched, zero rng drawn
                xb = augment_stacks(xb, rng)
            yb = torch.from_numpy(ys).to(dev)
            opt.zero_grad()
            if FOCAL == 0.0:  # default -> EXACT original BCEWithLogitsLoss(pos_weight)
                loss = lossf(net(xb), yb)
            else:
                logits = net(xb)
                # per-sample BCE with the SAME pos_weight balancing as lossf (reduction='none'
                # then .mean() is bit-identical to reduction='mean'); scale each by (1-p_t)^G.
                bce = F.binary_cross_entropy_with_logits(
                    logits, yb, pos_weight=pos_weight, reduction="none")
                p = torch.sigmoid(logits)
                p_t = p * yb + (1.0 - p) * (1.0 - yb)  # predicted prob of the true class
                loss = (bce * (1.0 - p_t).pow(FOCAL)).mean()
            loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        # ---- val: precision/recall at thresholds + AP ----
        net.eval()
        vp, vy = [], []
        with torch.no_grad():
            for _, imgs, y in val:
                idx = np.arange(len(y))
                for b in range(0, len(idx), 512):
                    xb = torch.from_numpy(stacks(imgs, idx[b:b + 512])).to(dev).float().div_(255.0)
                    vp.append(torch.sigmoid(net(xb)).cpu().numpy())
                vy.append(y)
        p = np.concatenate(vp); yv = np.concatenate(vy)
        line = f"ep{ep+1:02d} loss={tot/max(len(tr_idx),1):.4f}"
        for thr in (0.5, 0.7, 0.9):
            fire = p >= thr
            tp = int((fire & (yv > 0.5)).sum()); fp = int((fire & (yv < 0.5)).sum())
            fn = int((~fire & (yv > 0.5)).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            line += f" | @{thr}: P={prec:.2f} R={rec:.2f}"
        # average precision (area under P-R), the threshold-free separability number
        order2 = np.argsort(-p)
        ys = yv[order2]; tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
        recall = tps / max(ys.sum(), 1); precision = tps / np.maximum(tps + fps, 1)
        _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
        ap = float(_trap(precision, recall)) if ys.sum() else 0.0
        base = float(yv.mean())
        print(line + f" | AP={ap:.3f} (base rate {base:.3f}, lift {ap/max(base,1e-6):.1f}x)",
              flush=True)

    # ---- deployment metrics on held-out runs: PER-PIT recall + false-fire bursts/min ----
    # (frame P/R hides what matters live: did the head fire >=1x in each pit's approach window,
    #  and how often does it burst-fire on safe ground?)
    net.eval()
    print("\n[held-out DEPLOYMENT metrics — per-pit recall + false-fire bursts/min]", flush=True)
    for thr in (0.7, 0.9, 0.97):
        pits_hit = pits_tot = false_bursts = safe_min = 0
        with torch.no_grad():
            for name, imgs, y in val:
                idx = np.arange(len(y))
                pr = []
                for b in range(0, len(idx), 512):
                    xb = torch.from_numpy(stacks(imgs, idx[b:b+512])).to(dev).float().div_(255.0)
                    pr.append(torch.sigmoid(net(xb)).cpu().numpy())
                pr = np.concatenate(pr)
                fire = pr >= thr
                # per-pit: a pit's approach window is its contiguous run of y==1
                d = np.diff(np.concatenate([[0], (y > 0.5).astype(int), [0]]))
                starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
                for a, b in zip(starts, ends):
                    pits_tot += 1
                    if fire[a:b].any():
                        pits_hit += 1
                # false bursts = fire RISING EDGES that begin on safe ground (bug hunt #10: the
                # old `fire & safe` rising-edge counted the safe-side tail of a CORRECT on-pit
                # detection as a fresh alarm; count only edges where fire turns ON while safe).
                safe = (y < 0.5)
                fstart = np.where(np.diff(fire.astype(int)) == 1)[0] + 1
                false_bursts += int(safe[fstart].sum()) if len(fstart) else 0
                # true per-run fps (bug hunt #9: hardcoded 50 biased the rate when capture fps
                # differed, e.g. dxcam ~80 vs the CPU path); _RUN_FPS is filled in load_run.
                safe_min += safe.sum() / _RUN_FPS.get(name, 50.0) / 60.0
        print(f"  @{thr}: per-pit recall {pits_hit}/{pits_tot} "
              f"({100*pits_hit/max(pits_tot,1):.0f}%)  |  false-fire {false_bursts/max(safe_min,1e-6):.1f} bursts/min",
              flush=True)

    torch.save(net.state_dict(), os.path.join(BASE, "demo", f"{OUT}.pt"))
    json.dump({"arch": "hazard", "K": K, "H": H, "W": W, "crop": CROP,
               "conv": gmeta["conv"], "hazard_s": HAZARD_S, "enc_init": ENC_INIT,
               "focal": FOCAL, "augment": AUGMENT},
              open(os.path.join(BASE, "demo", f"{OUT}_meta.json"), "w"))
    print(f">> saved {OUT}.pt (+meta)", flush=True)


if __name__ == "__main__":
    main()
