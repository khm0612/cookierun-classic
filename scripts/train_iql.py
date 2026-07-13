"""Offline RL (IQL) on recorded runs — the first trainer here that can EXCEED imitation.

Every prior approach was behavioral-cloning-family (BC, AWR reweighting, unlikelihood
negatives, self-BC loops) and all of them plateaued at the demonstrator: BC cannot prefer
an action the data never rewarded. IQL learns VALUE from outcomes instead — "this state
leads to a hit in 0.8s" — using the reward signal the farm already produces:

  reward per frame:  -1.0 at a confirmed HP-hit (mined offline from the HP bar, same
                     detector as scripts/mine_negatives.py: >6% orange drop in 1.1s with
                     a 0.45s rebound-confirm), -5.0 terminal at the run-ending frame,
                     +0.01 per surviving frame.

Data: the bot's own recorded runs (demo_self_* — diverse, includes mistakes) MIXED with
the human demos (hf2,hf3,hf4 — near-optimal trajectories), states = the champion's exact
K-stack geometry, actions = the executed key at each frame (none/jump/slide spans).

IQL (Kostrikov et al. 2021): V via expectile regression (tau) toward min of twin target
Qs; Q toward r + gamma*V(s'); policy = advantage-weighted BC (exp(beta*A), clamped). The
policy net IS the small_cnn architecture, so the output checkpoint deploys through
LearnedAgent unchanged (arch small_cnn, no cond).

Usage: python scripts/train_iql.py [epochs] [--runs csv] [--encoder-init PATH]
       [--gamma 0.995] [--tau 0.7] [--beta 3.0] [--out-prefix iql]
       [--pit-r -4.0] [--pit-spread 1.0] [--pit-oversample 1]
"""
import os, json, sys, glob, time
from _runtime import DATA
import numpy as np, cv2
import torch, torch.nn as nn
import torch.nn.functional as F
from cookierun_bot.policies.learned import build_convs

BASE = str(DATA)
CLASSES = ["none", "jump", "slide"]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12


def _farg(flag, default):
    a = sys.argv[1:]
    for i, tok in enumerate(a):
        if tok == flag and i + 1 < len(a):
            return a[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


RUNS = _farg("--runs", None)
ENC = _farg("--encoder-init", None)
GAMMA = float(_farg("--gamma", 0.995))
TAU = float(_farg("--tau", 0.7))
BETA = float(_farg("--beta", 3.0))
OUT_PREFIX = _farg("--out-prefix", "iql")
HIT_R, DEATH_R, LIVE_R = -1.0, -5.0, 0.01
# PIT FALLS: the user's setup revive-tanks up to 3 falls, so a fall causes NO HP drop and
# NO terminal — without this explicit penalty the reward is blind to the exact failure the
# clean-run objective cares about most (why IQL-1 never learned pits). Falls are mined
# offline via the fixed-position "5 for 1 Pit Lift" revive-prompt template.
PIT_R = float(_farg("--pit-r", -4.0))
# Seconds before a detected fall over which PIT_R is spread. Applied at reward-construction
# time (below), NOT baked into cache_pits.npy — the cache stores raw pit frame indices, so
# one cache serves every spread value.
PIT_SPREAD = float(_farg("--pit-spread", 1.0))
# Duplicate the transitions inside pit-spread windows N times in the epoch index pool
# (1 = off). Q/V/policy all draw batches from that one pool, so the critics see the same
# oversampling as the actor — least-invasive for the GPU-resident gather pipeline.
PIT_OVERSAMPLE = max(int(_farg("--pit-oversample", 1)), 1)
torch.manual_seed(0)

meta = json.load(open(os.path.join(BASE, "demo", "model_meta.json")))
K, H, W = meta["K"], meta["H"], meta["W"]
CROP = meta.get("crop", [0.10, 0.20, 1.00, 0.90])
x0f, y0f, x1f, y1f = CROP

if RUNS:
    run_dirs = [os.path.join(BASE, r.strip()) for r in RUNS.split(",") if r.strip()]
else:
    run_dirs = sorted(d for d in glob.glob(os.path.join(BASE, "demo_self_*")) +
                      glob.glob(os.path.join(BASE, "botrun_*"))   # AIFARM_RECORD flywheel runs
                      if os.path.exists(os.path.join(d, "frames.json")))
    run_dirs += [os.path.join(BASE, r) for r in ("hf2", "hf3", "hf4")
                 if os.path.exists(os.path.join(BASE, r, "frames.json"))]
print(f"IQL corpus: {[os.path.basename(r) for r in run_dirs]}", flush=True)

# HP-bar ROI as fractions of the RAW 960x540 recording (mine_negatives.py calibration)
_HP_Y0, _HP_Y1, _HP_X0, _HP_X1 = 0.096, 0.141, 0.083, 0.823
# Pit-Lift revive prompt ROI (fractions; ai_farm.pitfall calibration off death frames)
_PIT_TPL = cv2.imread(os.path.join(BASE, "..", "templates", "pitlift_norm.png"),
                      cv2.IMREAD_GRAYSCALE)


def mine_pits(rdir, frames, ts):
    """Frame indices where a pit fall's revive prompt is visible (4s refractory), cached.
    Reads the COLOR originals — the prompt straddles the model crop's bottom edge."""
    cache = os.path.join(rdir, "cache_pits.npy")
    if os.path.exists(cache):
        pit_idx = np.load(cache)
        return pit_idx.tolist()
    if _PIT_TPL is None:
        print("  (no pitlift template — pit mining skipped)", flush=True)
        return []
    fdir = os.path.join(rdir, "frames")
    pit_idx, last = [], -9.0
    for i, fr in enumerate(frames):
        if ts[i] - last <= 4.0:
            continue
        im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"))
        if im is None:
            continue
        h, w = im.shape[:2]
        c = cv2.cvtColor(im[int(h * 0.830):int(h * 0.956), int(w * 0.372):int(w * 0.684)],
                         cv2.COLOR_BGR2GRAY)
        c = cv2.resize(c, (_PIT_TPL.shape[1], _PIT_TPL.shape[0]), interpolation=cv2.INTER_AREA)
        if float(cv2.matchTemplate(c, _PIT_TPL, cv2.TM_CCOEFF_NORMED)[0, 0]) >= 0.55:
            pit_idx.append(i)
            last = ts[i]
    np.save(cache, np.array(pit_idx, np.int64))
    return pit_idx


def mine_hp(rdir, frames):
    """Per-frame HP fraction (cached): needs the COLOR originals — the model-res gray
    caches crop the HP bar away."""
    cache = os.path.join(rdir, "cache_hp.npy")
    if os.path.exists(cache):
        hp = np.load(cache)
        if len(hp) == len(frames):
            return hp
    hp = np.zeros(len(frames), np.float32)
    fdir = os.path.join(rdir, "frames")
    for i, fr in enumerate(frames):
        im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"))
        if im is None:
            hp[i] = hp[i - 1] if i else 0.0
            continue
        h, w = im.shape[:2]
        strip = im[int(h * _HP_Y0):int(h * _HP_Y1), int(w * _HP_X0):int(w * _HP_X1)]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        hp[i] = float((cv2.inRange(hsv, np.array([0, 120, 120]),
                                   np.array([30, 255, 255])) > 0).mean())
    np.save(cache, hp)
    return hp


def hit_frames(ts, hp):
    """Confirmed-hit frame indices: >6% drop vs the 1.1s rolling max, still depressed
    0.45s later (rebound-confirm kills bonus-wash artifacts), 0.6s refractory."""
    hits, last = [], -9.0
    n = len(ts)
    j0 = 0
    for i in range(n):
        while ts[i] - ts[j0] > 1.1:
            j0 += 1
        rmax = hp[j0:i + 1].max() if i >= j0 else hp[i]
        if rmax - hp[i] > 0.06 and ts[i] - ts[0] > 4.0 and ts[i] - last > 0.6:
            k = i
            while k < n - 1 and ts[k] - ts[i] < 0.45:
                k += 1
            if hp[k] <= rmax - 0.05:
                hits.append(i)
                last = ts[i]
    return hits


imgs_all, act_all, rew_all, run_start, pit_win_all = [], [], [], [], []
offset = 0
for rdir in run_dirs:
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    ts = np.array([f["t"] for f in frames])
    n = len(frames)
    # model-res gray frames (reuse the SSL cache when present)
    _ctag = "-".join(f"{v:g}" for v in CROP)
    cache = os.path.join(rdir, f"cache_ssl_{H}x{W}_{_ctag}.npy")
    if os.path.exists(cache) and len(np.load(cache, mmap_mode="r")) == n:
        imgs = np.load(cache)
    else:
        imgs = np.zeros((n, H, W), np.uint8)
        fdir = os.path.join(rdir, "frames")
        for i, fr in enumerate(frames):
            im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            hh, ww = im.shape
            band = im[int(hh * y0f):int(hh * y1f), int(ww * x0f):int(ww * x1f)]
            imgs[i] = cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)
        np.save(cache, imgs)
    # executed action per frame: press spans [t, t+max(dur, 0.08)]
    act = np.zeros(n, np.int64)
    for k in keys:
        cls = CLASSES.index(k["action"])
        lo = np.searchsorted(ts, k["t"])
        hi = np.searchsorted(ts, k["t"] + max(float(k.get("dur", 0.0) or 0.0), 0.08))
        act[lo:hi] = cls
    # rewards
    hp = mine_hp(rdir, frames)
    rew = np.full(n, LIVE_R, np.float32)
    hidx = hit_frames(ts, hp)
    rew[hidx] += HIT_R
    pidx = mine_pits(rdir, frames, ts)
    # penalize the frames LEADING INTO the fall (the prompt shows ~0.5-1s after the miss;
    # the mistimed/missing jump happened ~0.5s earlier) so credit lands on the decision
    for pi in pidx:
        lo = np.searchsorted(ts, ts[pi] - PIT_SPREAD)
        rew[lo:pi + 1] += PIT_R / max(pi + 1 - lo, 1)
        pit_win_all.append(np.arange(lo, pi + 1, dtype=np.int64) + offset)
    rew[-1] += DEATH_R                     # run end = death (or the human stopped: close enough)
    print(f"  {os.path.basename(rdir)}: {n} frames | {len(hidx)} hits | {len(pidx)} PIT FALLS | "
          f"actions {dict(zip(CLASSES, np.bincount(act, minlength=3).tolist()))}", flush=True)
    imgs_all.append(imgs); act_all.append(act); rew_all.append(rew)
    run_start.extend([offset] * n)
    offset += n

imgs = np.concatenate(imgs_all)
act = np.concatenate(act_all)
rew = np.concatenate(rew_all)
run_start = np.array(run_start)
n = len(act)
# terminal mask: last frame of each run has no successor
is_term = np.zeros(n, bool)
is_term[np.unique(np.concatenate([np.where(np.diff(run_start))[0], [n - 1]]))] = True
del imgs_all
print(f"total {n} transitions | terminals {int(is_term.sum())}", flush=True)

if not torch.cuda.is_available():
    raise SystemExit("CUDA required")
dev = torch.device("cuda")
bank = torch.from_numpy(imgs).to(dev)
del imgs
ks = torch.arange(K - 1, -1, -1, dtype=torch.long, device=dev)
idx_all = torch.arange(n, dtype=torch.long, device=dev)
start_g = torch.from_numpy(run_start).to(dev)
act_g = torch.from_numpy(act).to(dev)
rew_g = torch.from_numpy(rew).to(dev)
term_g = torch.from_numpy(is_term).to(dev)


def stacks(i):
    """(B,K,H,W) float stacks for frame indices i, clamped at run starts."""
    idx = torch.maximum(i[:, None] - ks[None, :], start_g[i][:, None])
    return bank[idx].float().div_(255.0)


class Net(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        convs, c, h, w = build_convs(nn, {"K": K, "H": H, "W": W, "conv": meta["conv"]})
        self.convs = convs
        self.fc = nn.Linear(c * h * w, meta["fc"])
        self.head = nn.Linear(meta["fc"], out_dim)

    def forward(self, x):
        return self.head(torch.relu(self.fc(self.convs(x).flatten(1))))


def load_enc(net):
    if ENC and os.path.exists(ENC):
        net.convs.load_state_dict(torch.load(ENC, map_location="cpu")["convs"])


q1, q2, vf, pi = Net(3), Net(3), Net(1), Net(3)
for _n in (q1, q2, vf, pi):
    load_enc(_n)
    _n.to(dev)
q1t = Net(3).to(dev); q1t.load_state_dict(q1.state_dict())
q2t = Net(3).to(dev); q2t.load_state_dict(q2.state_dict())
opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 3e-4)
opt_v = torch.optim.Adam(vf.parameters(), 3e-4)
opt_p = torch.optim.Adam(pi.parameters(), 3e-4)
print(f"IQL: gamma={GAMMA} tau={TAU} beta={BETA} epochs={EPOCHS} "
      f"pit_spread={PIT_SPREAD} pit_oversample={PIT_OVERSAMPLE} "
      f"encoder={'ssl' if ENC else 'scratch'}", flush=True)

BATCH = 256
valid = idx_all[~term_g]                    # transitions with a successor
if PIT_OVERSAMPLE > 1 and pit_win_all:
    # np.unique: a frame in overlapping windows (possible when PIT_SPREAD > the 4s mining
    # refractory) still gets exactly N total copies
    extra = torch.from_numpy(np.unique(np.concatenate(pit_win_all))).to(dev)
    extra = extra[~term_g[extra]]           # same no-successor rule as `valid`
    valid = torch.cat([valid] + [extra] * (PIT_OVERSAMPLE - 1))
    print(f"pit oversample x{PIT_OVERSAMPLE}: {len(extra)} pre-fall transitions "
          f"duplicated -> pool {len(valid)}", flush=True)
for ep in range(EPOCHS):
    perm = valid[torch.randperm(len(valid), device=dev)]
    tot_q = tot_v = tot_p = 0.0
    nb = 0
    for b in range(0, len(perm), BATCH):
        i = perm[b:b + BATCH]
        s = stacks(i)
        a = act_g[i]
        r = rew_g[i]
        sp = stacks(i + 1)
        with torch.no_grad():
            vnext = vf(sp).squeeze(1)
            q_tgt = r + GAMMA * vnext
            qmin = torch.minimum(q1t(s).gather(1, a[:, None]).squeeze(1),
                                 q2t(s).gather(1, a[:, None]).squeeze(1))
        # V: expectile regression toward min target-Q
        u = qmin - vf(s).squeeze(1)
        lv = (torch.abs(TAU - (u < 0).float()) * u.pow(2)).mean()
        opt_v.zero_grad(); lv.backward(); opt_v.step()
        # Q: TD toward r + gamma V(s')
        lq = F.mse_loss(q1(s).gather(1, a[:, None]).squeeze(1), q_tgt) + \
             F.mse_loss(q2(s).gather(1, a[:, None]).squeeze(1), q_tgt)
        opt_q.zero_grad(); lq.backward(); opt_q.step()
        # policy: advantage-weighted BC
        with torch.no_grad():
            adv = qmin - vf(s).squeeze(1)
            wgt = torch.exp(BETA * adv).clamp(max=100.0)
        logp = F.log_softmax(pi(s), dim=1).gather(1, a[:, None]).squeeze(1)
        lp = -(wgt * logp).mean()
        opt_p.zero_grad(); lp.backward(); opt_p.step()
        # polyak targets
        with torch.no_grad():
            for tgt, src in ((q1t, q1), (q2t, q2)):
                for pt, ps_ in zip(tgt.parameters(), src.parameters()):
                    pt.mul_(0.995).add_(ps_, alpha=0.005)
        tot_q += lq.item(); tot_v += lv.item(); tot_p += lp.item(); nb += 1
    print(f"ep{ep + 1} q={tot_q / nb:.4f} v={tot_v / nb:.4f} pi={tot_p / nb:.4f}", flush=True)

# export the policy as a LearnedAgent-compatible small_cnn checkpoint
out_meta = dict(meta)
out_meta["arch"] = "small_cnn"
out_meta.pop("cond", None)
sd = {}
for k_, v_ in pi.convs.state_dict().items():
    sd[k_] = v_                            # "0.weight" ... conv prefix indices match
# small_cnn Sequential: convs(0..7) Flatten(8) Linear(9) ReLU Dropout Linear(12)
sd["9.weight"] = pi.fc.weight; sd["9.bias"] = pi.fc.bias
sd["12.weight"] = pi.head.weight; sd["12.bias"] = pi.head.bias
mp = os.path.join(BASE, "demo", f"{OUT_PREFIX}.pt")
json.dump(out_meta, open(os.path.join(BASE, "demo", f"{OUT_PREFIX}_meta.json"), "w"))
torch.save(sd, mp)
print(f">> saved {OUT_PREFIX}.pt (+meta) — deployable via LearnedAgent", flush=True)
