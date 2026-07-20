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
       [--mask-hits 0.0] [--human-weight 1.0] [--min-quality]   # M2 selective imitation
       [--nstep 1] [--cql-alpha 0.0]                            # M3 value-side (n-step + CQL)
"""
import os, json, sys, glob, time
from _runtime import DATA, recording_is_complete
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
# --- M2 selective imitation (all default to NO-OP so iql3 behaviour is exactly preserved) ---
# Turn 2 (more unfiltered bot data) plateaued because AWBC imitates whatever the recording did,
# and the bot's recordings are mostly its own flaws. These three knobs bias the ACTOR loss only
# (the Q/V critics still see everything, so the reward signal is untouched):
#   --mask-hits S   : zero the actor weight on the S seconds BEFORE each mined hit (those actions
#                     caused the hit — stop cloning them).
#   --human-weight W: multiply the actor weight by W for the human demos (hf2/hf3/hf4), so the
#                     near-optimal human prior is not drowned by bot self-play.
#   --min-quality   : drop bot runs with >1 pit fall from the corpus entirely (human runs kept).
MASK_HITS = float(_farg("--mask-hits", 0.0))
HUMAN_WEIGHT = float(_farg("--human-weight", 1.0))
MIN_QUALITY = "--min-quality" in sys.argv[1:]
_HUMAN = ("hf2", "hf3", "hf4")
# --- M3 value-side knobs (both default to a NO-OP so iql3 behaviour is bit-identical) ---
#   --nstep N     : n-step Q-target. N=1 (default) keeps the exact 1-step target
#                   q_tgt = r_i + GAMMA*V(s_{i+1}); N>1 bootstraps m = min(N, steps-to-run-end)
#                   ahead so it never crosses a run boundary (precomputed below, guarded).
#   --cql-alpha A : CQL conservative penalty A*(logsumexp_a Q(s,a) - Q(s,a_data)) on BOTH critics.
#                   A=0.0 (default) => the term is skipped => Q-loss is bit-identical.
NSTEP = max(int(_farg("--nstep", 1)), 1)
CQL_ALPHA = float(_farg("--cql-alpha", 0.0))
# --- memory budget (2026-07-17) ---------------------------------------------------------------
#   --max-frames N : cap the corpus to ~N frames. The frame bank is n*H*W uint8, and at the
#                    deployed K=10/96x224 geometry that is ~21.5KB PER TRANSITION — so the full
#                    1.43M-transition corpus is a ~30.6GB bank: bigger than this box's 16GB VRAM
#                    AND its 31GB RAM (iql5b thrashed the pagefile, then OOM'd the GPU load and
#                    died without saving). The cap keeps ALL human demos (the valuable prior)
#                    plus the FRESHEST bot runs, and LOGS every run it drops — a silent cap would
#                    read as "trained on everything". 0 (default) = no cap (original behaviour).
MAX_FRAMES = int(_farg("--max-frames", 0))
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

if MAX_FRAMES > 0:
    def _nframes(d):
        try:
            return len(json.load(open(os.path.join(d, "frames.json")))["frames"])
        except Exception:
            return 0
    _humans = [d for d in run_dirs if os.path.basename(d) in _HUMAN]
    _bots = [d for d in run_dirs if os.path.basename(d) not in _HUMAN]
    _kept, _total = list(_humans), sum(_nframes(d) for d in _humans)
    _dropped = []
    for _d in sorted(_bots, key=os.path.getmtime, reverse=True):   # freshest bot runs first
        _c = _nframes(_d)
        if _total + _c <= MAX_FRAMES:
            _kept.append(_d); _total += _c
        else:
            _dropped.append(_d)
    if _dropped:
        print(f"[budget] --max-frames {MAX_FRAMES}: keeping {len(_kept)} runs "
              f"({_total} frames, ~{_total * H * W / 1e9:.1f}GB bank); DROPPED "
              f"{len(_dropped)} oldest bot run(s): "
              f"{', '.join(os.path.basename(d) for d in _dropped)}", flush=True)
    run_dirs = sorted(_kept)
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
actor_w_all = []                            # M2: per-transition ACTOR-loss weight (1.0 = default)
offset = 0
for rdir in run_dirs:
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    if not recording_is_complete(fm):
        print(f"  {os.path.basename(rdir)}: incomplete recording, skipped", flush=True)
        continue
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    ts = np.array([f["t"] for f in frames])
    n = len(frames)
    # model-res gray frames (reuse the SSL cache when present)
    _ctag = "-".join(f"{v:g}" for v in CROP)
    cache = os.path.join(rdir, f"cache_ssl_{H}x{W}_{_ctag}.npy")
    if os.path.exists(cache) and len(np.load(cache, mmap_mode="r")) == n:
        # READ-ONLY MMAP (2026-07-17): loading every run's cache fully into RAM made the later
        # np.concatenate need BOTH the per-run copies and the destination (~2x the bank = ~61GB
        # on the 1.43M corpus) — it thrashed the pagefile on this 31GB box. The concatenate can
        # copy straight out of the mmap, so peak RAM becomes just the destination bank.
        imgs = np.load(cache, mmap_mode="r")
    else:
        imgs = np.zeros((n, H, W), np.uint8)     # cache miss: must be writable to build it
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
    is_human = os.path.basename(rdir) in _HUMAN
    if MIN_QUALITY and not is_human and len(pidx) > 1:
        print(f"  (min-quality: SKIP {os.path.basename(rdir)} — {len(pidx)} falls)", flush=True)
        continue
    # penalize the frames LEADING INTO the fall (the prompt shows ~0.5-1s after the miss;
    # the mistimed/missing jump happened ~0.5s earlier) so credit lands on the decision
    for pi in pidx:
        lo = np.searchsorted(ts, ts[pi] - PIT_SPREAD)
        rew[lo:pi + 1] += PIT_R / max(pi + 1 - lo, 1)
        pit_win_all.append(np.arange(lo, pi + 1, dtype=np.int64) + offset)
    rew[-1] += DEATH_R                     # run end = death (or the human stopped: close enough)
    # M2: ACTOR-loss weight (1.0 = default). mask-hits zeros the pre-hit window; human-weight
    # scales the human demos. Applied ONLY to the policy loss, so the critics are unaffected.
    aw = np.ones(n, np.float32)
    if MASK_HITS > 0:
        for hi in hidx:
            lo = np.searchsorted(ts, ts[hi] - MASK_HITS)
            aw[lo:hi + 1] = 0.0
    if is_human and HUMAN_WEIGHT != 1.0:
        aw *= HUMAN_WEIGHT
    print(f"  {os.path.basename(rdir)}: {n} frames | {len(hidx)} hits | {len(pidx)} PIT FALLS | "
          f"actions {dict(zip(CLASSES, np.bincount(act, minlength=3).tolist()))}", flush=True)
    imgs_all.append(imgs); act_all.append(act); rew_all.append(rew); actor_w_all.append(aw)
    run_start.extend([offset] * n)
    offset += n

if not imgs_all:
    raise SystemExit("no complete recordings to train")
imgs = np.concatenate(imgs_all)
act = np.concatenate(act_all)
rew = np.concatenate(rew_all)
actor_w = np.concatenate(actor_w_all)       # M2 per-transition policy-loss weight (default all 1)
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
# BANK MEMORY (OOM fix 2026-07-17): the frame bank is n*H*W uint8 — once the corpus grew past
# ~1.4M transitions that is tens of GB, and pushing it all to VRAM silently OOM-killed the run
# right after the corpus-load print (iql5b died here; iql5c's --min-quality corpus still fit).
# Auto-pick: keep the bank in VRAM while it comfortably fits (fast path, unchanged), else keep
# it in CPU RAM and move only the gathered per-batch stacks (B*K*H*W = a few MB) to the GPU.
_bank_bytes = int(imgs.nbytes)
try:
    _free_vram = int(torch.cuda.mem_get_info()[0])
except Exception:                                   # probe unavailable -> assume it won't fit
    _free_vram = 0                                  # (CPU streaming is the safe fallback)
BANK_ON_GPU = _bank_bytes < 0.5 * _free_vram        # headroom for nets, activations, batches
bank = torch.from_numpy(imgs)
if BANK_ON_GPU:
    bank = bank.to(dev)
else:
    print(f"[mem] frame bank {_bank_bytes / 1e9:.1f}GB vs {_free_vram / 1e9:.1f}GB free VRAM "
          f"-> streaming batches from CPU RAM", flush=True)
del imgs
ks = torch.arange(K - 1, -1, -1, dtype=torch.long, device=dev)
idx_all = torch.arange(n, dtype=torch.long, device=dev)
start_g = torch.from_numpy(run_start).to(dev)
act_g = torch.from_numpy(act).to(dev)
rew_g = torch.from_numpy(rew).to(dev)
term_g = torch.from_numpy(is_term).to(dev)
actor_w_g = torch.from_numpy(actor_w).to(dev)   # M2 policy-loss weight (all 1.0 by default)


def stacks(i):
    """(B,K,H,W) float stacks for frame indices i, clamped at run starts."""
    idx = torch.maximum(i[:, None] - ks[None, :], start_g[i][:, None])
    if BANK_ON_GPU:
        return bank[idx].float().div_(255.0)
    # bank lives in CPU RAM (corpus too big for VRAM): gather there, then move just this batch
    return bank[idx.cpu()].to(dev, non_blocking=True).float().div_(255.0)


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
      f"mask_hits={MASK_HITS} human_weight={HUMAN_WEIGHT} min_quality={MIN_QUALITY} "
      f"nstep={NSTEP} cql_alpha={CQL_ALPHA} "
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
if NSTEP != 1:
    # ---- M3 n-step returns (precomputed ONCE, numpy -> GPU) --------------------------------
    # q_tgt[i] = Σ_{k=0}^{m-1} GAMMA^k r_{i+k} + GAMMA^m V(s_{i+m}),  m = min(NSTEP, run_end[i]-i)
    # so the bootstrap frame i+m NEVER crosses a run boundary. The bootstrap is off the LAST
    # in-run frame (mirrors the existing 1-step target, which likewise bootstraps V at the —
    # possibly terminal — successor); the run-final DEATH_R reward is therefore captured through
    # V(s_{i+m}), never summed directly, keeping the run-boundary semantics identical to 1-step.
    # (Guarded by NSTEP != 1: with N=1 the epoch loop takes the original 1-step branch verbatim.)
    _arange = np.arange(n, dtype=np.int64)
    run_end = np.empty(n, np.int64)               # nearest terminal frame index at/after each frame
    _nxt = n - 1
    for _j in range(n - 1, -1, -1):
        if is_term[_j]:
            _nxt = _j
        run_end[_j] = _nxt
    _m = np.minimum(NSTEP, run_end - _arange).astype(np.int64)   # >=1 on non-terminals, 0 on terminals
    nstep_boot = _arange + _m                                    # i+m, guaranteed <= run_end[i]
    nstep_gpow = (GAMMA ** _m).astype(np.float32)                # GAMMA^m
    _disc = np.zeros(n, np.float32)
    for _k in range(NSTEP):
        _sel = np.where(_k < _m)[0]                             # transitions whose window still reaches step k
        _disc[_sel] += np.float32((GAMMA ** _k) * rew[_sel + _k])   # accumulate GAMMA^k * r_{i+k}
    nstep_boot_g = torch.from_numpy(nstep_boot).to(dev)
    nstep_disc_rew_g = torch.from_numpy(_disc).to(dev)
    nstep_gpow_g = torch.from_numpy(nstep_gpow).to(dev)
    print(f"n-step={NSTEP}: mean m={float(_m[~is_term].mean()):.2f} over "
          f"{int((~is_term).sum())} non-terminal transitions", flush=True)
for ep in range(EPOCHS):
    perm = valid[torch.randperm(len(valid), device=dev)]
    tot_q = tot_v = tot_p = 0.0
    nb = 0
    for b in range(0, len(perm), BATCH):
        i = perm[b:b + BATCH]
        s = stacks(i)
        a = act_g[i]
        r = rew_g[i]
        with torch.no_grad():
            if NSTEP != 1:
                boot = nstep_boot_g[i]; base_r = nstep_disc_rew_g[i]; gpow = nstep_gpow_g[i]
            else:
                boot = i + 1; base_r = r; gpow = GAMMA
            vnext = vf(stacks(boot)).squeeze(1)
            # DONE-MASK FIX (bug hunt 2026-07-14): when the bootstrap frame is a run-terminal,
            # V(terminal) is an UNTRAINED output (terminals are excluded from `valid`, so V never
            # learns them) and the terminal reward — which carries DEATH_R=-5 — was never used in
            # any target. Bootstrap the terminal's OWN reward instead of the garbage V, so the
            # death penalty finally trains. Non-terminal successors keep the normal V bootstrap.
            bootstrap = torch.where(term_g[boot], rew_g[boot], vnext)
            q_tgt = base_r + gpow * bootstrap
            qmin = torch.minimum(q1t(s).gather(1, a[:, None]).squeeze(1),
                                 q2t(s).gather(1, a[:, None]).squeeze(1))
        # V: expectile regression toward min target-Q
        u = qmin - vf(s).squeeze(1)
        lv = (torch.abs(TAU - (u < 0).float()) * u.pow(2)).mean()
        opt_v.zero_grad(); lv.backward(); opt_v.step()
        # Q: TD toward r + gamma V(s')
        q1a, q2a = q1(s), q2(s)                     # full Q(s,.) logits, reused by the CQL term
        lq = F.mse_loss(q1a.gather(1, a[:, None]).squeeze(1), q_tgt) + \
             F.mse_loss(q2a.gather(1, a[:, None]).squeeze(1), q_tgt)
        if CQL_ALPHA != 0.0:
            # CQL conservatism: push Q down on OOD actions via A*(logsumexp_a Q - Q(a_data)),
            # added to BOTH critics (no extra forward — reuses q1a/q2a). A=0.0 => skipped.
            cql = ((torch.logsumexp(q1a, dim=1) - q1a.gather(1, a[:, None]).squeeze(1)).mean()
                 + (torch.logsumexp(q2a, dim=1) - q2a.gather(1, a[:, None]).squeeze(1)).mean())
            lq = lq + CQL_ALPHA * cql
        opt_q.zero_grad(); lq.backward(); opt_q.step()
        # policy: advantage-weighted BC
        with torch.no_grad():
            adv = qmin - vf(s).squeeze(1)
            wgt = torch.exp(BETA * adv).clamp(max=100.0)
        logp = F.log_softmax(pi(s), dim=1).gather(1, a[:, None]).squeeze(1)
        lp = -(wgt * actor_w_g[i] * logp).mean()   # M2: actor_w defaults to 1.0 (no-op)
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
