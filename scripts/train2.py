import os, json, sys, time, glob
from _runtime import DATA, ROOT, recording_is_complete
import numpy as np, cv2
import torch, torch.nn as nn
from cookierun_bot.policies.learned import build_net_from_meta
from cookierun_bot.policies import condition

BASE = str(DATA)
OUT = os.path.join(BASE, "demo")               # model.pt destination (LearnedAgent path)
HITS = os.path.join(BASE, "ai_hits")
CLASSES = ["none", "jump", "slide"]
CHECK_CORR = "--check-corr" in sys.argv[1:]
# EPOCHS = a LEADING positional only (e.g. `train2.py 30 --arch ...`). Must NOT scan all argv for
# any digit or it swallows flag values like `--k 6` -> trains 6 epochs instead of 30 (silent under-train).
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
_ARCH_ALIASES = {
    "mobile": "mobilenet_v3_large",
    "mobilenet": "mobilenet_v3_large",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "b5": "efficientnet_b5",
    "efficientnet": "efficientnet_b5",
    "efficientnet_b5": "efficientnet_b5",
    "small": "small_cnn",
    "small_cnn": "small_cnn",
    "film": "small_cnn_film",
    "small_cnn_film": "small_cnn_film",
}
ARCH = "small_cnn"
ARCH_SET = False             # whether --arch was passed explicitly (else --meta-from inherits it)
OUT_PREFIX = "model"
RUN_NAMES = None
USE_WANDB = "--wandb" in sys.argv[1:]
WANDB_PROJECT = "cookierun-bot"
WANDB_ENTITY = None
WANDB_NAME = None
WANDB_MODE = None
K_OVERRIDE = None            # --k N overrides the temporal frame-stack depth (default 4)
CROP_OVERRIDE = None         # --crop x0,y0,x1,y1 overrides the input crop (for crop A/B tuning)
META_FROM = None             # --meta-from PATH: inherit K/H/W/crop/fps/win_pre/conv/fc from a deployed
                             # model_meta.json so a self-farm retrain keeps the EXACT arch it replaces
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--arch" and i + 2 < len(sys.argv):
        ARCH = _ARCH_ALIASES.get(sys.argv[i + 2], sys.argv[i + 2])
        ARCH_SET = True
    elif arg == "--k" and i + 2 < len(sys.argv):
        K_OVERRIDE = int(sys.argv[i + 2])
    elif arg.startswith("--k="):
        K_OVERRIDE = int(arg.split("=", 1)[1])
    elif arg == "--crop" and i + 2 < len(sys.argv):
        CROP_OVERRIDE = [float(x) for x in sys.argv[i + 2].split(",")]
    elif arg.startswith("--crop="):
        CROP_OVERRIDE = [float(x) for x in arg.split("=", 1)[1].split(",")]
    elif arg == "--meta-from" and i + 2 < len(sys.argv):
        META_FROM = sys.argv[i + 2]
    elif arg.startswith("--meta-from="):
        META_FROM = arg.split("=", 1)[1]
    elif arg == "--out-prefix" and i + 2 < len(sys.argv):
        OUT_PREFIX = sys.argv[i + 2]
    elif arg == "--runs" and i + 2 < len(sys.argv):
        RUN_NAMES = [r.strip() for r in sys.argv[i + 2].split(",") if r.strip()]
    elif arg == "--wandb-project" and i + 2 < len(sys.argv):
        WANDB_PROJECT = sys.argv[i + 2]
    elif arg == "--wandb-entity" and i + 2 < len(sys.argv):
        WANDB_ENTITY = sys.argv[i + 2]
    elif arg == "--wandb-name" and i + 2 < len(sys.argv):
        WANDB_NAME = sys.argv[i + 2]
    elif arg == "--wandb-mode" and i + 2 < len(sys.argv):
        WANDB_MODE = sys.argv[i + 2]
    elif arg.startswith("--arch="):
        ARCH = _ARCH_ALIASES.get(arg.split("=", 1)[1], arg.split("=", 1)[1])
        ARCH_SET = True
    elif arg.startswith("--out-prefix="):
        OUT_PREFIX = arg.split("=", 1)[1]
    elif arg.startswith("--runs="):
        RUN_NAMES = [r.strip() for r in arg.split("=", 1)[1].split(",") if r.strip()]
    elif arg.startswith("--wandb-project="):
        WANDB_PROJECT = arg.split("=", 1)[1]
    elif arg.startswith("--wandb-entity="):
        WANDB_ENTITY = arg.split("=", 1)[1]
    elif arg.startswith("--wandb-name="):
        WANDB_NAME = arg.split("=", 1)[1]
    elif arg.startswith("--wandb-mode="):
        WANDB_MODE = arg.split("=", 1)[1]
    elif arg in _ARCH_ALIASES:
        ARCH = _ARCH_ALIASES[arg]
        ARCH_SET = True
torch.manual_seed(0)

META = {
    "classes": CLASSES,
    "arch": ARCH,
    "K": 4, "H": 96, "W": 224,
    "crop": [0.10, 0.20, 1.00, 0.90],
    "fps": 35.0,
    "conv": [[24, 5, 2], [48, 3, 2], [64, 3, 2], [64, 3, 2]],
    "fc": 256,
    "win_pre": 0.25, "win_post": 0.03,
    "notyet_lo": 0.25, "notyet_hi": 0.70, "notyet_w": 4.0,
    "corr_w": 5.0,
}
# --meta-from: inherit the deployed model's EXACT architecture/labeling so a self-farm retrain
# reproduces it (e.g. the 60fps K10 win_pre-0.2 hf2 model) instead of resetting to the 35fps defaults.
if META_FROM and os.path.exists(META_FROM):
    _base = json.load(open(META_FROM))
    for _k in ("K", "H", "W", "crop", "fps", "conv", "fc",
               "win_pre", "win_post", "notyet_lo", "notyet_hi", "notyet_w", "cond",
               "label_shift_ms"):
        if _k in _base:
            META[_k] = _base[_k]
    if not ARCH_SET and "arch" in _base:      # a retrain of a film model must stay film
        ARCH = _ARCH_ALIASES.get(_base["arch"], _base["arch"])
        META["arch"] = ARCH
    print(f"meta-from {META_FROM}: K{META['K']} fps{META['fps']} win_pre{META['win_pre']} "
          f"arch {ARCH}", flush=True)
if K_OVERRIDE:
    META["K"] = K_OVERRIDE
if CROP_OVERRIDE:
    META["crop"] = CROP_OVERRIDE

# FiLM conditioning (arch small_cnn_film): the model additionally takes [t, speed, bonus]
# per frame (see policies/condition.py). speed_norm is calibrated below (encoder meta or
# corpus p90) and written into meta so LearnedAgent normalises live exactly like training.
FILM = ARCH == "small_cnn_film"
if FILM and "cond" not in META:
    META["cond"] = {"dims": list(condition.COND_DIMS), "t_norm_s": condition.T_NORM_S,
                    "speed_norm": None, "bonus_latch_s": condition.BONUS_LATCH_S}
if FILM:
    # cond is rebuilt from scratch every training, so a new checkpoint is always trained
    # with the CURRENT estimator. A meta-from-inherited speed_norm from a different
    # scroll_v is on the wrong scale — drop it so calibration reruns.
    if META["cond"].get("scroll_v") != condition.SCROLL_V:
        META["cond"]["speed_norm"] = None
    META["cond"]["scroll_v"] = condition.SCROLL_V
if not FILM:
    META.pop("cond", None)                    # a non-film arch must not carry cond meta
BT_TPL = condition.load_bonus_template(str(ROOT / "templates")) if FILM else None
if FILM:
    # recorded into meta so LIVE stays consistent: a model trained with bonus all-0 must
    # also get bonus=0 live (LearnedAgent honors this), even on a machine with the template
    META["cond"]["bonus_trained"] = BT_TPL is not None
if FILM and BT_TPL is None:
    print("WARNING: templates/bonustime_norm.png missing — bonus cond dim will be all-0",
          flush=True)

# --- no-labeling training improvements (all use signals we already have) ---
# (death-discount was tried + dropped: the per-run 85/15 split already holds the fatal last 15%
#  out of training, so the death is never cloned — an explicit pre-death down-weight is a no-op.)
AUG = "--no-aug" not in sys.argv        # train-time augmentation (brightness/contrast/shift/cutout)


def _farg(flag, default):               # value of `--flag V` or `--flag=V` (first occurrence)
    a = sys.argv[1:]
    for i, tok in enumerate(a):
        if tok == flag and i + 1 < len(a):
            return a[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


# AWR: weight each run's frames by its survival return, ramped REWARD_LO..REWARD_HI. Raising
# --reward-lo toward 1.0 lets long self-farm runs count as much as the human anchors (the one
# lever that can push past the base model — but risky, validate offline before deploying).
REWARD_LO = float(_farg("--reward-lo", 0.6))
REWARD_HI = float(_farg("--reward-hi", 1.4))
# NEGATIVE ("hit cam") signal: bot self-runs mined by scripts/mine_negatives.py into an .npz of
# pre-hit/pit K-stacks where the bot was passive ("none") and got hit/fell. An UNLIKELIHOOD loss
# pushes p(none) DOWN on those frames — the model learns to be less passive on obstacle patterns
# it currently dies to. Human demos remain the POSITIVE anchor. Opt-in via --neg-npz; off by default.
NEG_NPZ = _farg("--neg-npz", None)
NEG_LAMBDA = float(_farg("--neg-lambda", 0.5))
# --neg-mode: how the bot-failure frames are used.
#   "unlikelihood" (default): push p(none) DOWN (redistributes to jump+slide; can over-bias slide).
#   "jump": label them JUMP (positive CE) — targets pits + counters the slide-bias directly.
NEG_MODE = _farg("--neg-mode", "unlikelihood")
NEG_JUMP_W = float(_farg("--neg-jump-w", 0.3))    # weight of the jump-correction loss in "jump" mode
# SSL encoder transfer (scripts/pretrain_encoder.py): load pretrained conv weights, and
# optionally freeze them ("--freeze-enc all" or first-N conv layers) so the thin demos
# only train the head. --save-best keeps the best eval-epoch weights instead of the last
# epoch (the sweep proved last-epoch saves overfit).
ENCODER_INIT = _farg("--encoder-init", None)
FREEZE_ENC = _farg("--freeze-enc", None)
SAVE_BEST = "--save-best" in sys.argv[1:]
_ENC = None
if ENCODER_INIT:
    if not os.path.exists(ENCODER_INIT):
        raise SystemExit(f"--encoder-init {ENCODER_INIT} not found")
    _ENC = torch.load(ENCODER_INIT, map_location="cpu")
    _em = _ENC["meta"]
    for _k in ("K", "H", "W", "conv", "crop"):
        if _em.get(_k) != META.get(_k):
            raise SystemExit(f"encoder/model geometry mismatch on '{_k}': "
                             f"{_em.get(_k)} vs {META.get(_k)}")
if FILM and NEG_NPZ:
    raise SystemExit("--neg-npz is not supported with the film arch")
# --slide-span-cap: cap how much of a slide HOLD is labeled "slide". The full hold (default 3.0)
# labels every crouched frame as slide -> the model learns "crouched cookie -> slide" -> a live
# SLIDE-LOCK (once it slides it sees itself crouched and keeps sliding, never jumps pits). A short
# cap labels only the slide ONSET so the model reacts to OBSTACLES, not its own crouch.
SLIDE_SPAN_CAP = float(_farg("--slide-span-cap", 3.0))
# --label-shift-ms: shift key-event timestamps EARLIER by this many ms at load time, BEFORE any
# labeling windows are computed (live analysis: ~89% of hits are "fired-but-hit", the action lands
# ~50-100ms late — earlier onsets teach earlier fires). This is a window OFFSET, deliberately NOT
# a win_pre/notyet retune (that experiment is a known failure — do not touch those). Press and the
# implied release shift together (`dur` is relative), so hold durations are preserved. Positive =
# earlier; default 0.0 = today's exact behavior (timestamps never touched). Inherited via
# --meta-from like the other labeling knobs; an explicit flag overrides.
LABEL_SHIFT_MS = float(_farg("--label-shift-ms", META.get("label_shift_ms", 0.0)))
META["label_shift_ms"] = LABEL_SHIFT_MS        # provenance in the saved *_meta.json
if LABEL_SHIFT_MS:
    print(f"label-shift: key events shifted {LABEL_SHIFT_MS:.0f} ms EARLIER", flush=True)
# per-run reward multiplier, e.g. `--run-weight demo4=0.5` to down-weight the dominant anchor.
# Repeatable; applied AFTER the human floor so a factor <1.0 actually bites. Names may omit the
# `demo` prefix (`--run-weight 4=0.5`).
RUN_WEIGHTS = {}
_av = sys.argv[1:]
for _i, _a in enumerate(_av):
    _spec = _av[_i + 1] if (_a == "--run-weight" and _i + 1 < len(_av)) else (
        _a.split("=", 1)[1] if _a.startswith("--run-weight=") else None)
    if _spec and "=" in _spec:
        _nm, _mult = _spec.split("=", 1)
        RUN_WEIGHTS[_nm] = float(_mult)                 # exact run-dir name (e.g. hf2, hf3)
        if not _nm.startswith("demo"):                  # convenience: bare "4" also matches demo4
            RUN_WEIGHTS[f"demo{_nm}"] = float(_mult)


def load_corrections(meta) -> "tuple[np.ndarray, np.ndarray] | tuple[None, None]":
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
            img_rel = r.get("img")
            base_img = band(os.path.join(HITS, img_rel)) if img_rel else None
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

if RUN_NAMES is not None:
    # explicit --runs: take EXACTLY these dirs by name, ANY namespace (demo*/hf*/demo_self_*), so a
    # 60fps hf2 anchor + self-runs can be trained together. A bare "2" still maps to "demo2".
    wanted = [r if os.path.isdir(os.path.join(BASE, r)) else f"demo{r}" for r in RUN_NAMES]
    runs, missing = [], []
    for w in wanted:
        d = os.path.join(BASE, w)
        (runs if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json")) else missing).append(w)
    if missing:
        raise SystemExit(f"missing requested trainable demo(s): {', '.join(sorted(missing))}")
    runs = sorted(os.path.join(BASE, w) for w in runs)
else:
    runs = sorted(d for d in glob.glob(os.path.join(BASE, "demo*"))     # default: the 35fps demo* set
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json"))
                  and "test" not in os.path.basename(d))
complete_runs = []
for rdir in runs:
    metadata = json.load(open(os.path.join(rdir, "frames.json")))
    if recording_is_complete(metadata):
        complete_runs.append(rdir)
    else:
        print(f"  {os.path.basename(rdir)}: incomplete recording, skipped", flush=True)
runs = complete_runs
if not runs:
    raise SystemExit("no complete recordings to train")
print("runs:", [os.path.basename(r) for r in runs], flush=True)

wandb = None
wandb_run = None
if USE_WANDB:
    try:
        import wandb as _wandb
    except ModuleNotFoundError as exc:
        raise SystemExit("W&B logging requested but wandb is not installed. Run `python -m pip install wandb`.") from exc
    wandb = _wandb
    init_kwargs = {
        "project": WANDB_PROJECT,
        "name": WANDB_NAME or f"{OUT_PREFIX}-{time.strftime('%Y%m%d-%H%M%S')}",
        "config": {
            "arch": ARCH,
            "epochs": EPOCHS,
            "out_prefix": OUT_PREFIX,
            "runs": [os.path.basename(r) for r in runs],
            "run_filter": RUN_NAMES,
            "meta": META,
            "cuda_required": True,
        },
    }
    if WANDB_ENTITY:
        init_kwargs["entity"] = WANDB_ENTITY
    if WANDB_MODE:
        init_kwargs["mode"] = WANDB_MODE
    wandb_run = wandb.init(**init_kwargs)
    print(f"wandb: {wandb_run.url}", flush=True)

x0f, y0f, x1f, y1f = META["crop"]
H, W, K = META["H"], META["W"], META["K"]
imgs_all, y_all, notyet_all, run_id, run_start = [], [], [], [], []
tr_ids, va_ids, frame_dts = [], [], []
run_stats = []
run_returns, run_is_self = [], []        # AWR: survival-weight each run (better runs teach more)
cond_parts = []                          # film: per-run (ts, raw speeds, latched bonus)
offset = 0
for ri, rdir in enumerate(runs):
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    if LABEL_SHIFT_MS:
        # one shift at the single load point so EVERY consumer below (positive labels,
        # notyet windows, slide-span ends) sees the same earlier timeline; dur untouched.
        for k in keys:
            k["t"] -= LABEL_SHIFT_MS / 1000.0
    ts = np.array([f["t"] for f in frames])
    if len(ts) > 1:
        frame_dts.append(np.diff(ts))
    y = np.zeros(len(frames), np.int64)
    notyet = np.zeros(len(frames), bool)
    span_labeled = 0
    for k in keys:
        cls = CLASSES.index(k["action"])
        # SPAN labeling for slide: if the recorder captured how long S was held (`dur`),
        # label the whole hold [t .. t+dur] as slide so the model learns to SUSTAIN the
        # slide for the obstacle's length -> at inference the slide extends while the model
        # keeps deciding slide = obstacle-adaptive duration. Jump + legacy (no `dur`) demos
        # keep the point +/- window. Capped at 3s so a stuck key can't poison a huge span.
        dur = float(k.get("dur", 0.0) or 0.0)
        span = min(dur, SLIDE_SPAN_CAP) if (dur > 0.0 and k["action"] == "slide") else 0.0
        if span > 0.0:
            span_labeled += 1
        lo = np.searchsorted(ts, k["t"] - META["win_pre"])
        hi = np.searchsorted(ts, k["t"] + span + META["win_post"])
        y[lo:hi] = cls
        nlo = np.searchsorted(ts, k["t"] - META["notyet_hi"])
        nhi = np.searchsorted(ts, k["t"] - META["notyet_lo"])
        notyet[nlo:nhi] = True
    notyet &= (y == 0)                        # only none-frames get the boost
    label_counts = dict(zip(CLASSES, np.bincount(y, minlength=3).tolist()))
    print(f"  {os.path.basename(rdir)}: {len(frames)} frames, {len(keys)} keys, "
          f"labels {label_counts}, "
          f"not-yet {int(notyet.sum())}, span-slides {span_labeled}", flush=True)
    t0 = time.time()
    imgs = np.zeros((len(frames), H, W), np.uint8)
    bt_raw = np.zeros(len(frames), bool)
    fdir = os.path.join(rdir, "frames")
    for i, fr in enumerate(frames):
        im = cv2.imread(os.path.join(fdir, f"{fr['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        if FILM:                               # banner lives ABOVE the model crop band
            bt_raw[i] = condition.bonustime_gray(im, BT_TPL)
        h, w = im.shape
        band = im[int(h*y0f):int(h*y1f), int(w*x0f):int(w*x1f)]
        imgs[i] = cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)
    print(f"    loaded in {time.time()-t0:.0f}s", flush=True)
    if FILM:
        _speeds = condition.run_speeds(ts, imgs, scroll_v=META["cond"]["scroll_v"])
        _blatch = condition.latch_bonus(ts, bt_raw, META["cond"]["bonus_latch_s"])
        cond_parts.append((ts, _speeds, _blatch))
        print(f"    cond: speed med {np.median(_speeds[_speeds > 0]) if (_speeds > 0).any() else 0:.0f} px/s"
              f" | bonus frames {int(_blatch.sum())}", flush=True)
    imgs_all.append(imgs); y_all.append(y); notyet_all.append(notyet)
    _dur = float(fm.get("duration_s") or (ts[-1] - ts[0] if len(ts) > 1 else 1.0))
    run_returns.append(max(_dur, 1.0))
    run_is_self.append(os.path.basename(rdir).startswith("demo_self"))
    run_id.extend([ri] * len(frames)); run_start.extend([offset] * len(frames))
    cut = offset + int(len(frames) * 0.85)
    tr_ids.extend(range(offset, cut)); va_ids.extend(range(cut, offset + len(frames)))
    run_stats.append({
        "run": os.path.basename(rdir),
        "frames": len(frames),
        "keys": len(keys),
        "train_frames": cut - offset,
        "val_frames": offset + len(frames) - cut,
        "notyet": int(notyet.sum()),
        "span_slides": span_labeled,
        **{f"labels_{k}": v for k, v in label_counts.items()},
    })
    offset += len(frames)

imgs = np.concatenate(imgs_all); y = np.concatenate(y_all)
notyet = np.concatenate(notyet_all)
run_start = np.array(run_start); n = len(y)
print(f"total {n} frames | train {len(tr_ids)} | val {len(va_ids)}", flush=True)
# The K-stack gathers stride CONSECUTIVE recorded frames, so training spacing IS the recorder
# cadence. Measure it from the demos' real timestamps and write it into meta so LearnedAgent
# gates its live K-stack at the SAME span (fixes the 60fps-record vs 35fps-assumed drift).
if frame_dts:
    META["fps"] = round(1.0 / float(np.median(np.concatenate(frame_dts))), 1)
    print(f"measured recording fps: {META['fps']}", flush=True)

cond = None
if FILM:
    # speed_norm precedence: (1) a meta-from-inherited value wins so every retrain in a
    # lineage keeps ONE scale (comparable cond across hot-swaps), (2) else the pretrained
    # encoder's corpus-wide p90, (3) else this demo set's p90.
    spn = META["cond"].get("speed_norm") or (_ENC and _ENC["meta"].get("speed_p90"))
    if not spn:
        _pos = np.concatenate([s[s > 0] for _, s, _ in cond_parts]) if cond_parts else np.array([1.0])
        spn = float(np.percentile(_pos, 90)) if len(_pos) else 1.0
    META["cond"]["speed_norm"] = round(float(spn), 1)
    cond = np.concatenate([
        condition.build_run_cond(t_, s_, b_, META["cond"]["t_norm_s"], META["cond"]["speed_norm"])
        for t_, s_, b_ in cond_parts])
    print(f"cond: speed_norm {META['cond']['speed_norm']} px/s | "
          f"bonus share {cond[:, 2].mean():.3f}", flush=True)

if wandb_run:
    dataset_metrics = {
        "dataset/frames": n,
        "dataset/train_frames": len(tr_ids),
        "dataset/val_frames": len(va_ids),
        "dataset/runs": len(runs),
    }
    wandb_run.config.update({"dataset": dataset_metrics, "run_stats": run_stats}, allow_val_change=True)
    table = wandb.Table(columns=list(run_stats[0].keys()) if run_stats else ["run"], data=[
        [row[k] for k in run_stats[0].keys()] for row in run_stats
    ] if run_stats else [])
    wandb.log({**dataset_metrics, "dataset/run_stats": table}, step=0)

# DAgger corrections (scripts/correct.py) — train-only extra samples, never in val
corr_x, corr_y = load_corrections(META)
n_corr = 0 if corr_y is None else len(corr_y)
if n_corr:
    print(f"corrections: {n_corr} mixed in at weight x{META['corr_w']} | per class "
          f"{dict(zip(CLASSES, np.bincount(corr_y, minlength=3).tolist()))}", flush=True)

counts = np.bincount(y[tr_ids], minlength=3)
print("train class counts:", dict(zip(CLASSES, counts.tolist())), flush=True)
w_cls = 1.0 / np.sqrt(np.maximum(counts, 1))
w = w_cls[y[tr_ids]].astype(np.float64)
w[notyet[tr_ids]] *= META["notyet_w"]          # "obstacle coming but do NOT act yet"
# AWR survival-weighting: better-surviving runs teach more. Human demos (non-self) are floored
# at baseline so a tanking self-run can never outweigh the expert anchors.
_ret = np.array(run_returns, np.float64)
_rn = ((_ret - _ret.min()) / (_ret.max() - _ret.min())) if _ret.max() > _ret.min() else np.full(len(_ret), 0.5)
run_reward_w = REWARD_LO + (REWARD_HI - REWARD_LO) * _rn
run_reward_w = np.where(np.array(run_is_self), run_reward_w, np.maximum(run_reward_w, 1.0))
if RUN_WEIGHTS:                                # per-run down/up-weight, AFTER the floor so <1.0 bites
    for _j, _rdir in enumerate(runs):
        _nm = os.path.basename(_rdir)
        if _nm in RUN_WEIGHTS:
            run_reward_w[_j] *= RUN_WEIGHTS[_nm]
reward_w_frame = run_reward_w[np.array(run_id)]
w *= reward_w_frame[tr_ids]
print(f"[reward] lo={REWARD_LO} hi={REWARD_HI}"
      + (f" run-weights={RUN_WEIGHTS}" if RUN_WEIGHTS else "")
      + " | survival-weight/run: "
      + ", ".join(f"{os.path.basename(r)}={rw:.2f}" for r, rw in zip(runs, run_reward_w)), flush=True)
if n_corr:                                     # corrections: class weight x corr boost
    w = np.concatenate([w, w_cls[corr_y] * META["corr_w"]])

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this trainer; refusing to train on CPU.")
dev = torch.device("cuda")
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
print(f"arch: {ARCH} | epochs: {EPOCHS} | out-prefix: {OUT_PREFIX}", flush=True)
net = build_net_from_meta(torch, META).to(dev)
if _ENC is not None:
    _sd = _ENC["convs"]
    if FILM:
        net.convs.load_state_dict(_sd)
    else:                                # small_cnn: conv prefix shares Sequential indices
        _miss = net.load_state_dict(_sd, strict=False)
        assert not _miss.unexpected_keys, f"encoder keys not in net: {_miss.unexpected_keys}"
    print(f"encoder-init: {len(_sd)} conv tensors from {ENCODER_INIT} "
          f"(pretrain val L1 {_ENC['meta'].get('best_val_l1')})", flush=True)
if FREEZE_ENC:
    if ARCH not in ("small_cnn", "small_cnn_film"):
        raise SystemExit("--freeze-enc supports the small_cnn/small_cnn_film trunks only")
    _mods = net.convs if FILM else net
    _n = 10 ** 6 if FREEZE_ENC == "all" else int(FREEZE_ENC)
    _frozen = 0
    for _m in _mods:
        if isinstance(_m, nn.Conv2d):
            if _frozen >= _n:
                break
            for _p in _m.parameters():
                _p.requires_grad = False
            _frozen += 1
    print(f"freeze-enc: froze {_frozen} conv layer(s)", flush=True)
opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), 1e-3)
lossf = nn.CrossEntropyLoss()
BATCH = 32 if ARCH == "efficientnet_b5" else 64 if ARCH == "mobilenet_v3_large" else 128
VAL_BATCH = 64 if ARCH == "efficientnet_b5" else 128 if ARCH == "mobilenet_v3_large" else 256

tr_ids = np.asarray(tr_ids, dtype=np.int64)
va_ids = np.asarray(va_ids, dtype=np.int64)
ks = np.arange(K - 1, -1, -1, dtype=np.int64)
idx_mat = np.maximum(np.arange(n, dtype=np.int64)[:, None] - ks[None, :],
                     run_start[:, None])
imgs_g = torch.from_numpy(imgs).to(dev)
idx_g = torch.from_numpy(idx_mat).to(dev)
y_g = torch.from_numpy(y).to(dev)
tr_g = torch.from_numpy(tr_ids).to(dev)
va_g = torch.from_numpy(va_ids).to(dev)
w_g = torch.tensor(w, dtype=torch.float32, device=dev)
cond_g = torch.from_numpy(cond).to(dev) if FILM else None
corr_x_g = torch.from_numpy(corr_x).to(dev).float().div_(255.0) if n_corr else None
corr_y_g = torch.from_numpy(corr_y).to(dev) if n_corr else None
corr_cond_g = None
if FILM and n_corr:
    # ponytail: corrections are context-less hit clips — speed comes from their own stack
    # (consecutive slots are 1/fps apart), t is neutral 0.5, bonus unknown -> 0. They are
    # only 5x-weighted extras; full-fidelity cond for them isn't worth a second pipeline.
    _cc = np.zeros((n_corr, 3), np.float32)
    _cc[:, 0] = 0.5
    for _j in range(n_corr):
        _px = condition.estimate_scroll(corr_x[_j, -2], corr_x[_j, -1],
                                        META["cond"]["scroll_v"])
        if _px is not None:
            _cc[_j, 1] = min(_px * META["fps"] / META["cond"]["speed_norm"], 2.0)
    corr_cond_g = torch.from_numpy(_cc).to(dev)
# NEGATIVE stacks (opt-in): only the clean "bot was passive -> hit/pit" cases (bot_action==none),
# so we never penalise a jump/slide the bot actually attempted (mis-timed, possibly unavoidable).
NEG_X_G = None
if NEG_NPZ and os.path.exists(NEG_NPZ):
    _nz = np.load(NEG_NPZ, allow_pickle=True)
    _keep = (_nz["bot_action"] == "none")
    _nx = _nz["stacks"][_keep]
    if len(_nx):
        NEG_X_G = torch.from_numpy(_nx).to(dev).float().div_(255.0)
        from collections import Counter as _Cnt
        print(f"negatives: {len(_nx)} 'none' pre-hit/pit stacks (of {len(_nz['stacks'])}) | "
              f"kinds {dict(_Cnt(_nz['kind'][_keep].tolist()))} | unlikelihood lambda={NEG_LAMBDA}", flush=True)
    else:
        print(f"negatives: --neg-npz {NEG_NPZ} had 0 usable 'none' stacks", flush=True)
elif NEG_NPZ:
    print(f"negatives: --neg-npz {NEG_NPZ} not found — training WITHOUT negative signal", flush=True)
del imgs, imgs_all, y_all, notyet_all, idx_mat, notyet, run_start
import gc as _gc
_gc.collect()
print(f"GPU frame bank: {imgs_g.nelement() * imgs_g.element_size() / 1e6:.0f} MB VRAM | "
      f"sampler items: {len(w)} | batch={BATCH}", flush=True)
if wandb_run:
    wandb_run.config.update({
        "batch": BATCH,
        "val_batch": VAL_BATCH,
        "train_class_counts": dict(zip(CLASSES, counts.tolist())),
        "corrections": n_corr,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_frame_bank_mb": imgs_g.nelement() * imgs_g.element_size() / 1e6,
    }, allow_val_change=True)


def demo_stacks(frame_ids):
    return imgs_g[idx_g[frame_ids]].float().div_(255.0)


def fwd(xb, cb=None):
    return net(xb, cb) if FILM else net(xb)


def train_stacks(sample_ids):
    demo_mask = sample_ids < len(tr_ids)
    if bool(demo_mask.all()):
        frame_ids = tr_g[sample_ids]
        return demo_stacks(frame_ids), y_g[frame_ids], \
            (cond_g[frame_ids] if FILM else None)
    xb = torch.empty((sample_ids.numel(), K, H, W), dtype=torch.float32, device=dev)
    yb = torch.empty((sample_ids.numel(),), dtype=torch.long, device=dev)
    cb = torch.empty((sample_ids.numel(), 3), dtype=torch.float32, device=dev) if FILM else None
    if bool(demo_mask.any()):
        frame_ids = tr_g[sample_ids[demo_mask]]
        xb[demo_mask] = demo_stacks(frame_ids)
        yb[demo_mask] = y_g[frame_ids]
        if FILM:
            cb[demo_mask] = cond_g[frame_ids]
    corr_mask = ~demo_mask
    if bool(corr_mask.any()):
        ci = sample_ids[corr_mask] - len(tr_ids)
        xb[corr_mask] = corr_x_g[ci]
        yb[corr_mask] = corr_y_g[ci]
        if FILM:
            cb[corr_mask] = corr_cond_g[ci]
    return xb, yb, cb

def predict_val():
    net.eval(); pr = []
    with torch.no_grad():
        for b in range(0, len(va_ids), VAL_BATCH):
            ids = va_g[b:b + VAL_BATCH]
            xb = demo_stacks(ids)
            pr.append(torch.softmax(fwd(xb, cond_g[ids] if FILM else None), 1).cpu().numpy())
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
    fam = (fire & (yv == 0)).mean() * META["fps"] * 60   # false-fire frames/min at the measured fps
    return len(events), hits, fam

def _augment(xb):
    """Label-preserving train-time augmentation on a (B,K,H,W) [0,1] stack: brightness,
    contrast, small translation, per-batch cutout. NO flips — CookieRun runs strictly L->R."""
    B = xb.shape[0]
    bright = torch.empty(B, 1, 1, 1, device=xb.device).uniform_(-0.10, 0.10)
    contrast = torch.empty(B, 1, 1, 1, device=xb.device).uniform_(0.85, 1.15)
    m = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = (xb - m) * contrast + m + bright
    dx = int(torch.randint(-3, 4, (1,)).item()); dy = int(torch.randint(-2, 3, (1,)).item())
    if dx or dy:
        xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
    if torch.rand(1).item() < 0.4:                     # cutout -> occlusion robustness
        ch, cw = H // 5, W // 5
        yy = int(torch.randint(0, H - ch, (1,)).item()); xx = int(torch.randint(0, W - cw, (1,)).item())
        xb[:, :, yy:yy + ch, xx:xx + cw] = 0.0
    return xb.clamp_(0.0, 1.0)


print(f"augmentation: {'ON' if AUG else 'off'}"
      + (f" | save-best ON" if SAVE_BEST else ""), flush=True)
best = {"score": -1e9, "state": None, "ep": None}
for ep in range(EPOCHS):
    net.train(); tot = 0; tot_neg = 0.0; nb = 0
    sampled = torch.multinomial(w_g, len(w), replacement=True)
    for b in range(0, len(w), BATCH):
        xb, yb, cb = train_stacks(sampled[b:b + BATCH])
        if AUG:
            xb = _augment(xb)
        opt.zero_grad()
        l = lossf(fwd(xb, cb), yb)
        if NEG_X_G is not None:                        # "hit cam" signal on bot failure frames
            ni = torch.randint(0, NEG_X_G.shape[0], (min(BATCH, NEG_X_G.shape[0]),), device=dev)
            nb_x = _augment(NEG_X_G[ni]) if AUG else NEG_X_G[ni]
            logits_n = net(nb_x)
            if NEG_MODE == "jump":                      # label passive-hit/pit frames as JUMP
                l_neg = lossf(logits_n, torch.ones(nb_x.shape[0], dtype=torch.long, device=dev))
                l = l + NEG_JUMP_W * l_neg
            else:                                        # unlikelihood: push p(none) DOWN
                p_none = torch.softmax(logits_n, 1)[:, 0].clamp(max=1 - 1e-6)
                l_neg = -(torch.log1p(-p_none)).mean()   # -log(1 - p_none)
                l = l + NEG_LAMBDA * l_neg
            tot_neg += l_neg.item()
        l.backward(); opt.step()
        tot += l.item(); nb += 1
    if ep % 5 == 4 or ep == EPOCHS - 1:
        ne, hits, fam = event_eval()
        loss = tot / max(nb, 1)
        if SAVE_BEST:                          # sweep's canonical score: hit rate - fam/400
            _score = hits / max(ne, 1) - fam / 400.0
            if _score > best["score"]:
                best = {"score": _score, "ep": ep + 1,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in net.state_dict().items()}}
        print(f"ep{ep+1} loss={tot/max(nb,1):.3f}"
              + (f" neg={tot_neg/max(nb,1):.3f}" if NEG_X_G is not None else "")
              + f" events {hits}/{ne} hit, false-fires/min={fam:.0f}", flush=True)
        if wandb_run:
            wandb.log({
                "epoch": ep + 1,
                "train/loss": loss,
                "val/events": ne,
                "val/hits": hits,
                "val/hit_rate": hits / max(ne, 1),
                "val/false_fires_per_min": fam,
                "gpu/max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
            }, step=ep + 1)

if SAVE_BEST and best["state"] is not None:
    net.load_state_dict(best["state"])
    print(f"save-best: restored ep{best['ep']} (score {best['score']:.3f})", flush=True)

model_name = f"{OUT_PREFIX}.pt"
meta_name = "model_meta.json" if OUT_PREFIX == "model" else f"{OUT_PREFIX}_meta.json"
model_path = os.path.join(OUT, model_name)
meta_path = os.path.join(OUT, meta_name)
torch.save(net.state_dict(), model_path)
json.dump(META, open(meta_path, "w"))
ne, hits, fam = event_eval()
p = predict_val(); pred = p.argmax(1)
cm = np.zeros((3, 3), int)
for t_, p_ in zip(y[va_ids], pred): cm[t_, p_] += 1
print("val confusion (rows=true):", cm.tolist(), flush=True)
print(f">> saved {model_name} + {meta_name} | events {hits}/{ne} | false-fires/min {fam:.0f}", flush=True)
if wandb_run:
    cm_rows = [[CLASSES[i], CLASSES[j], int(cm[i, j])] for i in range(len(CLASSES)) for j in range(len(CLASSES))]
    cm_table = wandb.Table(columns=["true", "pred", "count"], data=cm_rows)
    final_metrics = {
        "final/events": ne,
        "final/hits": hits,
        "final/hit_rate": hits / max(ne, 1),
        "final/false_fires_per_min": fam,
        "final/confusion": cm_table,
    }
    wandb.log(final_metrics, step=EPOCHS)
    artifact = wandb.Artifact(
        name=f"{OUT_PREFIX}-{wandb_run.id}",
        type="model",
        metadata={**META, "events": ne, "hits": hits, "false_fires_per_min": fam},
    )
    artifact.add_file(model_path)
    artifact.add_file(meta_path)
    wandb_run.log_artifact(artifact)
    wandb_run.finish()
