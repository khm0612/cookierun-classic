"""Held-out dodge-quality scorer — the shared measuring stick for the self_farm promotion
gate (C) and the AWR-recipe experiments (D).

It scores a model the SAME way sweep.py picks its winner: on the last-15% val tail of the
HUMAN demos (demo2/3/4 — the stable expert ground truth), the canonical

    score = hits / events - fam / 400          (best over conf in {0.5, 0.6, 0.7})

where an "event" is a contiguous span of one non-none expert action and a "hit" is the model
firing that same action anywhere in the span at >= conf; `fam` = false-fire frames/min. The
eval is DETERMINISTIC (argmax + softmax, no sampling), so re-scoring the champion always
returns the identical number — the gate can't flap on noise.

The pure helpers (extract_events / event_score / best_conf_score / gate_accepts) import only
numpy so tests exercise them without a GPU; score_model lazily imports torch/cv2/_runtime.

CLI:
    python scripts/model_score.py                       # score the deployed model.pt
    python scripts/model_score.py path.pt meta.json     # score a specific checkpoint
    python scripts/model_score.py --demos demo2,demo3   # choose the eval set
"""
from __future__ import annotations
import numpy as np

EVAL_FPS = 35.0                              # demo2/3/4 recorded at 35fps (fam frames/min scaling)
DEFAULT_EVAL = ["demo2", "demo3", "demo4"]   # human anchors = the stable expert ground truth
CLASSES = ["none", "jump", "slide"]
CONFS = (0.5, 0.6, 0.7)


def extract_events(labels):
    """Contiguous spans of one non-none action -> list of (start, end_inclusive, cls)."""
    labels = np.asarray(labels)
    events, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] != 0:
            j = i
            while j + 1 < n and labels[j + 1] == labels[i]:
                j += 1
            events.append((i, j, int(labels[i])))
            i = j + 1
        else:
            i += 1
    return events


def event_score(events, pred, prob, yv, conf, fps=EVAL_FPS):
    """Canonical (score, hits, fam) at one conf (matches sweep.evaluate). `fire` = a non-none
    prediction above conf; a hit = the correct class fired anywhere in the event span; fam =
    false-fire frames/min over the none-frames."""
    pred = np.asarray(pred); prob = np.asarray(prob); yv = np.asarray(yv)
    fire = (pred != 0) & (prob > conf)
    hits = sum(1 for a, b, c in events if np.any(fire[a:b + 1] & (pred[a:b + 1] == c)))
    fam = float((fire & (yv == 0)).mean() * fps * 60) if len(yv) else 0.0
    score = hits / max(len(events), 1) - fam / 400.0
    return score, hits, fam


def best_conf_score(events, pred, prob, yv, fps=EVAL_FPS, confs=CONFS):
    """Pick the conf that maximizes score (matches sweep.evaluate's best-of-conf)."""
    best = None
    for conf in confs:
        s, hits, fam = event_score(events, pred, prob, yv, conf, fps)
        if best is None or s > best["score"]:
            best = {"score": s, "conf": conf, "hits": hits, "events": len(events), "fam": fam}
    return best


def gate_accepts(champion_score, challenger_score, margin=0.0):
    """The promotion gate: deploy a retrain ONLY if it BEATS the champion by > margin on the
    held-out expert demos. Ties/regressions keep the proven champion — this is what stops the
    survival-noise drift the old blind hot-swap caused. Deterministic scoring => no flapping."""
    return challenger_score > champion_score + margin


def _label_run(ts, keys, win_pre, win_post):
    """Per-frame class labels: point +/- window around each expert key (matches sweep's
    build_labels — no slide-span, so the metric equals the canonical sweep board number)."""
    y = np.zeros(len(ts), np.int64)
    for k in keys:
        cls = CLASSES.index(k["action"])
        lo = np.searchsorted(ts, k["t"] - win_pre)
        hi = np.searchsorted(ts, k["t"] + win_post)
        y[lo:hi] = cls
    return y


# Decoded val-tail frames per (demo, geometry) — the human demos are immutable, so the ~15k-JPEG
# decode is done ONCE per process; every subsequent gate scoring only re-runs (fast) inference.
_VAL_CACHE = {}


def _load_val(rdir, H, W, crop, K, stride, win_pre, win_post):
    """Decode + preprocess ONLY the val-tail frames a K-stack references ([cut-(K-1)*stride .. N)),
    cached by (demo, geometry). Returns (imgs, va, y, start, N) or None if the run has no val
    frames. The 85% training region is never touched — this is what keeps the promotion gate
    affordable inline (demo4 is 65k frames; decoding all of them each retrain would stall the farm)."""
    import os
    import json
    import cv2

    key = (rdir, H, W, tuple(crop), K, stride, win_pre, win_post)
    if key in _VAL_CACHE:
        return _VAL_CACHE[key]
    fj = os.path.join(rdir, "frames.json")
    if not os.path.exists(fj):
        return None
    fm = json.load(open(fj))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    ts = np.array([f["t"] for f in frames])
    y = _label_run(ts, keys, win_pre, win_post)
    N = len(frames)
    cut = int(N * 0.85)
    va = np.arange(cut, N)
    if len(va) == 0:
        return None
    x0f, y0f, x1f, y1f = crop
    start = max(0, cut - (K - 1) * stride)
    imgs = np.zeros((N - start, H, W), np.uint8)
    fdir = os.path.join(rdir, "frames")
    for i in range(start, N):
        im = cv2.imread(os.path.join(fdir, f"{frames[i]['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        h, w = im.shape
        band = im[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]
        imgs[i - start] = cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)
    out = (imgs, va, y, start, N)
    _VAL_CACHE[key] = out
    return out


def score_model(model_path, meta_path, eval_demos=None, data_dir=None, device=None):
    """Load a checkpoint + its meta, evaluate on the held-out human-demo val tails, return the
    canonical best-of-conf score dict {score, conf, hits, events, fam}. Lazily imports torch."""
    import os
    import json
    import torch
    from cookierun_bot.policies.learned import build_net_from_meta

    if data_dir is None:
        from _runtime import DATA
        data_dir = str(DATA)
    meta = json.load(open(meta_path))
    K, H, W = int(meta["K"]), int(meta["H"]), int(meta["W"])
    x0f, y0f, x1f, y1f = meta.get("crop", [0.0, 0.0, 1.0, 1.0])
    win_pre = float(meta.get("win_pre", 0.25))
    win_post = float(meta.get("win_post", 0.03))
    # temporal spacing of the K-stack: the model was trained at meta['fps'] on 35fps demos, so a
    # frame gap of round(35/fps) matches LearnedAgent's inference stacking (stride=1 for fps=35).
    stride = max(1, round(EVAL_FPS / float(meta.get("fps", EVAL_FPS))))
    demos = eval_demos or DEFAULT_EVAL

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    net = build_net_from_meta(torch, meta)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.to(dev).eval()

    crop = [x0f, y0f, x1f, y1f]
    ks = np.arange(K - 1, -1, -1)
    all_yv, all_pred, all_prob = [], [], []
    for name in demos:
        loaded = _load_val(os.path.join(data_dir, name), H, W, crop, K, stride, win_pre, win_post)
        if loaded is None:
            continue
        imgs, va, y, start, N = loaded
        # rebased val-frame K-stack indices, clamped to the loaded window's first frame (matches
        # sweep's np.maximum(..., run_start) and LearnedAgent's oldest-frame padding; only bites
        # for tiny runs where cut < (K-1)*stride — a no-op for the real demos).
        idx = np.maximum((va[:, None] - ks[None, :] * stride) - start, 0)
        imgs_t = torch.from_numpy(imgs).to(dev)
        idx_t = torch.from_numpy(idx).to(dev)
        preds, probs = [], []
        with torch.no_grad():
            for b in range(0, len(va), 512):
                xb = imgs_t[idx_t[b:b + 512]].float().div_(255.0)
                p = torch.softmax(net(xb), 1).cpu().numpy()
                preds.append(p.argmax(1)); probs.append(p.max(1))
        all_yv.append(y[va])
        all_pred.append(np.concatenate(preds))
        all_prob.append(np.concatenate(probs))
    if not all_yv:
        # a normal Exception (NOT SystemExit) so self_farm._gate_score's `except Exception` catches
        # it and the promotion gate fails CLOSED (keeps the champion) instead of killing the loop.
        raise RuntimeError("no eval demos found among: " + ", ".join(demos))
    yv = np.concatenate(all_yv)
    pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)
    return best_conf_score(extract_events(yv), pred, prob, yv)


if __name__ == "__main__":
    import os
    import sys
    from _runtime import DATA

    rec = os.path.join(str(DATA), "demo")
    demos = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--demos" and i + 1 < len(argv):
            demos = [d for d in argv[i + 1].split(",") if d]
        elif a.startswith("--demos="):
            demos = [d for d in a.split("=", 1)[1].split(",") if d]
    positional = [a for a in argv if not a.startswith("--")
                  and argv[argv.index(a) - 1] != "--demos"]
    model = positional[0] if len(positional) >= 1 else os.path.join(rec, "model.pt")
    meta = positional[1] if len(positional) >= 2 else os.path.join(rec, "model_meta.json")
    r = score_model(model, meta, eval_demos=demos)
    print(f"score {r['score']:.4f} | hits {r['hits']}/{r['events']} | "
          f"fam {r['fam']:.0f}/min | conf {r['conf']} | model {os.path.basename(model)}")
