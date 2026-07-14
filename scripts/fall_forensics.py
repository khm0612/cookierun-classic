"""M1.1 fall forensics — WHY the bot falls, from the mined pit corpus (no emulator, no train).

For every mined pit fall it answers three questions that route the rest of the roadmap:

  1. WHEN in a run do falls happen?  -> fall-time histogram (feeds M1.2's gate schedule).
  2. WHAT did the policy do just before?  -> executed-key classification per fall:
       no-jump      : no jump executed in the pre-fall window            (gating/blind problem)
       late-jump    : jump executed but < LATE_S before the fall prompt  (timing problem)
       fired-in-time: jump executed >= LATE_S before, still fell          (needs DOUBLE JUMP / wrong spot)
       wrong-action : a SLIDE executed in the immediate pre-fall window   (misread obstacle)
  3. Do falls CLUSTER after a revive?  -> fraction within POST_REVIVE_S of a prior fall.

Optional model replay (--replay) re-runs the deployed IQL base over the pre-fall frames to
split "no-jump" into GATED (model wanted it, conf below the live gate) vs BLIND (conf ~0),
which decides whether M1.2's gate schedule can even help. Replay uses the on-disk SSL frame
cache, so it never re-decodes JPEGs.

Usage:
    python scripts/fall_forensics.py                 # Part A only (fast)
    python scripts/fall_forensics.py --replay        # + iql3 confidence reconstruction
    python scripts/fall_forensics.py --replay --model iql3   # replay a specific base
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
from _runtime import DATA, recording_is_complete

BASE = str(DATA)
CLASSES = ["none", "jump", "slide"]

# The pit-lift PROMPT is detected some lag after the character actually falls; the jump had to
# fire before the ledge. These windows (seconds, relative to the prompt frame) are deliberately
# generous and reported as distributions so the thresholds are auditable, not load-bearing.
PRE_S = 3.0            # how far back to look for the pre-fall decision
LATE_S = 0.35         # a jump firing within this of the prompt is "too late to matter"
WRONG_SLIDE_S = 1.2   # a slide this close before the prompt = misread (pits need a jump)
POST_REVIVE_S = 8.0   # a fall within this of a prior fall = post-revive cluster
BUCKET_S = 15.0       # fall-time histogram bucket width


def _farg(flag, default=None):
    a = sys.argv[1:]
    for i, t in enumerate(a):
        if t == flag and i + 1 < len(a):
            return a[i + 1]
        if t == flag:
            return True
    return default


def run_dirs_with_falls():
    dirs = sorted(glob.glob(os.path.join(BASE, "botrun_*")) +
                  glob.glob(os.path.join(BASE, "demo_self_*")) +
                  [os.path.join(BASE, r) for r in ("hf2", "hf3", "hf4")])
    out = []
    for d in dirs:
        cp = os.path.join(d, "cache_pits.npy")
        if os.path.exists(cp) and os.path.exists(os.path.join(d, "frames.json")):
            metadata = json.load(open(os.path.join(d, "frames.json")))
            if recording_is_complete(metadata) and len(np.load(cp)):
                out.append(d)
    return out


def load_run(rdir):
    fm = json.load(open(os.path.join(rdir, "frames.json")))
    frames = sorted(fm["frames"], key=lambda f: f["idx"])
    ts = np.array([f["t"] for f in frames], np.float64)
    keys = json.load(open(os.path.join(rdir, "keys.json")))
    pits = np.load(os.path.join(rdir, "cache_pits.npy")).astype(int)
    return frames, ts, keys, pits


def classify_fall(ts, keys, pi):
    """Classify one fall by the executed keys in [t_prompt-PRE_S, t_prompt]."""
    tp = ts[pi]
    win = [k for k in keys if tp - PRE_S <= k["t"] <= tp]
    jumps = [k["t"] for k in win if k["action"] == "jump"]
    slides = [k["t"] for k in win if k["action"] == "slide"]
    last_slide_lead = (tp - max(slides)) if slides else None
    if slides and (tp - max(slides)) <= WRONG_SLIDE_S and (
            not jumps or max(slides) > max(jumps)):
        return "wrong-action", (last_slide_lead, None)
    if not jumps:
        return "no-jump", (None, None)
    lead = tp - max(jumps)                 # lead of the LAST jump before the prompt
    if lead < LATE_S:
        return "late-jump", (lead, len(jumps))
    return "fired-in-time", (lead, len(jumps))


def main():
    runs = run_dirs_with_falls()
    replay = bool(_farg("--replay"))
    model = _farg("--model", "iql3")

    rows = []            # (run, fall_time_s, run_len_s, klass, lead, njumps_in_win)
    per_run_falls = {}   # run -> sorted list of prompt times (for clustering)
    for rdir in runs:
        name = os.path.basename(rdir)
        frames, ts, keys, pits = load_run(rdir)
        run_len = float(ts[-1] - ts[0])
        pts = sorted(float(ts[p]) for p in pits)
        per_run_falls[name] = pts
        for p in pits:
            klass, (lead, nj) = classify_fall(ts, keys, p)
            rows.append((name, float(ts[p] - ts[0]), run_len, klass, lead, nj))

    n = len(rows)
    print("=" * 72)
    print(f"FALL FORENSICS  |  {n} falls across {len(runs)} runs")
    print("=" * 72)

    # ---- Q2: executed-key classification ----
    from collections import Counter
    cc = Counter(r[3] for r in rows)
    print("\n[WHAT the policy did in the {:.0f}s before each fall]".format(PRE_S))
    for k in ("no-jump", "late-jump", "fired-in-time", "wrong-action"):
        c = cc.get(k, 0)
        print(f"  {k:14s} {c:3d}  ({100*c/max(n,1):4.1f}%)")
    leads = [r[4] for r in rows if r[3] in ("late-jump", "fired-in-time") and r[4] is not None]
    if leads:
        leads = np.array(leads)
        print(f"  jump-lead (s) before prompt, of falls that DID jump: "
              f"n={len(leads)} median={np.median(leads):.2f} "
              f"p25={np.percentile(leads,25):.2f} p75={np.percentile(leads,75):.2f}")

    # ---- Q1: fall-time histogram (absolute seconds into run) ----
    print("\n[WHEN falls happen — seconds into run, {:.0f}s buckets]".format(BUCKET_S))
    times = np.array([r[1] for r in rows])
    maxb = int(times.max() // BUCKET_S) + 1
    for b in range(maxb):
        lo, hi = b * BUCKET_S, (b + 1) * BUCKET_S
        c = int(((times >= lo) & (times < hi)).sum())
        bar = "#" * c
        star = "  <-- HOT" if c >= max(3, n / maxb * 1.5) else ""
        print(f"  {lo:5.0f}-{hi:<5.0f}s  {c:3d} {bar}{star}")
    # suggested schedule = contiguous hot buckets merged
    hot = [b for b in range(maxb)
           if int(((times >= b*BUCKET_S) & (times < (b+1)*BUCKET_S)).sum()) >= max(3, n/maxb*1.5)]
    if hot:
        segs, s = [], hot[0]
        for a, b in zip(hot, hot[1:] + [None]):
            if b is None or b != a + 1:
                segs.append((s * BUCKET_S, (a + 1) * BUCKET_S)); s = b
        sched = ",".join(f"{int(a)}-{int(b)}:0.35" for a, b in segs)
        print(f"  suggested AIFARM_GATE_SCHEDULE=\"{sched}\"")

    # ---- Q3: post-revive clustering ----
    clustered = 0
    for name, pts in per_run_falls.items():
        for i in range(1, len(pts)):
            if pts[i] - pts[i - 1] <= POST_REVIVE_S:
                clustered += 1
    multi = sum(1 for pts in per_run_falls.values() for i in range(1, len(pts)))
    print(f"\n[POST-REVIVE clustering] {clustered}/{multi} non-first falls occur within "
          f"{POST_REVIVE_S:.0f}s of the previous fall "
          f"({100*clustered/max(multi,1):.0f}%)  "
          f"{'-> add post-revive caution window' if multi and clustered/max(multi,1) > 0.30 else '-> not a dominant mode'}")

    # ---- optional Part B: model-replay of the no-jump falls ----
    if replay:
        replay_no_jump(runs, rows, model)

    # machine-readable dump for downstream steps
    out = os.path.join(os.path.dirname(__file__), "..", "fall_forensics.json")
    json.dump({"n": n, "by_class": dict(cc),
               "rows": [{"run": r[0], "t": r[1], "run_len": r[2], "class": r[3],
                         "lead": r[4]} for r in rows]},
              open(out, "w"), indent=0)
    print(f"\nwrote {os.path.abspath(out)}")


def replay_no_jump(runs, rows, model):
    """Re-run the base model over the pre-fall window of each NO-JUMP fall and report the max
    jump confidence — splits gating (conf in [0.30, live-gate)) from blind (conf < 0.30)."""
    import torch, cv2
    from cookierun_bot.policies.learned import build_net_from_meta
    meta = json.load(open(os.path.join(BASE, "demo", f"{model}_meta.json")))
    K, H, W = int(meta["K"]), int(meta["H"]), int(meta["W"])
    crop = meta.get("crop", [0.1, 0.2, 1.0, 0.9])
    ctag = "-".join(f"{v:g}" for v in crop)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net_from_meta(torch, meta)
    net.load_state_dict(torch.load(os.path.join(BASE, "demo", f"{model}.pt"), map_location="cpu"))
    net.to(dev).eval()

    GATE = 0.45          # the live jump gate the hybrid runs at
    maxconfs = []
    for rdir in runs:
        cache = os.path.join(rdir, f"cache_ssl_{H}x{W}_{ctag}.npy")
        if not os.path.exists(cache):
            continue
        frames, ts, keys, pits = load_run(rdir)
        imgs = np.load(cache)
        if len(imgs) != len(frames):
            continue
        for p in pits:
            klass, _ = classify_fall(ts, keys, p)
            if klass != "no-jump":
                continue
            lo = np.searchsorted(ts, ts[p] - PRE_S)
            va = np.arange(max(lo, K - 1), p + 1)   # frames with a full K-history
            if len(va) == 0:
                continue
            ks = np.arange(K - 1, -1, -1)
            idx = np.maximum(va[:, None] - ks[None, :], 0)
            with torch.no_grad():
                xb = torch.from_numpy(imgs[idx]).to(dev).float().div_(255.0)
                pr = torch.softmax(net(xb), 1).cpu().numpy()
            jump_conf = pr[:, CLASSES.index("jump")]
            maxconfs.append(float(jump_conf.max()))
    if not maxconfs:
        print("\n[REPLAY] no SSL-cache-backed no-jump falls to replay")
        return
    mc = np.array(maxconfs)
    print(f"\n[REPLAY of {len(mc)} NO-JUMP falls — base={model}, live gate={GATE}]")
    print(f"  GATED  (0.30 <= maxconf < {GATE}): {int(((mc>=0.30)&(mc<GATE)).sum()):3d}  "
          f"-> a lower gate in these windows WOULD have jumped (M1.2 helps)")
    print(f"  BLIND  (maxconf < 0.30)          : {int((mc<0.30).sum()):3d}  "
          f"-> model never saw the pit (needs data/hazard-head, not gating)")
    print(f"  NEAR   (maxconf >= {GATE})        : {int((mc>=GATE).sum()):3d}  "
          f"-> would fire; execution/timing gap")
    print(f"  maxconf distribution: median={np.median(mc):.2f} "
          f"p25={np.percentile(mc,25):.2f} p75={np.percentile(mc,75):.2f} max={mc.max():.2f}")


if __name__ == "__main__":
    main()
