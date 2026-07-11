"""DAgger correction labeler — turn the bot's logged failures into training gold.

Every HP hit during ai_farm runs saves what the model saw just before impact
(data/ai_hits/rNN_hMMM_{pre07,pre03,impact}.jpg, newer batches also k0..k{K-1} stack frames).
This tool replays each pre-hit clip and asks YOU what the right move was:

    W = should have JUMPED        S = should have SLID
    N / space = nothing (unavoidable, or acting was wrong)
    X = skip (unsure)             R = replay clip
    B = undo previous label       Q / ESC = save + quit

~1 second per hit. Labels land in data/ai_hits/corrections.jsonl and each labeled image is
SNAPSHOTTED into data/ai_hits/corrections/ — from the EXACT frame shown to you (written from
memory, not re-copied from disk), so a farm running in parallel that overwrites the source
rNN_hMMM file can never make the stored label point at different pixels than you judged.
train2.py mixes these in as high-weight samples — they sit exactly on the model's own failure
distribution, which fresh demos never cover.

Usage:
  python scripts/correct.py            # label everything not yet labeled (newest first)
  python scripts/correct.py stats      # headless: counts of labeled / pending
"""
from __future__ import annotations
import sys
import os
import json
import time
import glob
import re

import cv2

from _runtime import DATA                        # ROOT/data, from the repo refactor

HITS = DATA / "ai_hits"
SNAP = HITS / "corrections"
LABELS = HITS / "corrections.jsonl"
LABEL_KEYS = {ord("w"): "jump", ord("W"): "jump",
              ord("s"): "slide", ord("S"): "slide",
              ord("n"): "none", ord("N"): "none", ord(" "): "none",
              ord("x"): "skip", ord("X"): "skip"}
CLIP = (("pre07", "-0.7s", 500), ("pre03", "-0.3s", 500), ("impact", "IMPACT", 700))
KFRAME_MTIME_TOL = 2.0        # k-frames must be written within this many seconds of pre03

_bad_lines: list[str] = []   # unparseable corrections.jsonl lines, preserved verbatim on save


def load_records() -> list[dict]:
    """Parse corrections.jsonl. Unparseable lines are PRESERVED (in _bad_lines) and written
    back on save, so a hand-edit typo can never silently delete good neighbouring labels."""
    _bad_lines.clear()
    recs = []
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                recs.append(json.loads(s))
            except Exception:
                _bad_lines.append(line)
    return recs


def save_records(recs: list[dict]) -> None:
    tmp = LABELS.with_suffix(".jsonl.tmp")
    body = "".join(json.dumps(r) + "\n" for r in recs)
    if _bad_lines:                                # never drop lines we couldn't parse
        body += "".join(bl + "\n" for bl in _bad_lines)
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(LABELS)


def _kframes_for(stem: str, ref_mtime: int) -> list[str]:
    """k-frames for a hit, numerically ordered (k0..k{K-1}) and mtime-coherent with the
    pre03 that defines the hit. The coherence gate is essential: run/hit numbers reset every
    session, so a later run's kN can sit next to an earlier run's pre03 on disk — mixing them
    would train a human label against a DIFFERENT hit's motion. Incoherent -> drop them all
    (train2 then falls back to replicating the verified pre03)."""
    kept = []
    for kp in glob.glob(str(HITS / f"{stem}_k*.jpg")):
        km = re.search(r"_k(\d+)\.jpg$", os.path.basename(kp))
        if km is None:
            continue
        try:
            if abs(int(os.path.getmtime(kp)) - ref_mtime) <= KFRAME_MTIME_TOL:
                kept.append((int(km.group(1)), kp))
        except OSError:
            continue
    return [p for _, p in sorted(kept)]


def pending_hits(records: list[dict]) -> list[dict]:
    """Unlabeled hits on disk, newest first. Identity = (source path, int mtime) so a hit
    image overwritten by a later session counts as NEW content to label again. Uses TWO bulk
    directory scans (pre03 + all k-frames) rather than a glob per hit — ai_hits holds
    thousands of files, and per-hit globbing made startup take minutes."""
    done = {(r.get("src"), int(r.get("src_mtime", 0))) for r in records}
    # one scan: group every k-frame by stem
    k_by_stem: dict[str, list[tuple[int, str, int]]] = {}
    for kp in glob.glob(str(HITS / "r*_h*_k*.jpg")):
        km = re.match(r"(r\d+_h\d+)_k(\d+)\.jpg", os.path.basename(kp))
        if km is None:
            continue
        try:
            kmt = int(os.path.getmtime(kp))
        except OSError:
            continue
        k_by_stem.setdefault(km.group(1), []).append((int(km.group(2)), kp, kmt))
    out = []
    for p in glob.glob(str(HITS / "r*_h*_pre03.jpg")):
        m = re.match(r"r(\d+)_h(\d+)_pre03\.jpg", os.path.basename(p))
        if not m:
            continue
        try:
            mt = int(os.path.getmtime(p))
        except OSError:
            continue
        if (p, mt) in done:
            continue
        stem = f"r{m.group(1)}_h{m.group(2)}"
        item = {"src": p, "src_mtime": mt, "stem": stem,
                "run": int(m.group(1)), "hit": int(m.group(2)), "frames": {}}
        for tag, _, _ in CLIP:
            fp = HITS / f"{stem}_{tag}.jpg"
            if fp.exists():
                item["frames"][tag] = str(fp)
        # k-frames: numerically ordered + mtime-coherent with THIS pre03 (see _kframes_for)
        item["kframes"] = [kp for _, kp, kmt in sorted(k_by_stem.get(stem, []))
                           if abs(kmt - mt) <= KFRAME_MTIME_TOL]
        if "pre03" in item["frames"]:
            out.append(item)
    out.sort(key=lambda h: -h["src_mtime"])
    return out


def load_traces() -> dict:
    """(run, hit) -> compact 'what the model did' string, best-effort header garnish only."""
    traces = {}
    fp = HITS / "hits.jsonl"
    if not fp.exists():
        return traces
    for line in fp.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        fired = [f"{a}@{dt}s" for dt, c, p, a in r.get("trace", []) if a != "none"]
        traces[(r.get("run"), r.get("hit"))] = (
            " ".join(fired[-4:]) if fired else "(model fired nothing)")
    return traces


def snapshot(item: dict, pre03_img, kframe_imgs) -> dict:
    """Write the EXACT in-memory frames that were displayed (not a re-read from disk) into
    the immutable corrections/ dir. shown == stored == labeled, regardless of a parallel farm
    overwriting the source file between display and keypress."""
    SNAP.mkdir(parents=True, exist_ok=True)
    stamp = f"{item['src_mtime']}_{item['stem']}"
    dst = SNAP / f"{stamp}_pre03.jpg"
    cv2.imwrite(str(dst), pre03_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out = {"img": str(dst.relative_to(HITS)).replace("\\", "/"), "kimgs": []}
    for kbase, kimg in kframe_imgs:               # kbase like 'r03_h007_k2.jpg'
        suffix = kbase.rsplit("_", 1)[-1]         # -> 'k2.jpg'
        kd = SNAP / f"{stamp}_{suffix}"
        cv2.imwrite(str(kd), kimg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out["kimgs"].append(str(kd.relative_to(HITS)).replace("\\", "/"))
    return out


def compose(img, header: str, footer: str):
    big = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(big, 40, 60, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
    cv2.putText(canvas, footer, (12, 745), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, "W=jump  S=slide  N=none  X=skip  R=replay  B=undo  Q=quit",
                (12, 772), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 120), 1)
    return canvas


def label_loop() -> None:
    records = load_records()
    traces = load_traces()
    queue = pending_hits(records)
    if not queue:
        print("Nothing to label -- all hit clips on disk are already labeled.")
        return
    print(f"{len(queue)} hit clip(s) to label ({len(records)} already done). "
          "Window opens now -- keys: W/S/N/X, R replay, B undo, Q quit.")
    win = "correct.py -- what SHOULD the bot have done?"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    idx = 0
    while 0 <= idx < len(queue):
        item = queue[idx]
        # Re-stat at display time: if the farm overwrote this file since the queue was built,
        # sync src_mtime to the content we're about to SHOW so the snapshot stamp + done-key
        # match what the human actually judges (no stale-identity duplicate later).
        try:
            item["src_mtime"] = int(os.path.getmtime(item["src"]))
        except OSError:
            idx += 1
            continue
        item["kframes"] = _kframes_for(item["stem"], item["src_mtime"])
        pre03_img = cv2.imread(item["frames"]["pre03"])
        if pre03_img is None:
            idx += 1
            continue
        # reuse the SAME pre03 array for both display and snapshot (never re-read it) so
        # shown == stored even if the farm overwrites the file mid-label. pre07/impact are
        # only displayed (not snapshotted), so their independent reads are harmless.
        clip = [((pre03_img if tag == "pre03" else cv2.imread(item["frames"][tag])), cap, hold)
                for tag, cap, hold in CLIP if tag in item["frames"]]
        clip = [(f, cap, hold) for f, cap, hold in clip if f is not None]
        kframe_imgs = []
        for kp in item["kframes"]:
            ki = cv2.imread(kp)
            if ki is not None:
                kframe_imgs.append((os.path.basename(kp), ki))
        trace = traces.get((item["run"], item["hit"]), "")
        header = (f"[{idx + 1}/{len(queue)}] run {item['run']} hit {item['hit']}"
                  f"   model: {trace}")
        key = None
        while key is None:
            for f, cap, hold in clip:              # loop the clip until a key arrives
                cv2.imshow(win, compose(f, header, f"frame: {cap}"))
                k = cv2.waitKey(hold) & 0xFF
                if k == 255:
                    continue
                if k in LABEL_KEYS:
                    key = LABEL_KEYS[k]
                elif k in (ord("q"), ord("Q"), 27):
                    key = "QUIT"
                elif k in (ord("b"), ord("B")):
                    key = "UNDO"
                # R (or anything else): just let the clip loop again
                if key:
                    break
        if key == "QUIT":
            break
        if key == "UNDO":
            if records:
                dropped = records.pop()
                save_records(records)
                print(f"  undid: {dropped.get('src')} ({dropped.get('label')})")
                queue = pending_hits(records)       # the undone item re-enters the queue
                idx = 0
            continue
        snap = snapshot(item, pre03_img, kframe_imgs)
        records.append({"img": snap["img"], "kimgs": snap["kimgs"], "label": key,
                        "src": item["src"], "src_mtime": item["src_mtime"],
                        "run": item["run"], "hit": item["hit"],
                        "model_trace": trace, "ts": int(time.time())})
        save_records(records)
        idx += 1
    cv2.destroyAllWindows()
    by = {}
    for r in records:
        by[r.get("label")] = by.get(r.get("label"), 0) + 1
    print(f"Saved {len(records)} label(s) total ({by}) -> {LABELS}")
    print("Next: python scripts/train2.py    (corrections are mixed in automatically)")


def stats() -> None:
    records = load_records()
    queue = pending_hits(records)
    by = {}
    for r in records:
        by[r.get("label")] = by.get(r.get("label"), 0) + 1
    print(f"labeled: {len(records)} {by}"
          + (f"  (+{len(_bad_lines)} unparseable lines preserved)" if _bad_lines else ""))
    print(f"pending: {len(queue)}")
    if queue:
        newest = time.strftime("%H:%M:%S", time.localtime(queue[0]["src_mtime"]))
        print(f"newest pending clip: {queue[0]['stem']} @ {newest}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    else:
        label_loop()
