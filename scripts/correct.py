"""DAgger correction labeler — turn the bot's logged failures into training gold.

Every HP hit during ai_farm runs saves what the model saw just before impact
(data/ai_hits/rNN_hMMM_{pre07,pre03,impact}.jpg, newer batches also k0..k3 stack frames).
This tool replays each pre-hit clip and asks YOU what the right move was:

    W = should have JUMPED        S = should have SLID
    N / space = nothing (unavoidable, or acting was wrong)
    X = skip (unsure)             R = replay clip
    B = undo previous label       Q / ESC = save + quit

~1 second per hit. Labels land in data/ai_hits/corrections.jsonl and each labeled image is
SNAPSHOTTED into data/ai_hits/corrections/ (hit images get overwritten across sessions, so
labels must point at immutable copies). train2.py mixes these in as high-weight samples —
they sit exactly on the model's own failure distribution, which fresh demos never cover.

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
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
HITS = ROOT / "data" / "ai_hits"
SNAP = HITS / "corrections"
LABELS = HITS / "corrections.jsonl"
LABEL_KEYS = {ord("w"): "jump", ord("W"): "jump",
              ord("s"): "slide", ord("S"): "slide",
              ord("n"): "none", ord("N"): "none", ord(" "): "none",
              ord("x"): "skip", ord("X"): "skip"}
CLIP = (("pre07", "-0.7s", 500), ("pre03", "-0.3s", 500), ("impact", "IMPACT", 700))


def load_records() -> list[dict]:
    recs = []
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def save_records(recs: list[dict]) -> None:
    tmp = LABELS.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    tmp.replace(LABELS)


def pending_hits(records: list[dict]) -> list[dict]:
    """Unlabeled hits on disk, newest first. Identity = (source path, mtime) so a hit
    image overwritten by a later session counts as NEW content to label again."""
    done = {(r.get("src"), int(r.get("src_mtime", 0))) for r in records}
    out = []
    for p in glob.glob(str(HITS / "r*_h*_pre03.jpg")):
        m = re.match(r"r(\d+)_h(\d+)_pre03\.jpg", os.path.basename(p))
        if not m:
            continue
        mt = int(os.path.getmtime(p))
        if (p, mt) in done:
            continue
        stem = f"r{m.group(1)}_h{m.group(2)}"
        item = {"src": p, "src_mtime": mt, "stem": stem,
                "run": int(m.group(1)), "hit": int(m.group(2)),
                "frames": {}, "kframes": []}
        for tag, _, _ in CLIP:
            fp = HITS / f"{stem}_{tag}.jpg"
            if fp.exists():
                item["frames"][tag] = str(fp)
        for kp in sorted(glob.glob(str(HITS / f"{stem}_k*.jpg"))):
            item["kframes"].append(kp)
        if "pre03" in item["frames"]:
            out.append(item)
    out.sort(key=lambda h: -h["src_mtime"])
    return out


def load_traces() -> dict:
    """(run, hit) -> compact 'what the model did' string, best-effort garnish only.
    hits.jsonl spans sessions with colliding run/hit numbers; the LAST record wins."""
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


def snapshot(item: dict) -> dict:
    """Copy the labeled images to the immutable corrections/ dir; return relative paths."""
    SNAP.mkdir(parents=True, exist_ok=True)
    stamp = f"{item['src_mtime']}_{item['stem']}"
    out = {"img": None, "kimgs": []}
    src03 = item["frames"]["pre03"]
    dst = SNAP / f"{stamp}_pre03.jpg"
    if not dst.exists():
        shutil.copy2(src03, dst)
    out["img"] = str(dst.relative_to(HITS)).replace("\\", "/")
    for kp in item["kframes"]:
        kd = SNAP / f"{stamp}_{os.path.basename(kp).split('_')[-1]}"
        if not kd.exists():
            shutil.copy2(kp, kd)
        out["kimgs"].append(str(kd.relative_to(HITS)).replace("\\", "/"))
    return out


def compose(img, header: str, footer: str):
    big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)
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
        print("Nothing to label — all hit clips on disk are already labeled.")
        return
    print(f"{len(queue)} hit clip(s) to label ({len(records)} already done). "
          "Window opens now — keys: W/S/N/X, R replay, B undo, Q quit.")
    win = "correct.py — what SHOULD the bot have done?"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    idx = 0
    while 0 <= idx < len(queue):
        item = queue[idx]
        frames = [(cv2.imread(item["frames"][tag]), cap, hold)
                  for tag, cap, hold in CLIP if tag in item["frames"]]
        frames = [(f, cap, hold) for f, cap, hold in frames if f is not None]
        if not frames:
            idx += 1
            continue
        trace = traces.get((item["run"], item["hit"]), "")
        header = (f"[{idx + 1}/{len(queue)}] run {item['run']} hit {item['hit']}"
                  f"   model: {trace}")
        key = None
        while key is None:
            for f, cap, hold in frames:            # loop the clip until a key arrives
                shown = compose(f, header, f"frame: {cap}")
                cv2.imshow(win, shown)
                k = cv2.waitKey(hold) & 0xFF
                if k == 255:
                    continue
                if k in LABEL_KEYS:
                    key = LABEL_KEYS[k]
                elif k in (ord("q"), ord("Q"), 27):
                    key = "QUIT"
                elif k in (ord("b"), ord("B")):
                    key = "UNDO"
                elif k in (ord("r"), ord("R")):
                    pass                            # just let the clip loop again
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
        snap = snapshot(item)
        records.append({"img": snap["img"], "kimgs": snap["kimgs"], "label": key,
                        "src": item["src"], "src_mtime": item["src_mtime"],
                        "run": item["run"], "hit": item["hit"],
                        "model_trace": trace, "ts": int(time.time())})
        save_records(records)
        idx += 1
    cv2.destroyAllWindows()
    done = [r for r in records if r.get("label") != "skip"]
    by = {}
    for r in done:
        by[r["label"]] = by.get(r["label"], 0) + 1
    print(f"Saved {len(records)} label(s) total ({by}) -> {LABELS}")
    print("Next: python scripts/train2.py    (corrections are mixed in automatically)")


def stats() -> None:
    records = load_records()
    queue = pending_hits(records)
    by = {}
    for r in records:
        by[r.get("label")] = by.get(r.get("label"), 0) + 1
    print(f"labeled: {len(records)} {by}")
    print(f"pending: {len(queue)}")
    if queue:
        newest = time.strftime("%H:%M:%S", time.localtime(queue[0]["src_mtime"]))
        print(f"newest pending clip: {queue[0]['stem']} @ {newest}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    else:
        label_loop()
