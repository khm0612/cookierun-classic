"""GPU-parallel sweep launcher — fills VRAM by training config shards concurrently.

Runs `sweep.py <mode> shard i N` in N parallel processes (each loads the cached frame bank and
trains its slice CONFIGS[i::N]), then merges the per-shard results and deploys the overall
winner exactly like the single-process sweep would. Results are identical to a sequential
sweep — only wall-clock changes. The live dashboard (sweep_dash.py) works unchanged: every
shard appends to the same sweep_progress.jsonl.

  python scripts/sweep_par.py r3        # 2 shards (default) on demo2+3+4
  python scripts/sweep_par.py r3 3      # 3 shards (more parallelism, needs a smaller bank)
  python scripts/sweep_par.py hifps 3   # hifps 1-demo bank is small -> 3-4 shards fit

VRAM per shard ~= the full frame bank (NOT shared across processes) + its resized copy + model.
For the 120x280 bank that is ~1.2GB/demo + a ~0.75GB/demo 96x224 copy ~= 2GB/demo/shard. So on
a 16GB card: 3 demos -> 2 shards (~12GB); 2 demos -> 3 shards; 1 small demo -> 4. If a shard
OOMs, lower N. (True max-VRAM would share one bank across shards via CUDA IPC — a bigger change.)
"""
import os
import sys
import time
import glob
import json
import subprocess

import torch

from _runtime import ROOT

_pos = [a for a in sys.argv[1:] if not a.startswith("--")]     # positional args (ignore --flags)
MODE = _pos[0] if _pos else "r1"
NSHARD = int(_pos[1]) if len(_pos) > 1 else 2
# wandb passthrough: all shards share ONE group so their runs appear together in W&B
WANDB_ARGS = []
if "--wandb" in sys.argv:
    grp = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--wandb-group=")),
               f"parsweep-{time.strftime('%Y%m%d-%H%M%S')}")
    WANDB_ARGS = ["--wandb", f"--wandb-group={grp}"]
    WANDB_ARGS += [a for a in sys.argv if a.startswith("--wandb-project=") or a.startswith("--wandb-mode=")]
BASE = str(ROOT / "data")
OUT = os.path.join(BASE, "_hifps_model" if MODE == "hifps" else "demo")
os.makedirs(OUT, exist_ok=True)
PROG = str(ROOT / "sweep_progress.jsonl")
PY = sys.executable

# fresh live log + clean stale shard files
open(PROG, "w").close()
for f in glob.glob(os.path.join(BASE, "_shard_*.pt")):
    os.remove(f)

print(f"launching {NSHARD} parallel shards for mode '{MODE}' -> {OUT}", flush=True)
procs = []
for i in range(NSHARD):
    out = open(str(ROOT / f"sweep_shard_{i}.out"), "w")
    args = [PY, "-u", "scripts/sweep.py"]
    if MODE not in ("r1",):
        args.append(MODE)
    args += ["shard", str(i), str(NSHARD)] + WANDB_ARGS
    procs.append((i, subprocess.Popen(args, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT), out))
    time.sleep(2)   # stagger cache/VRAM allocation so shards don't collide on load

# wait for all shards
for i, p, out in procs:
    p.wait()
    out.close()
    print(f"  shard {i} exited ({p.returncode})", flush=True)

# merge
boards, best = [], None
for i in range(NSHARD):
    f = os.path.join(BASE, f"_shard_{i}.pt")
    if not os.path.exists(f):
        print(f"  !! shard {i} produced no result (check sweep_shard_{i}.out)", flush=True)
        continue
    d = torch.load(f, map_location="cpu", weights_only=False)
    boards += d.get("board", [])
    if d.get("win") and (best is None or d["win"]["score"] > best["win"]["score"]):
        best = d

boards.sort(key=lambda r: -r["score"])
print("\n===== MERGED LEADERBOARD =====", flush=True)
for r in boards:
    print(f"{r['name']:>16}: score {r['score']:.3f} | {r['hits']}/{r['events']} events "
          f"| {r['fam']:.0f} false/min | conf {r['conf']} | ep{r['ep']}", flush=True)

if best is None:
    print("\n!! no shard produced a model — nothing to deploy", flush=True)
    raise SystemExit

win = best["win"]
prev_path = os.path.join(OUT, "sweep_results.json")
prev_score = -1e9
if os.path.exists(prev_path):
    prev = json.load(open(prev_path))
    if prev.get("board"):
        prev_score = prev["board"][0].get("score", -1e9)
if win["score"] <= prev_score:
    print(f"\n>> winner {win['name']} ({win['score']:.3f}) does NOT beat the deployed model "
          f"({prev_score:.3f}) — keeping it.", flush=True)
else:
    torch.save(best["state"], os.path.join(OUT, "model.pt"))
    json.dump(best["meta"], open(os.path.join(OUT, "model_meta.json"), "w"))
    json.dump({"winner": win["name"], "conf": win["conf"], "board": boards},
              open(os.path.join(OUT, "sweep_results.json"), "w"), indent=1)
    print(f"\n>> WINNER {win['name']} saved to {OUT}/model.pt (deploy conf={win['conf']})", flush=True)
