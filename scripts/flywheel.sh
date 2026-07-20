#!/usr/bin/env bash
# AUTONOMOUS IMPROVEMENT FLYWHEEL — "fully automated until a no-hit run" (user directive
# 2026-07-20). Bounded at 3 cycles. Each cycle:
#   A) 10-run RECORDED production batch on the full deployed stack (hybrid.json)
#   B) retrain IQL (M2 recipe: mask-hits + human-weight, freshest-350k budget) -> challenger
#   C) same-session A/B challenger vs current base, 3+3 runs, FULL stack via env
#   D) verdict: challenger deploys only if hits/min improves AND PITS doesn't regress
# A 0-hit run anywhere is the goal condition (monitored externally; the loop still finishes
# its cycle so the corpus keeps what produced it).
set -u
ROOT=/c/Users/singh/Desktop/cookierun-bot/.claude/worktrees/youthful-torvalds-329645
PY="/c/Users/singh/Desktop/cookierun-bot/.venv/Scripts/python.exe"
HYB="C:/Users/singh/Desktop/cookierun-bot/data/demo/hybrid.json"   # Windows path: read by Windows python
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/src"

stack_env() {   # replicate the FULL deployed stack for env-driven arms (env bypasses hybrid.json)
  export AIFARM_HYBRID_CONFS="0.5,0.45"
  export AIFARM_HAZARD="hazard"
  export AIFARM_HAZARD_THR="0.7"
  export AIFARM_GATE_SCHEDULE="150-165:0.35,225-315:0.35"
  export AIFARM_POSTFALL_S="8"
  export AIFARM_POSTFALL_HAZTHR="0.6"
  export AIFARM_RECORD="1"
  export AIFARM_FPS_MIN="0"
}

boot() {
  "$PY" -c "import sys; sys.path.insert(0,'scripts'); import monitor; sys.exit(0 if monitor.refresh_emulator(print) else 1)"
}

echo "=== FLYWHEEL START $(date '+%F %H:%M:%S') ===" | tee "$ROOT/flywheel.log"
boot >> "$ROOT/flywheel.log" 2>&1 || { echo "BOOT FAILED — abort" | tee -a "$ROOT/flywheel.log"; exit 1; }

for CYCLE in 1 2 3; do
  BASE=$("$PY" -c "import json;print(json.load(open(r'$HYB'))['base'])")
  if [ -z "$BASE" ]; then
    echo "!! could not read deployed base from hybrid.json — falling back to iql5b" | tee -a "$ROOT/flywheel.log"
    BASE="iql5b"
  fi
  echo "=== CYCLE $CYCLE start $(date +%H:%M:%S) | deployed base=$BASE ===" | tee -a "$ROOT/flywheel.log"

  # A) production batch (deployed hybrid.json stack — no model env override)
  unset AIFARM_HYBRID 2>/dev/null || true
  stack_env
  unset AIFARM_HYBRID_CONFS AIFARM_HAZARD AIFARM_HAZARD_THR AIFARM_GATE_SCHEDULE AIFARM_POSTFALL_S AIFARM_POSTFALL_HAZTHR
  echo "--- C$CYCLE PROD batch (10 runs, hybrid.json stack) ---" | tee -a "$ROOT/flywheel.log"
  "$PY" -u scripts/run_all.py 10 >> "$ROOT/fw_c${CYCLE}_prod.log" 2>&1
  grep -aE "RUN [0-9]+ OVER" "$ROOT/fw_c${CYCLE}_prod.log" | tail -10 >> "$ROOT/flywheel.log"

  # B) retrain challenger on the freshest corpus (now includes this cycle's runs)
  CHAL="iql6c$CYCLE"
  echo "--- C$CYCLE TRAIN $CHAL ---" | tee -a "$ROOT/flywheel.log"
  "$PY" -u scripts/train_iql.py --out-prefix "$CHAL" --max-frames 350000 \
        --mask-hits 0.5 --human-weight 3 >> "$ROOT/fw_c${CYCLE}_train.log" 2>&1
  if [ ! -f "/c/Users/singh/Desktop/cookierun-bot/data/demo/$CHAL.pt" ]; then
    echo "C$CYCLE: training produced no checkpoint — keeping $BASE, next cycle" | tee -a "$ROOT/flywheel.log"
    continue
  fi

  # C) same-session A/B: challenger then base, 3 runs each, full stack via env
  for ARM in "$CHAL" "$BASE"; do
    stack_env
    export AIFARM_HYBRID="$ARM,sslfilm_hf4"
    echo "--- C$CYCLE A/B arm $ARM ---" | tee -a "$ROOT/flywheel.log"
    "$PY" -u scripts/run_all.py 3 >> "$ROOT/fw_c${CYCLE}_ab_$ARM.log" 2>&1
  done
  unset AIFARM_HYBRID

  # D) verdict + conditional deploy
  "$PY" - "$CYCLE" "$CHAL" "$BASE" <<'PYEOF' >> "$ROOT/flywheel.log" 2>&1
import json, re, sys
cycle, chal, base = sys.argv[1], sys.argv[2], sys.argv[3]
root = r"C:\Users\singh\Desktop\cookierun-bot\.claude\worktrees\youthful-torvalds-329645"
def stats(path):
    hits, pits, n = [], [], 0
    try: text = open(path, encoding="utf-8", errors="replace").read()
    except Exception: return None
    for m in re.finditer(r"RUN \d+ OVER @ (\d+)s \| (\d+) hits \(([\d.]+)/min.*?PITS=(\d+)", text):
        hits.append(float(m.group(3))); pits.append(int(m.group(4))); n += 1
    if not n: return None
    return {"n": n, "hits_min": sum(hits)/n, "pits": sum(pits)/n,
            "zero_hit_runs": sum(1 for h in hits if h == 0.0)}
c = stats(rf"{root}\fw_c{cycle}_ab_{chal}.log")
b = stats(rf"{root}\fw_c{cycle}_ab_{base}.log")
print(f"C{cycle} VERDICT: {chal}={c} vs {base}={b}")
if c and b and c["hits_min"] < b["hits_min"] and c["pits"] <= b["pits"] + 0.25:
    p = r"C:\Users\singh\Desktop\cookierun-bot\data\demo\hybrid.json"
    h = json.load(open(p)); h["base"] = chal; json.dump(h, open(p, "w"))
    print(f"C{cycle} DEPLOYED {chal} (hits {c['hits_min']:.1f}<{b['hits_min']:.1f}, "
          f"pits {c['pits']:.2f} vs {b['pits']:.2f})")
else:
    print(f"C{cycle} KEEPING {base}")
PYEOF
done
echo "=== FLYWHEEL COMPLETE $(date '+%F %H:%M:%S') ===" | tee -a "$ROOT/flywheel.log"
