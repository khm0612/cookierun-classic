"""Categorize the AI-farm hit diagnostics (data/ai_hits/hits.jsonl) to direct model
improvement. For each hit, look at the model's decision trace in the pre-impact window
[-1.0s, -0.2s] (the span where a dodge decision must fire to clear the obstacle):
  blind            max action prob < 0.30 there — the model saw nothing (data gap)
  hesitant         action prob 0.30-0.60 but below the deploy conf gate (threshold issue)
  fired-but-hit    an action FIRED in-window yet HP still dropped (timing / wrong action)
  cooldown-blocked a jump was wanted but the cooldown suppressed it (cooldown issue)
Each category implies a different fix — count them before changing anything."""
import json, os, sys
from collections import Counter
from _runtime import DATA

OUT = str(DATA / "ai_hits")
W0, W1 = -1.0, -0.2

recs = [json.loads(ln) for ln in open(os.path.join(OUT, "hits.jsonl")) if ln.strip()]
if len(sys.argv) > 1:                      # optional: restrict to run numbers
    keep = {int(a) for a in sys.argv[1:]}
    recs = [r for r in recs if r["run"] in keep]
print(f"{len(recs)} hits")

cats = Counter(); examples = {}
for r in recs:
    win = [t for t in r["trace"] if W0 <= t[0] <= W1]
    if not win:
        cats["no-trace"] += 1; continue
    fired = [t for t in win if t[3] in ("jump", "slide")]
    blocked = [t for t in win if t[1] == "jump-cooldown"]
    action_probs = [t[2] for t in win if t[1] in ("jump", "slide")]
    pmax = max(action_probs, default=0.0)
    if fired:
        cat = "fired-but-hit"
    elif blocked:
        cat = "cooldown-blocked"
    elif pmax >= 0.6:
        cat = "conf-gate-missed"          # argmax was none but an action prob crossed conf?
    elif pmax >= 0.3:
        cat = "hesitant"
    else:
        cat = "blind"
    cats[cat] += 1
    examples.setdefault(cat, []).append(f"r{r['run']:02d}_h{r['hit']:03d}")

print("\ncategory counts:")
for c, n in cats.most_common():
    print(f"  {c:>16}: {n:3d} ({n/len(recs)*100:.0f}%)  e.g. {', '.join(examples.get(c, [])[:4])}")

# fired-but-hit detail: what fired and how long before impact?
lead = []
for r in recs:
    for t in r["trace"]:
        if W0 <= t[0] <= W1 and t[3] in ("jump", "slide"):
            lead.append((t[3], t[0])); break
if lead:
    import statistics
    by = Counter(a for a, _ in lead)
    print(f"\nfired-but-hit detail: {dict(by)}; median lead {statistics.median(d for _, d in lead):.2f}s before impact")
