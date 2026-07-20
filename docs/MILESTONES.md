# Clean-Run Milestones — the no-human improvement roadmap

Objective (standing user directive, 2026-07-12): **runs without hits and without pit falls.**
Ranking metric: **PITS/run** (primary, fps-robust) → contact/min (fps-sensitive!) → survival s → coins.
Constraint for this roadmap: **no human demo recording.** Every step below is fully autonomous.

Account facts that shape the problem: equipped treasures grant **3 Pit Lifts** (Citrus Life
Preserver +9) + **2 revives @55 HP** (Champion's Crown +9), so a run usually ends by *fall
exhaustion*, not HP — every fall removed is directly more survival. Cookie and all three
equipped treasures are maxed; there is no purchasable power lever left.

---

## M0 — DONE 2026-07-13 (commits `1fb31863` + `af5ff477`)

State: hybrid **iql3**@0.50 + **sslfilm_hf4**@0.45 (data/demo/hybrid.json).
Honest baseline on a fresh emulator (@~51fps): **PITS 2.5/run** (3,2,2,3), contact ~19/min,
survival ~250-290s, ~95-100k coins/run.

Shipped:
- `train_iql.py --pit-spread / --pit-oversample` (defaults bit-identical to iql3)
- `train2.py --label-shift-ms` (default 0 = no-op; recorded in meta; inherited via --meta-from)
- **OCR settle fix** (detect.py + farm_common.py): the Result coin tally ANIMATES; the old
  3-stable early exit locked mid-count values → live sessions under-read up to ~45%.
  Now min_settle_s=5 + edge-clip veto + grown ROI + 400k cap + modal consensus.
  → Coin logs BEFORE this commit are unreliable; use wallet deltas for history.
- **Auto emulator refresh**: fps < `AIFARM_FPS_MIN` (default 45, 0=off) for 2 consecutive
  runs → ai_farm exits 17 → monitor runs ldconsole quit/launch + game restart + window fix +
  conditional popup dismissal → relaunches remaining runs (cap 2/batch). UNTESTED live —
  see M1 item 3.

Verdicts baked into this roadmap (do not re-litigate):
- **Flywheel turn 1** (falls added to a human-dominated corpus): PITS −23%. WORKED.
- **Flywheel turn 2** (corpus doubled with *unfiltered* bot recordings): NEUTRAL/plateau.
  iql4a (1s spread) 2.75, iql4b (2s spread) 3.0, iql4c (2s + 6x oversample) 3.0 vs iql3 2.5.
  Wider pit-spread and oversampling are **falsified** as pit levers.
- **Control-arm law**: contact/min swings with fps (~14/min @41fps ≡ ~19/min @51fps for the
  SAME model). Never compare contact across emulator sessions; always run a same-session
  iql3 control arm; judge on PITS.
- Offline gate-recall anti-correlates with live falls. Never promote on it.

Falsified pile (never retry): wider encoder, scroll_v2 retrain, frozen encoder, win_pre
precision labeling, hybrid CV-override, Claude-label DAgger, global jump gate < 0.45,
wider pit-spread, plain pit-oversampling, unfiltered self-play scaling.

---

## M1 — Fall forensics DONE 2026-07-13 → roadmap RE-ROUTED

**M1.1 fall forensics shipped (`scripts/fall_forensics.py`, run over 80 mined falls). The
findings overturn the original M1/M2 priority order — read these before doing anything else.**

Failure-mode breakdown (executed keys, 3s pre-fall window):
- **no-jump 57.5%** (46/80) — DOMINANT. Model never fired a jump.
- **fired-in-time 26.2%** (21/80) — jumped with adequate lead (median 0.89s) and fell anyway.
- **late-jump 8.8%** (7/80) — fired < 0.35s before the prompt (too late).
- **wrong-action 7.5%** (6/80) — slid into the pit (misread obstacle).

**Model replay of the 46 no-jump falls (iql3 base, live gate 0.45) — THE routing result:**
- **BLIND 41/46**: max jump-conf over the WHOLE pre-fall window < 0.30 (median 0.00). The
  model does not see these pits at all.
- **GATED 0/46**: none sat in [0.30, 0.45). **There is no suppressed confidence to un-gate.**
- NEAR 5/46: conf ≥ 0.45 but no executed jump (replay/live stride mismatch; execution gap).

**CONSEQUENCE — a gate schedule cannot fix the dominant mode** (you can't un-gate a 0.00).
This also explains why iql4's wider pit-spread/oversample did nothing and why turn-2 plateaued:
the problem is VISION (the policy is blind to these pits), not gating or imitation quality.
IQL's advantage-weighted BC can only upweight actions *present* in the data, and the data has
almost no successful jumps over the 225-315s pits (humans fell only 2× total in hf2/3/4).

Fall-time histogram: **starkly bimodal** — ~zero falls before 150s, a small cluster 150-180s,
and a massive cluster **225-315s** (the ~55-65k-collected candy gauntlet, long-known). This
window is where any pit-specific intervention must act.

Post-revive clustering: **67%** (32/48 non-first falls within 8s of the previous fall) — the
character revives at/just-before the same hazard and re-falls. Well above the 30% threshold.

**RE-PRIORITIZED next steps (supersedes the old M1.2 gate-schedule plan):**
1. **[DONE 2026-07-20, re-promoted + DEPLOYED] Gate schedule + post-fall caution** — the
   2026-07-17 25-run corpus re-ran the forensics (fresh windows `150-165` + `225-315`, 72%
   post-revive clustering) and the build was cheap after all (~40 lines in ai_farm:
   `AIFARM_GATE_SCHEDULE` / `AIFARM_POSTFALL_S` / `AIFARM_POSTFALL_HAZTHR`, stateless
   per-frame min-against-original). Same-session A/B on the iql5b stack: PITS 2.75 vs 3.0
   and **survival 247s vs 175s (+41%)** — notably it acted as a resilience backstop while
   external GPU load starved the async hazard (fires 10-13 vs 1-6). Deployed via
   hybrid.json `gate_schedule`/`postfall_s`.
2. **Post-revive forced-jump heuristic** (cheap, model-independent, ~15 lines in ai_farm):
   when `pitfall()` fires, script a single cautionary jump ~0.8-1.2s after the revive
   resumes. Attacks the 67% clustering WITHOUT needing the model to see the pit. Needs a
   supervised first batch (live-file change). **Do this first of the live changes.**
3. **Hazard-prediction head (was M4) — PROMOTED to the primary no-human lever.** The 80 falls
   are perfect supervision for "pit within 1.5s"; a small head on the SSL encoder learns to
   SEE the pit from pixels — the one thing gating/imitation can't add. See M4 (now the main line).
4. **Auto-refresh live verification** (`AIFARM_FPS_MIN=200`, 3-run batch) — still worth doing.
5. **Double-jump probe**: the 21 fired-in-time falls may need a second tap. Check hf* demos
   for rapid double-taps near pits; if present, oversample them in M2, else the head (item 3)
   must also trigger a double-tap in the 225-315s band.

## M2 — Selective imitation: the corrected flywheel (one overnight)

**Acceptance: IQL-5 beats same-session iql3 control by ≥0.5 PITS/run on 6-run arms.**

The fix for what turn 2 got wrong — three mechanisms, all on existing data:
1. **Hit-window masking** (`--mask-hits`, ~20 lines in train_iql.py mirroring the pit-window
   code): exclude the 0.5s before each mined hit (cache_hp.npy) from the *imitation* pool —
   those actions caused the hits. Rewards keep seeing everything.
2. **Run-quality gate** (`--min-quality`): only runs with PITS ≤ 1 and contact below the
   batch median enter the corpus (per-run stats are already printed at mine time).
3. **Human-anchor weighting**: hf2/hf3/hf4 weighted ~3x bot runs (--run-weight plumbing
   exists) so the human prior never drowns — the measured failure mode of turn 2.

Loop: 20-run recorded batch with the M1 winner → IQL-5 with all three flags → 6-run arm +
control. Falls still enter the reward (pit windows never masked); only flawed *actions*
stop being imitated. If M1 forensics said "insufficient-jump" dominates, additionally
oversample the double-tap sequences found in hf* demos.

## M3 — The timing attack (one session)

**Acceptance: contact/min −15% vs same-session control at equal-or-better PITS, or fps +10%.**

89% of hits are "fired-but-hit" = actions land ~50-100ms late. Two independent attacks:
1. **Label-shift arm**: train `sslfilm_shift67` (--label-shift-ms 67 ≈ 1 frame @15fps) on
   hf2+3+4, offline sanity via model_score, then 6-run arm on contact. If it helps,
   bracket 33/100. NOTE: train_iql reads keys.json separately and is NOT shifted — add the
   same flag there before mixing shifted BC with IQL artifacts.
2. **Inference fp16 / torch.compile** (env-guarded, ~5 lines in learned.py): every fps
   gained is less reaction latency — the same lateness attacked from the compute side.
   Keep only if the RUN OVER fps stat gains ≥10% with no control-quality regression.

## M4 — Hazard-prediction head — PROMOTED to the primary no-human lever (M1.1 finding)

**Acceptance: PITS ≤ 1.5/run sustained over a 12-run batch.**

M1.1 proved the dominant failure is BLINDNESS (41/46 no-jump falls have jump-conf ~0), so the
only no-human way to reduce falls is to make the policy SEE the pit.

**M4.1 hazard head DONE 2026-07-13 (`scripts/train_hazard.py`) — THE SIGNAL EXISTS.** Small
binary head on iql3's warm-started conv trunk, supervised by "pit-lift prompt within 1.5s"
(109 mined falls across 42 runs), run-level held-out split:
- **AP 0.33 vs 0.011 base rate = ~30x lift** — the pits are genuinely visible in the pixels.
- **Per-pit recall (deployment metric): 75% @thr 0.7 (21/28 held-out pits) at 5.7 false-jumps/min;
  57% @0.9 at 3.7/min; 54% @0.97 at 3.0/min.**
- Conclusion: the blindness is a POLICY limitation, not an information limit — a detector can
  recover most pits with no human data. Confirms the roadmap re-route.

**M4.2 hazard trigger wired + live-tested 2026-07-14 (`policies/hazard_trigger.py`, env-gated,
OFF by default) — MECHANICALLY WORKS, but net-harmful as implemented; two bigger findings:**
- 5-run control (hybrid, no trigger) on a FRESH emulator: **PITS 0,0,0,0,0 = 0.0/run** at fps
  50-54 (bonus-heavy runs). The fresh-emulator hybrid is ALREADY pit-free — **falls correlate
  with DEGRADED fps, not model blindness.** Tonight's earlier 2.5-PITS baseline was a degraded
  (fps 37-45) session. So the mined fall corpus is largely a low-fps artifact.
- hazard arm: the head's EVERY-FRAME forward dropped fps 50->37 and PITS rose to 2/run — even
  on a run where it fired 0 times. Since fps is the dominant fall driver, the compute cost is
  net-negative. A frame throttle (every 3rd, `AIFARM_HAZARD_EVERY`) still left fps ~38-40
  because the per-frame preprocess (crop/gray/resize) is also a cost.
- **BONUS: the M0 auto-refresh FIRED LIVE for the first time** (the hazard's fps drop read as
  degradation), rebooted the emulator in 28s and resumed the batch — validates task #5.
- Run-type variance (bonus fraction) swamped the A/B: control drew bonus-heavy runs (0 PITS),
  hazard drew base-heavy runs (2 PITS) — not comparable.

**REVISED conclusion: the #1 real pit lever is keeping fps HIGH (auto-refresh), not adding a
detector.** The hazard trigger is kept in-tree, OFF by default, for a future async/cheaper
re-do. To make it net-positive it must (a) run on a background thread so it never blocks the
decision loop, or (b) be a tiny distilled net + throttled preprocess, so fps stays >45. Only
then re-test, and only in a controlled base-heavy, fps-matched A/B (this session couldn't
isolate the effect). Realistically, if healthy-fps runs are already ~0 falls, the head's
headroom is small — spend effort on fps stability first.

## M5 — Stretch / opportunistic (attempt when M1-M4 land)

**North star: first 0-fall run streak (3 consecutive), then PITS ≤ 1.0/run average.**

- **Ensemble distillation**: label the whole corpus with the argmax of (iql3 + iql4c +
  sslfilm) and train one student net — ensemble quality at single-model fps.
- **IQL hyper sweep** (flags exist): tau 0.7→{0.8,0.9}, beta 3→{5,8} (concentrates
  imitation on high-advantage actions — a softer cousin of M2's masking), and n-step
  returns (3/5-step, small train_iql change) for faster pit credit propagation.
- **W&B arm tracking**: log every arm's PITS/contact/fps to wandb (project cookierun-sweep)
  so week-over-week trends are visible.
- **Nightly cadence** (optional, user-armed): evening 20-run recorded batch with champion +
  auto-refresh; weekly selective-imitation retrain + control arm; monthly SSL encoder
  refresh on the full corpus.
- NOT recommended: model-based RL / world-model imagination (effort >> expected gain here),
  any global gate below 0.45, anything on the falsified pile.

---

## Standing methodology (applies to every milestone)

- Judge on **PITS only**, 6-run arms minimum (4-run arms have ±0.5 falls of noise), with a
  **same-session iql3 control arm**.
- Every batch: `AIFARM_RECORD=1` (data is free) + auto-refresh on.
- Coins truth = wallet deltas (pre-M0 logs under-read); post-M0 OCR is trustworthy.
- Champion backup before every deploy; deploy via data/demo/hybrid.json; revert = restore
  the JSON.
- Never touch crystals (485). Purchases are the user's decision alone.

## Honest ceiling

M1+M2 should take falls from 2.5 toward ~1-1.5/run; M4 is the credible path under 1.0.
A literal zero-fall *average* likely still needs the one excluded input — a few deep human
demos of the pit sections — because human data is the only source of *correct* pit behavior
and it has not grown since hf4. This roadmap maximizes everything extractable without it.

---

# THE 20-IDEA IMPROVEMENT BACKLOG (2026-07-14)

Full super-detailed menu of every lever worth trying, ranked within groups by expected
falls-reduction-per-effort. Status: **DONE** (shipped this session), **READY** (code in
place, needs a train/validate run), **PLANNED** (spec'd, not coded). Every idea judged on the
same law: **PITS on a 6-run arm with a same-session iql3 control; contact/min secondary; fps
must not regress.** The root problem every perception idea attacks: *the base policy is BLIND
to the normal-stage pits (jump-conf ~0 on 41/46 no-jump falls) and the data lacks correct
pit-jumps, so imitation alone can't fix it.*

## Group A — Perception / hazard-head (the primary no-human lever)

**1. Async hazard trigger — DONE (commit 8916d173).** Inference on a background thread; the
decision loop only stores a frame ref + reads a cached P(pit), so fps is unaffected (the sync
version's −13 fps was the whole reason it backfired). `AIFARM_HAZARD_ASYNC=1` (default).
Next: the live A/B (fps hold + falls drop) is the M4.2 re-test.

**2. Hazard hysteresis + chained double-jump — DONE (this session).** `AIFARM_HAZARD_CONFIRM=N`
requires N consecutive above-thr reads before the FIRST fire (false spikes need N-in-a-row →
rate falls geometrically; a real ~1.5s approach barely notices). `AIFARM_HAZARD_MAXJUMP=2`
lets the trigger fire a chained 2nd jump while P stays high — directly targets the **26%
"jumped-in-time-but-still-fell"** falls (wide pits needing a double jump). Tune CONFIRM=2-3.

**3. Hazard gate-BLEND (soft) instead of hard trigger — PLANNED.** Instead of "force jump at
P≥thr", blend: `effective_jump_conf = base_conf + k·P(pit)`, so the head *nudges* the base
over its 0.45 gate. Problem noted at M4.2: base conf is ~0 on blind pits, so a soft nudge
won't cross the line alone — but a blend with a low k plus the hard trigger as backstop could
cut false fires on ambiguous frames. Worth one arm vs the pure hard trigger.

**4. Bigger/deeper hazard head + augmentation — READY-ish.** Current head = iql3 trunk +
128-wide MLP, AP 0.33 / 75% per-pit recall on 109 falls. Try: (a) a 2-3 layer head, (b) train
augmentation (small time-jitter, brightness, horizontal micro-shift), (c) more epochs with
early-stop on val per-pit recall. Target 85%+ recall. Pure `scripts/train_hazard.py` work,
offline, low-risk. Every recorded batch grows the 109-fall corpus so this compounds.

**5. Temporal-median hazard input (de-animate) — PLANNED.** The card-solver learned that the
game sprites ANIMATE (sparkles) and single-frame matching is noisy; the same likely hurts the
hazard head. Feed a 3-frame temporal median (like `monitor.median_grab`) as the newest slot so
the pit edge is stable. Small change to the worker's frame grab + a retrain.

**6. Focal loss + hard-negative mining for the head — PLANNED.** The head's false-fire rate
(5.7/min @0.7) costs jumps. Retrain with focal loss (down-weight easy negatives) and mine the
hardest negatives (frames the head false-fired on) as extra training negatives. Reduces false
fires without lowering recall — improves the precision/recall knee the trigger rides.

## Group B — Offline RL / imitation (the model itself)

**7. IQL-5 selective imitation — READY (flags committed 2b312492).** `--mask-hits S` zeros the
actor weight on the S s before each hit (stop cloning hit-causing actions), `--human-weight W`
keeps the human prior from drowning in bot self-play, `--min-quality` drops bot runs with >1
fall. THE untested fix for why turn-2 plateaued. Train 3 variants (min-quality / mask+human /
all-three) after the current batch frees the GPU; validate vs iql3.

**8. IQL n-step returns — PLANNED.** Pit credit currently propagates 1 step at a time
(`r + γV(s')`); a fall's −4 reward takes many updates to reach the decision frames. Add
3-/5-step returns (`Σγ^k r + γ^n V`) so pit blame reaches the jump decision faster. ~15-line
change to the Q-target in `train_iql.py`. Could sharpen pit-avoidance value estimates.

**9. IQL hyper-sweep — READY (flags exist).** `--tau {0.8,0.9}` (more optimistic V → imitates
higher-advantage actions), `--beta {5,8}` (concentrates AWBC on the best actions — a softer
cousin of mask-hits). 4-6 short trainings + offline model_score, then live-arm the top 1-2.

**10. CQL-style conservative penalty — PLANNED.** With only 109 falls, IQL can overestimate Q
on out-of-distribution pit-jumps it never saw. Add a small conservative penalty (push down Q on
random/OOD actions) so the policy doesn't confidently pick an unlearned jump. Guards against the
"confident but wrong" failure the DAgger cycles showed.

**11. Potential-based reward shaping toward pits — PLANNED.** The terminal PIT_R is sparse.
Add a dense potential `Φ(s) = -distance_to_next_pit` (estimable offline from the fall-time
histogram / scroll position) so every frame approaching a pit carries a gradient. Potential-
based shaping is policy-invariant (won't change the optimum) but speeds learning.

**12. Ensemble distillation — PLANNED.** Label the whole corpus with the argmax of
(iql3 + iql4c + sslfilm + hazard-gated jumps), then train ONE small_cnn student on those labels.
Gets ensemble-quality decisions at single-model fps (no hybrid double-inference cost). If it
works, it also frees the fps budget the async hazard wants.

## Group C — Data / flywheel

**13. Clean-run "gold" curriculum — PLANNED.** Auto-tag runs by quality (PITS, contact) and
maintain a `gold/` corpus of only the cleanest; retrain weekly on gold + human demos. Prevents
the corpus from drifting toward mediocre self-play (the turn-2 failure mode) automatically.

**14. Auto hard-example mining (self-DAgger) — PLANNED.** After each batch, use the hazard
head to LABEL the frames where the base policy should have jumped but didn't (P(pit) high, no
jump), and add those as high-weight training targets. Closes the loop: the detector teaches the
policy. Auto-labeled (no human), unlike the old DAgger cycles.

**15. Synthetic pit augmentation — PLANNED.** 109 real falls is thin. Composite mined pit
sprites/edges into non-pit frames to synthesize more positive examples for the hazard head
(and IQL pit windows). Standard vision augmentation; addresses the data-scarcity ceiling
directly. Validate the head still generalizes to REAL held-out pits.

**16. Fully-automated retrain loop in self_farm — PLANNED.** `self_farm.py` already has a
promotion gate. Wire: every N runs → mine new falls → train IQL (with M2 flags) → gate-promote
to hybrid.json ONLY if it beats the champion on the held-out score. The flywheel, unattended.
Now UNBLOCKED because card auto-drain + auto-refresh make long unattended runs reliable.

## Group D — Timing / speed (the "fired-but-late" hits + fps)

**17. Label-shift BC — READY (flag committed).** `train2.py --label-shift-ms 67` shifts key
labels ~1 frame earlier to fix the measured 50-100ms "fired-but-hit" lateness (89% of hits).
Train `sslfilm_shift67`, offline-score, then a contact/min arm with same-session control.
Bracket 33/67/100 if it helps. NOTE: also shift train_iql's keys.json read before mixing.

**18. fp16 / torch.compile inference — READY (small).** Every fps gained is less reaction
latency = fewer hits AND fewer falls (the dominant fall driver), and it frees GPU budget for
the async hazard. Env-gated `model.half()` in `learned.py` (~5 lines, default off). Measure the
RUN OVER fps stat; keep only if +10% with no control-quality regression. (Apply after the
current batch — learned.py is on the live path.)

**19. Adaptive jump-cooldown by run speed — PLANNED.** The human double-jumps with gaps down
to 0.12s; a fixed 0.30s cooldown blocks the fast late-run dodges. Scale the cooldown by the
live scroll speed (already estimated for the film cond vector) so late-run (high-speed) allows
tighter double-jumps — helps exactly the 225-315s high-speed pit band where falls cluster.

## Group E — Reliability / ops (makes unattended actually work)

**20. Proactive scheduled emulator refresh + stack watchdog — PARTIALLY DONE, extend.** The
reactive auto-refresh (fps<45) fired live tonight, but the emulator OOM'd at 16.7GB and CRASHED
5× after ~8h. Add: (a) a PROACTIVE `ldconsole quit/launch` every ~90 min or 25 runs BEFORE it
balloons; (b) a top-level watchdog that hard-restarts the whole stack (emulator + game + farm)
if no `RUN OVER` line appears for >6 min (covers the rc=1073807364 terminations + device-offline
wedges seen tonight). Card auto-drain (DONE, commit 781b185f) already removed the biggest
unattended blocker. This closes the last reliability gaps for true set-and-forget farming.

## Realistic combined ceiling of the full backlog

Best case if A(1-6)+B(7-12)+D(17-19) all land: **~0.8-1.2 falls/run mixed average** and
**~10-13 contact/min** at a bulletproof unattended uptime. A consistent **sub-1 fall/run** is
the credible no-human floor. The last stretch to a true 0-fall AVERAGE still needs the one
excluded lever — a few human demos of the 225-315s pit gauntlet — because it is the only source
of *correct* pit-jump behavior; the hazard head can SEE the pit but its forced jumps are
sometimes too late to clear it, and no offline trick manufactures a jump the data never showed.
