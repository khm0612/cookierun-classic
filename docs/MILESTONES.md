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
1. **[demoted] Gate schedule** — only 5 NEAR falls could benefit; not worth a night. Skip
   unless it rides along free with something else.
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
