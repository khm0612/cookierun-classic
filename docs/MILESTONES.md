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

## M1 — Cheap config-level wins (one evening, zero training)

**Acceptance: PITS ≤ 2.0 on a 6-run arm with a same-session iql3 control.**

1. **Fall forensics (do FIRST — it routes everything after).** Pure offline analysis of the
   ~80 mined falls across botrun_*/demo_self_*/hf* recordings (~60-line script):
   - For each fall index in `cache_pits.npy`, inspect the recorded keys/model decisions in
     the 1.5s before it and classify: **(a) no-jump** (never fired → gating problem),
     **(b) late-jump** (fired <300ms before → timing problem), **(c) insufficient-jump**
     (fired in time but fell anyway → needs DOUBLE JUMP; the policy head emits per-frame
     actions so a second tap is expressible, but the model may never have learned it —
     check hf* demos for rapid double-taps near pits).
   - Also test **post-revive clustering**: fraction of falls within 8s of a previous
     pit-lift (disorientation hypothesis) — if >30%, add a temporary post-revive caution
     window (lower gate for 5s after the pitfall detector fires; the detector already runs
     live in ai_farm).
   - Build the **fall-time histogram** (falls vs seconds-into-run, 15s buckets) → feeds item 2.
2. **Data-driven jump-gate schedule** (top expected gain/effort). New env
   `AIFARM_GATE_SCHEDULE="120-180:0.35,240-300:0.35"` (~15 lines in ai_farm agent step):
   inside histogram hot windows the jump gate drops 0.45→0.35; outside, unchanged.
   Rationale: global 0.40 was cleaner but −55s survival (false jumps everywhere); windowed
   gates buy pit-jump recall only where pits live. Wrong jumps are HP-cheap; missed
   pit-jumps are fatal. Expected −0.5 to −1.0 falls/run.
3. **Auto-refresh live verification**: force one trip (`AIFARM_FPS_MIN=200`, 3-run batch),
   watch a full refresh cycle once, then leave default 45 on for every batch forever.
4. **OR-ensemble arm** (config-only): `AIFARM_HYBRID="iql4c,sslfilm_hf4"` + gate schedule.
   iql4c was the contact winner (16.0 vs 19.2 same-session); with schedule-patched pit
   coverage it may beat iql3-alone. 6-run arm.
5. **Per-run zone report**: append fall bucket times to the RUN OVER line (one print change)
   so every future batch refines the histogram for free.

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

## M4 — Hazard-prediction head (the big build, one full session)

**Acceptance: PITS ≤ 1.5/run sustained over a 12-run batch.**

The miners produce perfect supervised labels with zero human effort: "hit within 400ms" /
"pit fall within 1.5s" for every recorded frame. Add a small aux head on the SSL encoder
(reuse pretrain_encoder.py plumbing), train on ~80 falls + thousands of hits, and use it at
inference as a **gate modulator**: `gate = base_gate − k·P(hazard)`. This is M1's time
schedule learned per-frame from pixels — strictly more general, and every recorded batch
improves it. Build only after M1/M2 prove gate modulation moves PITS at all.

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
