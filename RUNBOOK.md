# CookieRun Bot — Runbook

Everything needed to record, train, and run the auto-farmer in one place.

The bot plays **CookieRun Classic** (Episode 1) on **LDPlayer**, farming Coins with
**Double Coins** + boosts active on every run. Dodging is done by an imitation-learning
CNN trained on **your own play**.

---

## 0. Prerequisites (one-time)

| Requirement | Detail |
|---|---|
| **LDPlayer** | CookieRun Classic installed, **2560×1440 landscape**, ADB debugging ON (Settings → Other → "Open local connection"). Window kept **visible & not minimized** while farming. |
| **ADB** | Reachable at `127.0.0.1:5555` (the bot auto-runs `adb connect`). |
| **Python 3.10+** | `pip install -r requirements.txt` |
| **GPU (optional but recommended)** | NVIDIA GPU for capture (`dxcam`) + model (`torch` CUDA). CPU works, just slower. Verify: `python -c "import torch;print(torch.cuda.is_available())"` |
| **Key mapping in LDPlayer** | For recording only: **W → Jump**, **S → Slide**. |
| `config.yaml` | Set to `capture: ldplayer`; serial can be blank when only one emulator is ready. |

All commands below are run from the repo root: `C:\Users\singh\Desktop\cookierun-bot`.

---

## 1. TL;DR — just farm

A trained model already ships in `data/demo/model.pt`.

**⭐ One command — everything at once** (model + logging + card-game solver + adb recovery +
supervisor relaunch, streamed live to console *and* a timestamped `logs/run_*.log`):
```bash
python scripts/run_all.py 15
```
That's the recommended way. It's a thin wrapper over `monitor.py supervise` (below) that adds
the combined live log. Ctrl+C stops the whole stack cleanly.

**Other ways to run:**
```bash
python scripts/monitor.py supervise 15   # same stack, no combined-log tee (what run_all wraps)
python scripts/supervisor.py 15          # ATTENDED: farm only — YOU solve card games (§5)
python scripts/ai_farm.py 3              # plain 3-run batch, no supervisor/monitor
```

All of them run the farming runs with the full Double-Coins gate + Head Start + the learned
dodger and auto-recover from capture/emulator hiccups. `run_all` / `supervise` additionally own
the whole batch: they auto-solve the card game (§5), reconnect adb if the device drops, and
relaunch the farm if the supervisor process itself dies (bounded; and they kill any orphaned
farm first so two farms can never run at once).

**While it runs:**
- 💰 **Double Coins** bought + verified before *every* run (never plays un-boosted).
- ⏩ **Head Start** auto-pressed at each run start.
- 🃏 **Card game** → *attended*: the bot beeps every 20 s and waits for **you** (§5).
  *Unattended (`supervise`)*: the monitor solves it automatically (four decoys + the pose that
  shows as a pair) and saves each board to `data/ai_hits/` for later tuning.
- 📊 **Trustworthy numbers**: a run whose Result screen gets hidden (by a card game / level-up
  modal) is logged as **UNREAD** — banked but uncounted — never a fake 0, and the session
  reports a **wallet ground-truth net** so totals match your in-game balance.
- 🪟 **Keep the LDPlayer window visible and don't move it mid-run** (moving it blinks capture
  for a few seconds until it self-heals).

---

## 2. Full pipeline: record → train → farm

### Step 1 — Record a demo run

```bash
python scripts/recorder.py
```

- Auto-saves to the next free `data/demoN/` folder (won't overwrite older demos).
- Captures frames @ ~30–60 fps + your **W/S keypresses**, on one clock.
- **Play a full run**: activate your boosters, use Head Start, relay after the first death,
  keep going until dead. It stops automatically at the Result screen.
- Confirmed working when you see `INPUT CONFIRMED` in the output.

> More demos = better dodging. 3–4 runs is the sweet spot. Each new recording auto-lands in
> `demo2`, `demo3`, … and training uses them all.

### Step 2 — Train the model

```bash
python scripts/train2.py          # trains on ALL data/demo* recordings, 30 epochs
```

- Behavioral-cloning CNN, runs on GPU if available (prints `device: cuda …`).
- Saves `data/demo/model.pt` + `data/demo/model_meta.json` (architecture/crop/fps live in
  meta so training and inference can never drift).
- Reports held-out event hit-rate + false-fires/min per run.

**Step 2.5 — DAgger corrections (the biggest lever without recording new demos):**

Every farm run auto-logs what the model saw right before each hit. Label those failure
moments and retrain — corrections sit exactly on the model's own mistakes, which demos
never cover:

```bash
python scripts/correct.py         # shows each pre-hit clip; W=jump S=slide N=none X=skip
                                  # ~1s per clip; newest first; Q quits anytime (labels saved)
python scripts/correct.py stats   # how many labeled / pending
python scripts/train2.py          # corrections are mixed in automatically (high weight)
```

Label the newest few hundred (this batch's) and quit — no need to clear the whole backlog.
⚠️ Don't retrain while a farm batch is running (GPU training starves the capture pipeline).

**Optional — hyperparameter sweep** (find the best label window / motion spacing / weighting):

```bash
python scripts/sweep.py           # round 1: 12 configs, saves the winner to model.pt
python scripts/sweep.py r2        # round 2: fine grid + augmentation around the winner
```

The sweep only overwrites `model.pt` if a config **beats** the currently-deployed score, and
writes the winning deploy-confidence to `data/demo/sweep_results.json` (which `ai_farm.py`
reads automatically).

### Step 3 — Farm (see §1)

```bash
python scripts/supervisor.py 15
```

---

## 3. Diagnostics & tuning

| Command | What it does |
|---|---|
| `python scripts/learned_check.py` | One live run with the **learned** model; logs every HP drop + action mix + effective fps. |
| `python scripts/dodge_check.py` | One live run with the old **rule-based** agent (baseline comparison). |
| `python scripts/analyze_hits.py [run#…]` | Categorizes logged hits: *blind* / *fired-but-hit* / *cooldown-blocked* / *hesitant* → tells you **why** it got hit. |
| `python scripts/eval_deploy.py` | Offline sweep of confidence × persistence × cooldown on held-out data → pick an operating point without retraining. |

Hit diagnostics accumulate in `data/ai_hits/` (`hits.jsonl` = source of truth; per-hit
`rNN_hMMM_*.jpg` = what the model saw 0.7 s / 0.3 s before impact).

---

## 4. What each run does (the automated loop)

```
menu → Play → boost screen
   ├─ check 3 boost tiles (HP potion, watch, x2)        ← mandated, verified
   ├─ Multi-Buy → Double Coins banner                   ← verified, else it refuses to play
   ├─ Play → arm Head Start watch → press ⏩ at start
   ├─ learned CNN plays until death (jump / slide)
   ├─ Result → read coins → OK
   ├─ card game?  → STOP, beep, wait for you            ← §5
   └─ back to menu → repeat
```

Spend guardrail: the bot only ever spends on the 3 boost tiles + the Double Coins Multi-Buy
(~4,000 coins/run). It never taps revive / buy / crystal / purchase.

---

## 5. Solving the card game (your job)

When "Surprise! Find the X card!" appears, the bot freezes and beeps. **Rule:**

> Four of the six cards show the **same** (decoy) pose. The answer is the pose that appears
> as exactly a **pair of two**. Tap **both** of those two cards (you get 3 tries).

Ignore what the poses *look* like — trust the pair. It's a **3-round** game; solve each round,
then it auto-continues. (The bot logs a heuristic guess as a hint but never taps.)

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `fps=0` / `BLIND RUN` in the log | Capture went stale after an emulator/window hiccup. The child exits and the **supervisor relaunches** clean. If using `ai_farm.py` directly, just re-run it. |
| Stuck spamming `unrecognized … sending BACK` | Usually a flaky adb moment. The `_safe_to_back` guard prevents BACK from ever quitting the game; if adb is truly offline, run `adb connect 127.0.0.1:5555`. |
| Boost gate stuck on `tile_x2 missing` | Tile icon art drifts with stock/price; the badge-fallback handles it. If it persists, re-crop `templates/tile_x2.png` to the tile **icon only** (top ~62%, no price strip). |
| `device offline` / black screencap | `adb disconnect 127.0.0.1:5555 && adb connect 127.0.0.1:5555`; relaunch the game if needed. |
| Coins read as 0 or wrong | OCR miss on the Result screen; the audit frame is saved to `data/ai_hits/result_rNN.jpg`. The counted total is a floor — the in-game balance is the truth. |
| Model dodges poorly | It's data-bound. Record 1–2 more demos (§2 step 1) and retrain. No amount of tuning beats more of your play. |

---

## 7. Repo map

```
src/cookierun_bot/
  farm.py            navigation (ensure_running) + run loop (play_until_death) + main loop
  farm_common.py     shared helpers: frame reads, timing, template polling, result OCR
  farm_boosts.py     boost gate, Double Coins Multi-Buy, Head Start
  farm_cards.py      card-game detection + stand-down policy
  device.py          capture backends (LDPlayer window-grab / dxcam GPU) + adb input
  detect.py          template matching + digit OCR
  policies/
    learned.py       LearnedAgent (the CNN dodger, GPU inference)
    rule_based.py    old hand-tuned agent (baseline)
scripts/
  run_all.py         ⭐ one-command wrapper: whole stack + combined live log
  monitor.py         card-game solver + adb recovery + `supervise` (owns the batch)
  supervisor.py      crash-relauncher for ai_farm (the farm loop)
  ai_farm.py         runs the model N times (boost gate, Head Start, per-hit diagnostics)
  correct.py         DAgger labeler: turn logged failures into training corrections
  recorder, train2, sweep, *_check, analyze_hits, eval_deploy   (record/train/tune/diagnose)
data/
  demo/, demo2/ …    your recorded runs (frames + keys) — training data
  demo/model.pt      the deployed model
  ai_hits/           per-run hit diagnostics
config.yaml          device + gesture + spending config (gitignored, machine-local)
```
