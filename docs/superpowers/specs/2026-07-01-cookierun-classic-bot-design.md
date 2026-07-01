# CookieRun Classic auto-play bot — Design

**Date:** 2026-07-01
**Status:** Draft for review
**Game:** CookieRun Classic (Devsisters) — Android package `com.devsisters.crg`

## 1. Goal

Build a Python system that plays **CookieRun Classic** on a real Android phone with one
overriding objective: **gather as many resources as possible per run — Coins and
Ingredients — NOT high score / distance.**

At the center is a **Gymnasium-compatible environment** wrapping the phone
(capture → act → reward → auto-reset). A **rule-based computer-vision agent** plays and
survives from day one; that same environment is what a **reinforcement-learning (RL)
agent** trains in later. The rule-based agent doubles as (a) a benchmark and (b) a
behavioral-cloning teacher that warm-starts the RL agent.

### Success metric

**Coins/hour and Ingredients/hour** while playing hands-off — explicitly *not* score or
distance records.

## 2. Game mechanics that drive the design

Verified against the live game and the Cookie Run wikis:

- **Two-button auto-runner.** The cookie runs right automatically. The only inputs are
  **Jump** and **Slide**, shown as two on-screen buttons.
- **Double-jump = tap Jump twice.** Many collectibles/safe-landings require it.
- **Slide = hold** the Slide button (not a tap) for low obstacles; commit early.
- **Button positions are user-swappable** (Devsisters has a swap Jump/Slide setting) →
  button coordinates **must be calibrated per device**, never hard-coded.
- **Coins** = dense pickups (gold + giant coins) scattered along the track; the primary
  farmed currency.
- **Ingredients** come **only from Mystery Boxes**, and there are **exactly 3 Mystery
  Boxes per stage** — sparse, high-value, the sole ingredient source in a run.
- **Energy drains over time; jellies refill it.** Collecting *some* jellies is
  instrumentally required to survive long enough to keep collecting — but jelly points
  (score) are **not** a reward target.
- **Stage-based.** Efficient farming means replaying one good stage (e.g. Episode 1)
  repeatedly. A run ends by crashing, being caught by the Oven, or reaching stage end.

## 3. Objective → reward mapping

The reward function encodes "grab resources, don't chase score":

```
reward_step = w_coin      * Δcoins            # dense, from HUD coin-counter OCR
            + w_box        * mystery_box_hit   # large bonus each of the 3 boxes/run
            + w_survive    * alive_tick         # SMALL — only to stay alive to keep
                                                #   collecting; NOT a distance reward
            - death_penalty (on run end by death)
```

- **No term rewards score, jelly points, or distance.** `w_survive` is deliberately tiny.
- **Authoritative per-run reward:** OCR the **end-of-run results screen** for total
  **Coins** and **Ingredients** earned. Per-tick coin OCR is dense shaping; the
  results-screen totals are the ground-truth episodic reward (robust to per-frame OCR
  noise).
- Weights (`w_coin`, `w_box`, `w_survive`, `death_penalty`) live in `config.yaml`.

## 4. Chosen approach & key technical decisions

- **Capture + control channel:** **scrcpy** (H.264 video stream + touch injection over one
  USB/ADB tunnel, ~30–60 fps, low-latency taps), driven from Python via `scrcpy-client`.
  **Plain ADB** (`screencap` / `input tap`) is the fallback and is used for slow
  menu-level taps. Both sit behind a single `Device` interface. *This is the riskiest
  integration (PyAV + scrcpy-server version matching on Windows) and is proven first.*
- **RL observation:** 4 stacked **84×84 grayscale** crops of the **play area only**
  (HUD and control overlay excluded so the agent can't read the score off-screen).
- **RL algorithm:** **DQN** (off-policy, replay buffer — most sample-efficient for scarce
  real-device samples), via **Stable-Baselines3**. Algorithm stays swappable (PPO/QR-DQN).
- **Warm-start:** record the rule-based agent's `(obs, action)` play → **behavioral
  cloning** to pre-train the policy → **then** DQN fine-tune. This is the payoff of the
  environment-first hybrid; it turns "weeks of random flailing" into something tractable.
- **Language/stack:** Python 3.10+, OpenCV, `scrcpy-client`, Stable-Baselines3,
  Gymnasium, an OCR lib (e.g. Tesseract via `pytesseract`, or a small digit template
  matcher for the HUD counters).

## 5. Architecture & data flow

```
        +--------------- Android phone (USB / ADB) ---------------+
        |              CookieRun Classic running                  |
        +-------^--------------------------------+----------------+
         touch events                       H.264 video
                |                                |
        +-------+--------------------------------v----------+
        |  device.py  - scrcpy capture+control (adb fallback)|
        +-------^--------------------------------+----------+
                | gesture(x,y,action)   raw frame |
        +-------+------+   +-----------+   +-------v----------+
        | gestures.py  |   |  menu.py  |   | capture.py       |
        | action->tap  |   | restart / |   | crop/gray/resize |
        | (Jump/Slide) |   | rewards / |   | /frame-stack     |
        +-------^------+   | setup     |   +-------+----------+
                |          | (allow/   |           |
                |          |  denylist)|   +--------v---------+  +-----------+
                |          +-----^-----+   |   detect.py      |  | reward.py |
                |                |         | death/results/   |->| coins +   |
                |                +---------+ coin+box counters |  | boxes +   |
                |                          | (templates+OCR)  |  | done      |
                |                          +--------+---------+  +-----+-----+
        +-------+-------------------------------------------------+---+------+
        |                 env.py - CookieRunEnv (Gymnasium)                  |
        |  reset() -> fresh run    step(a) -> (obs, reward, done, info)      |
        +-------^-----------------------------------------------^-----------+
                |                                                |
      +---------+----------+                         +-----------+----------+
      | policies/rule_based|  ------ imitate ------> | agents/train.py (DQN)|
      |  survive>coins>box |                         | agents/play.py (live)|
      +--------------------+                         +----------------------+
```

## 6. Components (each file one job, all kept < 500 lines)

| Module | Job |
|---|---|
| `device.py` | `Device` interface: scrcpy capture+control; ADB fallback. The only code that talks to the phone. |
| `capture.py` | Frame preprocessing: crop play-area, grayscale, resize, 4-frame stack. |
| `detect.py` | Template matching + OCR: death/results detection, find named buttons, read **coin counter**, **Mystery-Box ×/3**, and results-screen **coins + ingredients**. |
| `gestures.py` | Config-driven map: action → touch gesture on the **Jump / Slide** buttons (double-jump = two taps; slide = timed hold). |
| `menu.py` | `MenuNavigator` state machine: restart/replay target stage, collect rewards, optional pre-run setup. **Allowlist-only tapping + spend-button denylist.** |
| `reward.py` | Step reward (coins + Mystery Boxes + tiny survival term) + episode-done signal; results-screen authoritative totals. |
| `env.py` | `CookieRunEnv(gym.Env)`: `reset()`, `step()`, action/observation spaces. |
| `policies/rule_based.py` | `RuleBasedAgent`: CV features → action, priority **survive → coins → Mystery Boxes**. Works immediately; teaches the RL agent. |
| `agents/train.py` | SB3 training: BC warm-start from recorded rule-based play, then DQN fine-tune. |
| `agents/play.py` | Run live: rule-based *or* a trained model; outer farm loop over the target stage. |
| `calibrate.py` | Helper: grab a screenshot to crop regions & capture button/counter templates into `config.yaml` + `templates/`. |
| `config.py` | Load/validate `config.yaml` (screen regions, gesture coords, template paths, reward weights, target stage). |
| `metrics.py` | Track coins/run, ingredients/run, coins/hour, ingredients/hour — the real success metric. |

## 7. Action & observation spaces

- **Action space:** `Discrete(3)` — `0 = noop`, `1 = tap Jump`, `2 = Slide (hold)`.
  Double-jump = Jump on two consecutive decision ticks. Small on purpose; expandable.
- **Decision tick:** act every *k* frames at a fixed rate so latency jitter does not wreck
  timing; slide uses a configurable hold duration.
- **Observation (RL):** `Box` of 4 × 84×84 grayscale play-area crops.

## 8. Safety guardrails

- **Never spend currency.** `MenuNavigator` only taps buttons on an *allowlist* of
  templates; any detected spend / revive-with-crystals / purchase / "watch ad" dialog is
  on a *denylist* → the bot backs off and waits, never taps.
- **Device-specific files gitignored:** `config.yaml` + `templates/` are gitignored;
  `config.example.yaml` is committed.
- **ToS caveat (honest heads-up):** automating a game can violate its Terms of Service and
  *could* risk the account. The currency guardrail limits damage, but the risk is not
  zero. This is the user's call.

## 9. Build order (phases)

1. **Device I/O** — scrcpy capture+control + ADB fallback + `calibrate.py`. *(highest risk
   — prove capture + measure real loop latency before anything else)*
2. **Detect + menu + reward** — death/results detection, coin + Mystery-Box reading,
   auto-restart/replay, reward guardrails.
3. **Env** — wrap 1–2 into `CookieRunEnv`; verify with a random agent.
4. **Rule-based agent** — survive → coins → boxes; first genuinely working farming bot;
   wire up `metrics.py`.
5. **RL** — record rule-based play → BC warm-start → DQN fine-tune → `play.py`.
6. **(stretch)** Pre-run setup automation (cookie/pet select) — lowest priority, config-driven.

## 10. Top risks & mitigations

1. **scrcpy/PyAV integration on Windows** → `Device` interface + ADB fallback; prove
   capture in Phase 1.
2. **End-to-end latency** → fixed decision tick + action-repeat; measure real loop time
   early.
3. **Coin/Mystery-Box OCR reliability** → prefer template-matched digit counters over
   general OCR; lean on the results screen for authoritative totals; make regions
   calibratable.
4. **RL sample scarcity** → BC warm-start + off-policy DQN; the rule-based agent always
   works regardless of RL progress.
5. **Template brittleness across game updates** → templates in config; `calibrate.py` to
   re-grab quickly.

## 11. Prerequisites (user environment)

- Python 3.10+
- Phone with **USB debugging on**, `adb` on PATH, **scrcpy** installed
- CookieRun Classic installed and signed in on the phone
- (For OCR path) Tesseract installed, or use the built-in digit-template matcher

## 12. Config schema (sketch)

`config.yaml` (gitignored; `config.example.yaml` committed):

```yaml
device:
  serial: null              # adb device serial; null = first device
  capture: scrcpy           # scrcpy | adb
  max_fps: 60
regions:                    # pixel rects on the captured frame
  play_area: [x, y, w, h]   # RL observation crop (excludes HUD + buttons)
  coin_counter: [x, y, w, h]
  mystery_box_counter: [x, y, w, h]   # the "x/3" indicator
  results_coins: [x, y, w, h]
  results_ingredients: [x, y, w, h]
gestures:
  jump_button: [x, y]
  slide_button: [x, y]
  slide_hold_ms: 300
loop:
  target_stage: "Episode 1"
  decision_hz: 15
reward:
  w_coin: 1.0
  w_box: 50.0
  w_survive: 0.01
  death_penalty: 10.0
menu:
  allowlist_templates: [restart, replay, collect, ok, start]
  denylist_templates: [revive_crystals, buy, purchase, watch_ad]
```

## 13. Explicitly out of scope (YAGNI)

- No PvP / league-ranked optimization beyond resource farming.
- No cloud training infra; training runs locally against the one phone.
- No multi-device orchestration.
- No emulator support in v1 (real phone only, per the chosen environment).
