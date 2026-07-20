# CookieRun Bot — Project Reference

> The complete reference for the CookieRun Classic autonomous farming bot: what it is,
> how it plays, how it learns, how it recovers, and every lesson that shaped it.
> Companions: [README.md](../README.md) (setup), [RUNBOOK.md](../RUNBOOK.md) (commands),
> [docs/MILESTONES.md](MILESTONES.md) (improvement roadmap M0–M5).

## Contents

1. [What this project is](#1-what-this-project-is)
2. [Current deployed configuration](#2-current-deployed-configuration-2026-07-20)
3. [Model lineage and measured results](#3-model-lineage-and-measured-results)
4. [Operations catalog](#4-operations-catalog--every-failure-mode-we-have-met-and-its-fix)
5. [Safety rails](#5-safety-rails-deliberate-tested--do-not-fix)
6. Runtime core: capture, device, detection, config
7. Unattended farm stack: run_all → monitor → supervisor → ai_farm
8. Policies and agents: the model zoo at decision time
9. Training pipeline and data flywheel
10. Support tools, tests, and repo layout
11. Reference: environment variables and config.yaml

## 1. What this project is

An autonomous agent that plays **CookieRun Classic** (Episode 1 "Escape from the Oven") on an
**LDPlayer 14** Android emulator, unattended, for hours: it navigates menus, activates boosts
(including the **Double Coins** doubler via Multi-Buy, every run), plays each run with a learned
CNN policy at ~40–60 decisions/sec, survives the game's popups and minigames, recovers from
emulator/ADB failures, records every run for training, and retrains itself from its own play.

**The objective (standing user directive, 2026-07-12): clean play — zero wall hits and zero
pit falls.** Coins are secondary. Runs are ranked on:

| Metric | Meaning | Trust |
|---|---|---|
| **PITS/run** | pit falls detected via the fixed-position "Pit Lift" revive prompt | primary — fps-robust |
| contact/min | hits + rebounds a human would perceive | fps-sensitive; only compare within one emulator session |
| survival s | run length (runs end by *fall exhaustion*, not HP — treasures grant 3 Pit Lifts + 2 revives) | secondary |
| coins/run | wallet delta is ground truth (per-run Result OCR historically under-reads) | secondary |

## 2. Current deployed configuration (2026-07-20)

Deployed via `data/demo/hybrid.json` (env vars override; delete keys to revert):

```json
{"base": "iql5b", "bonus": "sslfilm_hf4", "confs": [0.5, 0.45],
 "hazard": "hazard", "hazard_thr": 0.7}
```

Three models cooperate at decision time:

1. **iql5b** (offline-RL IQL policy, M2 selective imitation: hit-window masking + 3x human
   up-weighting on the freshest-350k budgeted corpus) plays normal stages at jump-gate 0.5.
   Deployed 2026-07-20 after beating iql3 in a same-session tiebreak (PITS 2.25 vs 2.75;
   pooled ≈1.86 vs ≈2.38). Previous base iql3 backed up via hybrid.json.pre-iql5b.
2. **sslfilm_hf4** (SSL-pretrained FiLM-conditioned imitation model) takes over during
   BONUSTIME — it is the cleanest pit-gauntlet dodger.
3. **hazard.pt** (binary "pit within 1.5s" detector) runs on a **background thread** and
   force-jumps when P(pit) ≥ 0.7 outside BONUSTIME — the base policy is *blind* to normal-stage
   pits (M1.1 forensics: jump-conf ≈ 0 on 41/46 no-jump falls), so a detector that sees them
   and fires on its own is the lever that works. Live A/B: **~1.5 falls/run vs 2.67 without**
   (~44% reduction), no hit penalty, fps-safe. The *synchronous* version was net-harmful
   (fps 50→37) — async is not an optimization, it is the difference between helping and hurting.

The boost gate activates every run: the three mandated tiles (HP potion 800c, pocket watch
800c, x2 **score** booster 800c — note x2 doubles XP/score, *not* coins) **plus Double Coins**
via the Random-Boost Multi-Buy (first 1,200c, rerolls 600c; the game auto-stops when Double
Coins lands, so the spend is self-bounding — observed ~9k coins for a ~2× doubling of a
~45–90k run).

## 3. Model lineage and measured results

| Era | Model | Live result (session-comparable) | Verdict |
|---|---|---|---|
| pre-07 | small_cnn "champion" (imitation) | ~39 hits/min jump-spam, 134–326 s | coins-first era baseline |
| 07-12 | **sslfilm_hf4** @ gate 0.45 | 19.9 hits/min, avg 336 s | cleanest dodger; BONUSTIME specialist today |
| 07-12 | plain_hf4 | 153.9k coins/run | best pure-coins alternative |
| 07-13 | IQL-1/2 + pit mining | first honest PITS counts (falls were invisible before) | measurement fixed |
| 07-13 | **iql3** hybrid base | **PITS 2.0/run** (−23%), contact 14.5/min | deployed base since |
| 07-13 | IQL-4 a/b/c (pit-spread/oversample) | 2.75–3.0 PITS vs control 2.5 | falsified — unfiltered self-play does not compound |
| 07-14 | **+ async hazard trigger** | **1.5 falls/run vs 2.67 control** | deployed — the falls fix |
| 07-17 | 25-run production batch | PITS ≈ 1.75/run avg, **two 0-fall runs (363 s / 340 s)** | longest runs = cleanest runs |
| 07-17 | IQL-5 b/c (M2 selective imitation) | iql5c PITS 3,3,3,3 = rejected (min-quality starves critics of fall examples) | falsified |
| 07-20 | **iql5b** | same-session tiebreak: **2.25 vs iql3 2.75**; pooled ≈1.86 vs ≈2.38 | **deployed base** |

Lessons that keep proving themselves:

- **Judge on live PITS with a same-session control arm.** Offline gate-recall anti-correlates
  with fall avoidance; contact/min swings ±40% with fps alone.
- **Labels move *which* obstacle kills, not *whether*** (4 DAgger cycles proved it). Data
  quality levers (masking, human up-weighting, min-quality filtering) beat data quantity.
- **fps is the dominant fall driver.** The two 0-fall runs were the two highest-fps runs.
  External GPU load (a game, an animated wallpaper) directly costs falls.
- **Falls cluster**: 225–315 s into a run (the gauntlet) and within 8 s of a previous fall
  (72% post-revive clustering) — see `fall_forensics.py`; the data-derived gate schedule is
  `AIFARM_GATE_SCHEDULE="150-165:0.35,225-315:0.35"` (not yet wired).

## 4. Operations catalog — every failure mode we have met, and its fix

| Symptom | Root cause | Fix (mostly automated now) |
|---|---|---|
| Farm stuck on boost screen, `ready=False` forever | old wait-for-Double-Coins deadlock | rewritten: one bounded buy attempt, then Play regardless |
| Farm BACK-spams "unrecognized settled modal" | a reward popup nav refuses to confirm (by design: generic Confirm is never tapped, so a purchase can never be auto-confirmed) | banner-gated dismissers in monitor.py: `mysterybox`, `levelup`, `congrats` (they QUEUE — the 4-tap loop drains, next poll self-heals); new popup type ⇒ add template + one `elif` |
| Card game stalls the farm | pair-solver margin < 3 = can't tell the pair | deliberate stand-down; human or Claude solves visually. "Find the jumping card" round: minority animation group = the answer pair (frame-diff clustering; nothing visibly jumps — don't wait for it) |
| fps degrades over hours | emulator RAM bloat / OOM | `AIFARM_FPS_MIN` → exit 17 → monitor `ldconsole quit/launch` refresh (budget 2/batch, then continues degraded rather than thrash) |
| fps low from the start | external GPU load (game/wallpaper) | close them; refresh cannot fix external contention |
| `adb open_transport` crash loop | duplicate/competing adb servers | kill ALL `adb.exe`, start ONE server, reconnect (LDPlayer ships its own adb 1.0.41 — never mix) |
| Every run goes blind mid-batch | Windows display sleep starves dxcam | `SetThreadExecutionState` display-hold in run_all **and** monitor supervise |
| Emulator boots to wrong position/screen | fresh LDPlayer drifts | `_reposition_ld_window` hard-locks to `LD_WINDOW_POS=(3520,60,1600,930)` — **the non-HDR second monitor (DISPLAY2, X≥3440). HDR on the primary breaks capture; never park the emulator there** |
| Recording corrupt after force-kill | manifest written mid-kill | frames.json/keys.json now temp+`os.replace` atomic |
| Training dies silently after corpus load | frame bank (~21.5 KB/transition) exceeds 16 GB VRAM / 31 GB RAM | `--max-frames` budget (keeps all human demos + freshest bot runs, logs drops), bank auto-placement (GPU if it fits, else CPU-streamed), mmap concatenate |
| Batch dies overnight with no error | **Windows Update rebooted the machine** | pause updates / set active hours before long batches |
| Coin totals look wrong | Result-screen OCR animates/misreads | wallet delta is truth; per-run tallies are best-effort |
| Two python procs per farm process | venv python.exe is a launcher shim (parent+child pairs) | cosmetic — do not "fix" the double-process sighting |
| Game sits in Party Run lobby / Life Shop / Show Off after manual play | taps landed on menu items | X-close only; never Share, never buy Lives, never spend crystals (487+ balance is user's) |

## 5. Safety rails (deliberate, tested — do not "fix")

- `menu.is_allowed("confirm") is False`: nav never taps a generic Confirm, so it can never
  auto-confirm a spend/crystal dialog. Reward popups are dismissed only by **banner-gated**
  monitor handlers that fire on a distinctive spend-free banner template.
- The spend-dialog guard (`is_spend_dialog`, denylist `revive_crystals/buy/purchase/watch_ad`)
  blocks *all* tapping while any denylisted template is visible.
- The card solver taps only on margin ≥ 3 boards; low-confidence boards wait (a wrong pair
  costs the reward; the round-clear tap-timing rule: tap 1st pair member, verify removal via
  screenshot, then 2nd).
- One card-tap owner (`monitor.py`); the farm process only pauses and waits.
- Never tap Show Off/Share; never buy Lives; crystals are never spent (`forbid_crystals`).

---

## 6. Runtime core: capture, device, detection, config

The runtime core is four layers: a `Device` protocol with five backends (`device.py`), a canonical-resolution presentation contract every consumer relies on, template/OCR detection (`detect.py`), and a validated YAML config (`config.py`). Everything the higher layers (farm loop, monitor, trainers) do — region crops, template thresholds, tap coordinates — assumes the invariants documented here, most of them earned from live failures whose rationale is preserved in code comments.

### Device abstraction and backend selection

`Device` is a `runtime_checkable` Protocol: `start/stop/last_frame/resolution/tap/hold` (`device.py:43`). `open_device(cfg)` dispatches on `device.capture` in config.yaml (`device.py:888`):

| `capture` value | Class | Capture | Input |
|---|---|---|---|
| `ldplayer` (production) | `LDPlayerDevice` (`device.py:621`) | dxcam DXGI → GDI → adb screencap | adb persistent shell |
| `scrcpy` (default/fallthrough) | `ScrcpyDevice` (`device.py:54`) | H.264 stream over adb | adb (control socket disabled) |
| `adb` | `AdbDevice` (`device.py:316`) | adb screencap per frame | adb persistent shell |
| `bluestacks` | `BlueStacksDevice` (`device.py:451`) | adb screencap | Windows SendInput (`win_input.py`) |
| `network` | `NetworkDevice` (`device.py:507`) | MediaProjection via CR Bridge TCP | AccessibilityService via bridge |

Serial resolution (`select_adb_serial`, `device.py:7`) prefers an explicitly requested serial if present in `adb devices`, else the first connected device.

### The canonical 2560×1440 present space

`LDPlayerDevice._PRESENT = (2560, 1440)` (`device.py:633`). Every frame — dxcam, GDI, or adb — is `cv2.resize`d into this space before being returned, and `resolution` always reports it (`device.py:857`). The class docstring states why: all templates, regions, and gesture coordinates were calibrated at 2560, so the canonical space keeps them valid "no matter the emulator's real render resolution." Consumers of absolute pixels (hp_frac, HUD templates) depend on it, which is why even the adb-only fallback resizes (`device.py:794`). Taps travel the opposite direction: `_scale_tap` (`device.py:860`) converts canonical coords to the device's *real* input resolution captured once at start (`_input_res`, e.g. 1600×900), because `adb input` coordinates are real device pixels.

### Capture backends and when each engages (LDPlayerDevice)

The production ladder, per the class docstring (`device.py:622`): scrcpy's H.264 path collapsed to ~1 fps in-run on this box, CPU GDI caps at ~16 fps, and dxcam (DXGI Desktop Duplication) reads the composited window off the GPU at monitor refresh (~100+ fps) — "what finally gives the bot enough frames to dodge."

1. **Startup calibration** (`device.py:648`): forces the window foreground, takes one adb screencap as ground truth, records `_input_res`, then locates the game render inside the window grab via multi-scale template match (`win_input.match_gamearea`, `win_input.py:106`, scales 0.25–1.05). If the window is minimized/offscreen (`rect[0] <= -30000` or tiny) or the match fails, the device drops to **adb-only mode** permanently for the session.
2. **dxcam init** (`_init_dxcam`, `device.py:674`): creates a camera *per output* and validates each against a GDI grab of the same region — dxcam regions are output-local, and on multi-monitor desktops capturing output 0 unconditionally "silently served wrong/stale content" (live-debugged 2026-07-04: window on monitor 2 at x=3668, dxcam watching the primary → silent ~12 fps GDI fallback that starved the policy). Validation is normalized correlation on mean-subtracted grayscale, threshold 0.6 (`device.py:740`), because GDI/DXGI color pipelines differ (gamma/night-light) and an absolute pixel diff false-rejected the *correct* output. A fresh duplicator yields nothing until the screen changes, so the init jiggles the cursor over that monitor to force a composition (`device.py:709`).
3. **Steady state** (`last_frame`, `device.py:789`): dxcam grab; `None` means "unchanged" → serve the cached presented frame. If dxcam is unavailable, GDI `grab_bbox` of the live window rect (re-queried every frame, so a moved window still works).
4. **Streaming** (`wait_frame`, `device.py:810`): blocks until the screen *changes* (dxcam's change signal), timeout returns the cache. **Self-heal:** a DXGI duplicator "silently dies on desktop events" — observed live as the bot playing *blind* off a cached frame for whole runs — so >3 s without a fresh frame triggers duplicator re-creation (revalidated against GDI), falling to GDI if that fails (`device.py:833`).
5. **`nav_frame`** (`device.py:845`): a sharp adb screencap (~0.5 s) for menu/template navigation only — the window-grab path softens detail (game rendered at ~62% window scale then upscaled), dropping template scores ~0.10 below calibrated thresholds. Menus are static, so slow is fine; in-run reads stay on `wait_frame`.

`ScrcpyDevice` is kept as a support backend: it drives scrcpy-server v1.24 directly over adb (no `scrcpy.Client`, for adbutils 2.x compatibility), tolerates corrupt NALs, has the same `wait_frame` semantics, and self-heals a dead decode thread with a 3 s rate limit (`device.py:229`). Its control-socket touch injection is **disabled** (`device.py:139`): on LDPlayer the socket `send()` succeeded but the server dropped every event, so no fallback fired and *zero taps landed* — input goes through adb instead.

### Input paths

**Persistent adb shell** (`AdbDevice`, `device.py:316`): adbutils `.shell("input ...")` costs ~116 ms *per tap* in adb exec setup — measured, and "a direct cause of late dodges." A single long-lived `adb shell` fed via stdin makes each tap a ~0 ms fire-and-forget write. Key behaviors, each with a live-verified rationale:

| Method | Command | Why |
|---|---|---|
| `tap` (`device.py:412`) | `input touchscreen swipe x y x y 70` | Not `input tap`: on some LDPlayer boots default-source taps stop registering in-game, and instant taps get dropped by stale/restored modals — a press spanning several game frames survives both (verified 2026-07-03) |
| `hold` (`device.py:421`) | same-point swipe with duration | fixed-length hold |
| `press`/`release` (`device.py:427`) | `input motionevent DOWN/UP` | True finger-down for variable-length slides; instant, never queues backlog like a long swipe |
| `back` (`device.py:440`) | `input keyevent 4` | Keyevents register even when touch injection is swallowed; dismisses tap-deaf restored modals |
| `reset_shell` (`device.py:375`) | kill + respawn | Fire-and-forget can't detect a shell whose *remote* adb session died while local adb.exe lives (`poll()` healthy, every line silently discarded — observed live 2026-07-04: verified Play taps not landing for 10+ min). Callers seeing repeated no-effect taps call this |

Every send falls back to a one-shot `self._dev.shell(...)` if the pipe is broken twice.

**win_input** (BlueStacks path): capture stays on adb screencap; taps map a guest-frame pixel through window rect minus chrome (`top_bar`/`right_bar`), aspect-preserving letterbox fit (`map_guest_to_screen`, `win_input.py:13` — pure, unit-testable), then `SetCursorPos` + `mouse_event` click. The cursor physically moves, so the emulator window must stay visible/foreground.

**Gesture layer** (`gestures.py`): actions are NOOP/JUMP/SLIDE. Anti-detection humanization: `_jitter_point` scatters taps with a Gaussian around the button centre ("tapping the SAME exact pixel hundreds of times per run is a trivial server-side bot tell"), `_jitter_hold` jitters durations (`gestures.py:11`). `SlideHold` (`gestures.py:34`) implements variable-length slides via `press`/`release`: down while the model predicts slide, `grace_s` bridges single-frame flickers, deliberately **no time cap** (CookieRun allows indefinite holds; a cap would stand the cookie up mid-tunnel). `force_release` (`gestures.py:91`) re-sends UP unconditionally at run boundaries — a silently-rejected UP would perma-slide the next run. Devices without press/release fall back to one fixed `slide_hold_ms` hold per span.

### TemplateMatcher and digit OCR

`TemplateMatcher` (`detect.py:32`) loads every `templates/*.png` as grayscale; `present(frame, name, threshold=0.8)` and `find` (returns template centre) use `TM_CCOEFF_NORMED`. `detect_death` = `results` or `gameover` template at 0.8 (`detect.py:234`).

`read_int(frame, region, templates_dir)` (`detect.py:225`) tries template digit OCR first, falling through to Tesseract:

- **Segmentation** (`_digit_boxes`, `detect.py:122`): HSV mask (light-on-dark or dark-on-light auto-selected by median V/S), connected components. Wide blobs (w/h > 1.15) are **split**, not dropped: bold comma-grouped balances render touching digits as one component, and the old drop-them code silently truncated leading digits (438,651 read as 8651 — the ~20% result misreads).
- **Classification**: `TM_CCOEFF_NORMED` on a normalized 32×48 glyph (more shape-discriminative than mean-abs-diff, which rewards matching background). Gates `_DIGIT_ACCEPT = 0.35`, `_DIGIT_MARGIN = 0.03` (`detect.py:15`) are deliberately lenient, but any single low-score or near-tie digit makes the *whole* field return `None` → Tesseract — never a silently-wrong partial number.
- **Tesseract fallback** (`detect.py:200`): 3× upscale, Otsu, polarity flip, digit whitelist; any failure degrades to `None` rather than crashing the running bot.

**Result-screen coins hardening** (`detect.py:25`, from the 2026-07-13 offline audit): a fixed crop that clips the digit row mid-panel-animation still "reads" as a confident wrong number (108,963 → 8963/408963/1113063 on shifted ROIs). Defenses in `read_results` (`detect.py:263`): the ROI is grown by `_RESULT_GROW = (40, 12, 8, 24)` px (measured to stay inside the bonus icons and dashed separators); any ink touching within `_RESULT_EDGE_PX = 3` of the crop border **vetoes** the read (checked on the raw mask, since a clipped sliver can vanish from the boxes while its ink hugs the edge, `detect.py:158`); reads above `_RESULT_COINS_MAX = 400_000` (above the ~280k per-run ceiling ×2) are discarded; and there is **no Tesseract fallback for this field** — it can't apply the veto and returned confident wrong numbers exactly when clipped. Unreadable → 0, which the caller's settle loop treats as "keep polling / flag UNREAD, never count garbage." `read_mystery_boxes` clamps the "n/3" counter to ≤3 (`detect.py:242`).

### config.yaml schema

`load_config` (`config.py:114`) raises `ConfigError` on a missing file, missing keys, unknown backend, or (via `_read_bool`, `config.py:98`) a non-boolean spending flag. Sections:

| Section | Keys | Notes |
|---|---|---|
| `device` | `serial`, `capture`, `max_fps` (60), `adb_path` | `capture` ∈ {scrcpy, adb, ldplayer, bluestacks, network} (`config.py:95`) |
| `window` | `title` ("BlueStacks App Player"), `top_bar` (40), `right_bar` (40) | ldplayer/bluestacks window lookup + chrome offsets |
| `loop` | `decision_hz`, `target_stage` | required |
| `regions` | `play_area`, `coin_counter`, `mystery_box_counter`, `results_coins`, `results_ingredients` | all five required (`config.py:91`); `[x, y, w, h]` **on the 2560×1440 canonical frame** |
| `gestures` | see below | |
| `reward` | `w_coin`, `w_box`, `w_survive`, `death_penalty` | all required; feeds the RL scaffold |
| `spending` | see below | all optional |
| `menu` | `allowlist`, `denylist` | required; safety semantics below |
| `phone` | `host`, `port` (8080) | `host` required iff `capture: network` |
| `templates_dir` | string ("templates") | |

**Gestures** (`config.py:21`) — the dataclass comments encode live A/B results: `jump_button`/`slide_button`/`slide_hold_ms` required; `jump_hold_ms` (default 250; production config uses 100 because demo human jumps are ~90 ms taps and a 250 ms hold mistimed landings); `tap_jitter_px`/`hold_jitter_frac` (0 = deterministic; production 50/0.15); `slide_grace_s` (0.40), `slide_min_hold_s` (1.5 — once sliding, a jump does not cut it short inside the window; the comment warns raising min-hold only helps a model that *under-holds correct* slides, since forcing R3's mistimed slides to 0.45 s slid it into pits: 97 s/3k vs 318 s/99k), `jump_cooldown_s` (0.25), `slide_conf` (0.35 — the comment records that R3 needed the strict 0.90 because its low-confidence slides were wrong and a held slide blocks the one-finger jump; a model with a reliable slide head can gate low).

**Spending** (`config.py:57`): `allow_coin_boosts` (default **false**; must be explicitly enabled for the target-video pre-run-boost behavior), `max_boost_cost_per_run` (0; gates the boost purchase — `farm_boosts.py:130` refuses if the fixed cost exceeds it), `double_coins_first_cost` (1200) / `double_coins_reroll_cost` (600) / `max_double_coin_rolls` (3) — the Double Coins roulette budget consumed by `farm_boosts.py:194`, and `forbid_crystals` (default **true**).

**Menu allowlist/denylist — SAFETY semantics** (`menu.py`): the allowlist is the *complete* set of template names the navigator will ever tap (`tap_allowed`, `menu.py:28`), with aliases `play → start/restart/replay`, `openall → collect`. The denylist defines `is_spend_dialog` (`menu.py:16`): if *any* denied template (`revive_crystals`, `buy`, `purchase`, `watch_ad` in the shipped configs) is visible, `tap_allowed` returns without tapping **anything** — "hard guardrail: never tap near a spend dialog" — and `advance` reports `spend_blocked`. `forbid_crystals: true` force-appends `revive_crystals` to the denylist even if the YAML omits it (`menu.py:18`), so a config edit can't accidentally re-enable crystal revives.

### The calibrate flow

`python -m cookierun_bot.calibrate [config.yaml]` (`calibrate.py:9`): loads the config, opens the configured device, waits 2 s for frames, saves one frame as `calibration_screenshot.png`, and prints the device resolution. Calibration itself is manual, as the tool's own output instructs: open the PNG in an image editor, read the pixel rects for each `regions` entry, and crop button/counter images into `templates/`. Because `LDPlayerDevice` presents at canonical 2560×1440, coordinates read off this screenshot are directly valid regardless of the emulator's render resolution. (`LDPlayerDevice` additionally self-calibrates the game-area rect inside its host window automatically at every `start()` — that part needs no human step.)

### Frame preprocessing and the RL scaffold

`capture.py` is small: `preprocess` crops `play_area`, grayscales, and resizes to 84×84 uint8 (`capture.py:7`); `FrameStack(k=4)` maintains the 4-frame observation stack. These feed `CookieRunEnv` (`env.py:17`), a Gymnasium env (obs `(4,84,84)`, 3 actions) whose header comment marks it explicitly as the **planned DQN warm-start scaffold, not wired into the shipped behavioral-cloning pipeline** — intentional, not dead code; don't delete it (or the gymnasium dep) without retiring that plan (`env.py:1`).

---

## 7. Unattended farm stack: run_all -> monitor -> supervisor -> ai_farm

The unattended batch is a chain of four processes, each owning exactly one concern, launched as `python scripts/run_all.py N`. `run_all` tees output and holds the display awake; `monitor.py supervise N` is the single owner of the emulator (device lock + card taps + adb/emulator recovery); `supervisor.py` is the crash-relauncher for the farm child; `ai_farm.py` runs the model, the boost gate, and per-run diagnostics. Progress is counted by exactly one signal — the `>> RESULT:` line ai_farm prints per completed run — parsed independently by supervisor (`scripts/supervisor.py:37`) and monitor (`scripts/monitor.py:559`), so a crash never loses banked runs.

### Ownership chain

| Layer | Entry | Owns | Key exit/limit |
|---|---|---|---|
| `run_all.py` | `main(n)` (`scripts/run_all.py:50`) | Combined `logs/run_*.log` tee, display-awake hold (`run_all.py:30`), Ctrl+C handoff (waits 30s for monitor cleanup, then kills, `run_all.py:91`) | rc 130 on Ctrl+C |
| `monitor.py supervise` | `_main_with_device_lock` (`monitor.py:749`) | Device lock (one monitor per serial, `monitor.py:109`), card taps, popup dismissers, adb reconnect, emulator refresh, supervisor relaunch | `MAX_SUP_RELAUNCH=3` consecutive no-progress relaunches (`monitor.py:64,584`) |
| `supervisor.py` | `main(target)` | Relaunching ai_farm with the *remaining* count; passing rc 17 upward | Hard fault after 2 consecutive zero-run attempts (`supervisor.py:54`) |
| `ai_farm.py` | module main loop | LearnedAgent decisions, boost gate (via `farm.ensure_running`), hit/pit diagnostics, per-run recording | exit 2 = blind run / recorder failure; exit 17 = fps-degraded (`ai_farm.py:463,500,515`) |

Cross-worktree safety: the monitor tags every child with `--cookierun-session=<sha256(serial)[:16]>` (`monitor.py:55`), and `_kill_stray_farm` (`monitor.py:483`) kills any python matching that tag before every (re)launch — Windows doesn't reap a dead parent's children, and two farms on one emulator over one adb is "the catastrophic double-tap case". The pattern never matches monitor.py itself.

### Navigation state machine: `ensure_running` (`farm.py:43`)

Drives any settled post-run screen back into a live run, tapping **only buttons visible by image, never a spend/revive button**. Core mechanics:

- **Settled screens** — acts only when `_diff(snapshot, prev) < 2.5` (`farm.py:86`): post-run modals ignore taps during intro animation, and scrcpy stops pushing frames on a static screen, so "settled" = "frame stopped changing" (~1 tap per screen instead of 8-9). Screens that never go pixel-static (button gleams) act after the same safe template is seen 3 consecutive times (`farm.py:88-99`, marked `ponytail:`).
- **Spend-dialog guard** — every iteration first checks `MenuNavigator.is_spend_dialog` (`farm.py:81`, `menu.py:16`): any configured denylist template (plus `revive_crystals` when `forbid_crystals`) blocks all action that cycle.
- **Safe-action ladder** (`farm.py:193-199`): `openall`, `confirm`, `confirm2`, `ok`, `close`, `close2` at 0.82 (high threshold because the League leaderboard's green message buttons false-matched and opened Friend's Info / Medal Shop popups), then `play` at 0.80. Every name is filtered through the menu **allowlist** (`farm.py:200`); in the deployed config `is_allowed("confirm")` is False by design, so nav never taps a generic teal Confirm — that is precisely why a purchase-confirm can never be auto-tapped, and why the monitor's banner-gated dismissers exist (`monitor.py:693-699`).
- **Play handoff** — the menu Play and boost-screen Play both match `play`. Before any generic Play tap, nav polls for the boost screen (`tile_hp`/`chesttile`, `farm.py:206-222`); once `boost_seen`, generic Play is never tapped again — a bare tap would start a boost-less run (the exact live failure that kept Double Coins off). The boost screen itself is keyed on the always-visible tile grid, not `multibtn`, because the right-hand panel cycles (`farm.py:118-125`). `close`/`close2` are vetoed on the boost screen — they'd X the panel and hide the tile grid (`farm.py:202-205`).
- **BACK gating** — two triggers: every 4th repeated no-effect tap (`farm.py:239-246`; also respawns the persistent input shell, whose remote session dies silently) and 3+ consecutive unrecognized-settled frames (`farm.py:255-269`). Both go through `_safe_to_back` (`farm_common.py:300`): BACK is vetoed if the monitor's `card_active` flag is fresh (<90s mtime — a stray BACK forfeits the card and once walked out to the Android launcher), if the frame is blank/broken (`std<12` — flaky captures once BACK-BACK-quit the game), or if Play is visible (BACK on the menu opens the quit dialog). BACK is never sent for a stuck `play` tap for the same reason.
- Nav taps are paced by `_wait_for_change` + 0.35s (`farm.py:247-252`) so the fire-and-forget adb shell can't queue taps that land on the *next* screen.

### Boost gate end-to-end

**Three mandated tiles** (`farm_boosts.py:29`, user directive "always check this three options"):

| Tile | Cost/run | Required? |
|---|---|---|
| `tile_hp` (HP potion+) | 800c | hard |
| `tile_watch` (pocket watch) | 800c | hard |
| `tile_x2` (x2 Point Booster) | 0 (owned stock) | best-effort (`_OPTIONAL_TILES`) — depleted stock must not halt the farm |

`ready_to_play` = all three tiles checked **and** `dblbanner` present (`farm_common.py:77`). If already ready, nav presses Play immediately — re-running the gauntlet only risks a re-check flake toggling a tile OFF (`farm.py:147-159`). Otherwise `ensure_run_boosts` (`farm_boosts.py:122`) verifies each tile: fast path reads the green-check badge at fixed grid centers (`_TILE_CENTERS`, `farm_boosts.py:72`) on a ~3ms dxcam frame (`_boost_read_fast`, `farm_common.py:97`) because icon templates rot with stock/price art while the badge scores 0.96-1.0; unchecked tiles get **exactly one** enable tap then poll-verify (a second blind tap would toggle it back off, `farm_boosts.py:163`).

**Double Coins Multi-Buy** (`buy_double_coins`, `farm_boosts.py:177`, live-mapped 2026-07-18): if `dblbanner` isn't already up, the flow is `chesttile` (pink ? Random-Boost tile) -> `multibtn` (Multi pill) -> `pickboosts` dialog -> Double Coins row (`dblcheck`, re-checked via `dblrow` if needed) -> `multibuy`, which starts the *game's own* roll loop (first roll 1,200c, rerolls 600c) that stops itself when a selected boost lands; success = `dblbanner` above Play. Every step is template-gated, so a missing screen just skips. **Never-stall semantics** (user directive 2026-07-17): one bounded attempt per navigation cycle — a 75s watchdog stops *waiting* (never the game), and on any failure the caller sets `cycle["double_coin_failed"]` and presses Play without the doubler (`farm.py:167-179`); the old wait-for-banner loop deadlocked whenever the game withheld the offer.

**Head Start**: `_watch_headstart` (`farm_boosts.py:37`) is armed immediately after the boost Play tap — the prompt fires the instant the run starts and post-detection watchers missed the window. Position gate (centre box, y<130 offset excludes the bottom-HUD ⏩ icon that got dead-tapped) + two consecutive matches within 20px = animation settled, tap the live match. `play_until_death` keeps a guarded in-run fallback (`farm.py:356-387`); ai_farm deliberately does *not* add a third watcher (raced stale frames, `ai_farm.py:280`). The `_HS_STOCK_BOX` badge read (`farm_boosts.py:88`) is the eyes-free proof: stock decrements on real activation.

### Card-game solver: monitor owns all taps

The farm side **stands down**: `_cardgame` (`farm_cards.py:65`) logs, beeps, and loops until the template disappears — "one tap owner is safer than coordinating two independent solvers" (`farm_cards.py:77`). A None frame is *not* a cleared screen (`farm_cards.py:84-88`).

The monitor's solver (`monitor.py:240`) is heavily gated because earlier auto-tapping was "very dangerous": 2-consecutive-poll settle gate (with the `card_active` flag set on the *first* sighting to close the cold-start BACK race, `monitor.py:678-681`), same-frame re-check each round, a Play-visible veto (menu mis-detect ⇒ never restart/tap — restarting spawned the 2026-07-05 launcher wedge, `monitor.py:259`), and `median_grab` (5-frame pixel median, `monitor.py:169`) to de-animate sprite sparkles — the documented root cause of low-confidence boards. `_card_pair` (`farm_cards.py:31`) exploits the confirmed board structure (4 identical decoys + 2 identical answers): the answer pair has the *lowest* average NCC to the group; `margin` measures the 2-vs-4 cluster gap. Below `MARGIN_OK=3.0` the solver **stands down fail-closed** — `_wait_for_manual_card_clear` (`monitor.py:204`) keeps the flag armed and pings every 20s ("wrong card taps cost more than pausing"). Confident rounds tap each card once, wait ~4s (user rule), cap at 6 rounds, and leave protection armed on cap. Every board is saved full-res to `data/ai_hits/` for offline tuning (`monitor.py:193`).

### Banner-gated popup dismissers

Nav won't tap generic Confirm (spend safety, above), so benign reward popups would get BACK-wedged. The monitor dismisses them, but **only** while a distinctive, spend-free banner template is confirmed present on the same frame — so a purchase/crystal dialog can never be tapped (`dismiss_modal`, `monitor.py:291`). `_dismiss_modal_safely` (`monitor.py:312`) serializes with the card solver lock and re-checks after locking; `card_active` is held during the dismiss so nav stands down.

| Popup | Banner template | Confirm tap (adb coords) | Notes |
|---|---|---|---|
| League Results (weekly) | `league_results` | (1280, 1210) | the original 2026-07-05 BACK-spam wedge |
| Mystery Box | `mysterybox` | (1285, 1247) | ~1/run without it (`monitor.py:693`) |
| Level Up | `levelup` | (1267, 1235) | rare; crystals+coins reward (`monitor.py:702`) |
| Congratulations (challenge) | `congrats` | (1280, 1143) | popups QUEUE (3 back-to-back live 2026-07-17); the 4-tap loop drains, next poll self-heals (`monitor.py:710`) |

### Recovery machinery

- **adb reconnect**: 3 consecutive `adb exec-out screencap` failures trigger `adb reconnect` + `adb connect` (`monitor.py:63,144`); applied in the watch loop and card-wait loop.
- **fps-degrade / emulator refresh (rc=17)**: LDPlayer degrades over long batches (measured 70→37 fps over ~26h). ai_farm exits 17 at a **run boundary only** (RESULT already printed, so the run is counted) after 2 consecutive runs below `AIFARM_FPS_MIN` (default 45), never on the last run (`ai_farm.py:502-515`). Supervisor passes 17 up (`supervisor.py:41`); the monitor runs `refresh_emulator` (`monitor.py:422`): ldconsole quit/launch, poll `sys.boot_completed` via `adb connect` + getprop (**never** `adb wait-for-device` — it hangs on the 5555 transport), `am start` the game (it doesn't auto-start), reposition/front the window to `LD_WINDOW_POS` (dxcam captures the desktop window; a fresh boot drifts geometry), then `_dismiss_startup_popups` — a ladder that never blind-taps the News-X spot (2255,148), because with no popup up it's the friends-list heart. **Refresh budget** `MAX_EMU_REFRESH=2` per batch; exhausted ⇒ relaunch with `AIFARM_FPS_MIN=0` to finish the batch degraded-but-whole instead of cycling the emulator (`monitor.py:565-583`). Refresh relaunches skip the no-progress accounting entirely — they're not failures.
- **Refresh/card/tap mutual exclusion**: `_refresh_emulator_safely` sets `_REFRESH_PENDING` and takes `_CARD_SOLVER_LOCK` so taps can't land on a rebooting emulator (`monitor.py:473`).
- **Display-awake holds**: both `run_all.py:30` and `monitor.py:621` assert `SetThreadExecutionState(CONTINUOUS|SYSTEM|DISPLAY)` every 30s — a sleeping display starves DXGI/dxcam of frames on every output, every run goes blind, and the supervisor hard-faults (observed live 2026-07-11 the moment the user walked away). The standalone `monitor.py supervise` path needs its own hold.
- **Blind-run exit**: 0 decisions in a run means the capture stack is broken; ai_farm exits 2 so the supervisor recycles it with a fresh dxcam init instead of burning boosted runs blind (`ai_farm.py:456-463`).
- **Per-run recording** (`AIFARM_RECORD=1`): `_RecordingWriter` (`ai_farm.py:49`) is a bounded background JPEG writer whose metadata only ever contains *completed* writes; the ring's already-encoded JPEG bytes are reused, so recording is nearly free. `frames.json`/`keys.json` are written `.tmp` + `os.replace` (atomic on the same volume) because a force-kill mid-`json.dump` left truncated manifests that crashed the next training-corpus load (`ai_farm.py:411-424`). A run whose recorder failed still counts (RESULT printed first) but ai_farm then exits 2 rather than leak the writer (`ai_farm.py:496-500`).
- **Watch-loop crash tolerance**: a transient per-frame cv2/adb error must not reach the `finally` that kills the farm — the watcher logs and continues; only real Ctrl+C/SystemExit reaches cleanup (`monitor.py:726-746`).

### Per-run metrics (the `>> RUN n OVER` line, `ai_farm.py:465-469`)

| Field | Meaning |
|---|---|
| `N hits (x/min)` | rebound-confirmed HP drops: a dip must still be depressed 0.4s later (`ai_farm.py:352`) — overlays sweeping the HP strip rebound within a frame |
| `bonus-artifact skipped` | dips latched out by the BONUSTIME banner detector (3s latch, `ai_farm.py:124-139`) — bonus washes fake HP drops (verified 2026-07-11: 10/10 sampled "hits" were artifacts) |
| `rebound-discarded` | pending dips that bounced back = overlay artifacts |
| `PITS=n` | pit falls via the "5 for 1 Pit Lift" revive-pill template (4s refractory, `ai_farm.py:146-156,342`) — falls cause no HP drop and were invisible before this |
| `contact=x/min` | `(hits + rebounds)/min` — every HP-strip contact incl. discarded ones |
| `fps` | decisions/second; feeds the rc-17 refresh check |
| `jump=/slide=` | action counts |
| `hazard-fires=` | only when a HazardTrigger wraps the agent (`ai_farm.py:244-255`): forced jumps from the async pit detector |

The following `>> RESULT:` line carries the same prefix in *both* read-ok and UNREAD branches — the supervisor counts runs by that prefix, and an unread run (Result screen pre-empted by a card/level-up/box screen) is still a completed run (`ai_farm.py:481-494`). Session ground truth is the menu wallet delta (`read_wallet`, `farm_common.py:234`), since the Result OCR is fragile: `read_run_result` (`farm_common.py:173`) additionally requires 5s on-screen settle (the coin tally animates — the 2026-07-13 batch locked mid-tally values) and returns the modal positive read on timeout, reporting `coins=0` as `read_ok=False` because a completed run never truly banks zero.

---

## 8. Policies and agents: the model zoo at decision time

At decision time the bot is a stack of interface-compatible wrappers, each exposing `decide(frame) -> ActionDecision` / `act(frame)` / `reset()` (and, for the learned family, `observe()` and an `explore` attribute). The deployed composition, assembled in `scripts/ai_farm.py:207-255`, is: `HazardTrigger( HybridPhaseAgent( LearnedAgent(base), LearnedAgent(bonus) ) )` — an imitation-learned earner for normal stages, a cleaner FiLM dodger for BONUSTIME, and an async pit detector that can force a jump over the top of both. Activation is durable via `data/demo/hybrid.json` (`{"base","bonus","confs",[...], "hazard","hazard_thr"}`; env vars `AIFARM_HYBRID` / `AIFARM_HAZARD` override; deleting the file reverts to plain `model.pt`). The three possible actions are `ACTION_NOOP/JUMP/SLIDE` (`gestures.py:5-7`); decisions carry a `reason` string that the whole stack (and ai_farm's log parser) relies on.

### LearnedAgent — behavioral-cloning CNN (`policies/learned.py`)

`LearnedAgent` (`learned.py:133`) runs a CNN trained by `scripts/train2.py` on the user's own play. Everything that must match training exactly — architecture, crop, input size, frame-stack temporal spacing — is driven by `model_meta.json` so train and inference cannot drift (`learned.py:5-7`). Torch is imported lazily; `FORCE_CUDA=1` aborts rather than fall back to CPU (`learned.py:148-152`).

**Input.** K grayscale frames, cropped per `meta["crop"]`, resized to (W,H), stacked as (1,K,H,W) (`learned.py:225-252`). Live capture can run far faster than the training recorder (dxcam ~270fps), so frames are appended to the K-ring only every `1/meta["fps"]` seconds — `meta["fps"]` is *measured* from the demos' real cadence at train time; a 15ms-span stack would be out-of-distribution (`learned.py:159-163`). Sub-gap ticks refresh only the newest slot with fresher pixels (`learned.py:246-249`).

**Architectures** (`build_net_from_meta`, `learned.py:109-130`): `small_cnn` (default; conv stack flattened *with spatial layout preserved — no global pooling, obstacle POSITION is the signal*), `small_cnn_film` (same trunk + FiLM conditioning on [t, speed, bonus]; FiLM gamma/beta zero-init so an untrained FiLM is an exact identity — `learned.py:76-106`), plus `mobilenet_v3_large` / `efficientnet_b5` with the first conv widened to K channels. `build_convs` (`learned.py:62-73`) keeps layer indexing identical across small_cnn, FilmCNN and the SSL pretrainer so a pretrained encoder state_dict loads 1:1.

**Gates and caps** (`learned.py:301-308`):

| Knob | Default | Meaning |
|---|---|---|
| `conf` | 0.6 | Min softmax prob to act at all; below → NOOP. Trades a few missed dodges for far fewer spurious ones. |
| `conf_slide` | 0.60 (`cfg.gestures.slide_conf`) | Slide-specific gate. USER CORRECTION 2026-07-06: sliding is CHEAP — a wrong slide doesn't kill; its only cost is that a held slide blocks the one-finger jump. So the gate is low ("duck readily"), not the old 0.90 that wrongly assumed slide→pit death (`learned.py:137-142`). |
| `jump_cooldown_s` | 0.30 (`cfg.gestures.jump_cooldown_s`) | TIME-based (tick-based breaks at 100+fps loops). Kept small: human demos double-jump with gaps down to ~0.12s (`learned.py:204-211`). |
| `explore` | 0.0 | Set per-run by the farm. Only fires where top prob < 0.85 — samples the action from p and COMMITS (bypassing conf), so alternatives are actually tried; confident dodges are never randomised. Explored slides below `conf_slide` are still vetoed: a wrong slide keeps the cookie low through a platform gap = un-tankable pit death, asymmetric vs a jump which just lands back (`learned.py:280-300`). |

**`observe(frame)`** (`learned.py:254-261`) updates the frame stack and cond tracker *without* inference — this is what lets a wrapper keep an idle model's temporal state warm so a hand-off never sees a stale/duplicated K-stack.

**FiLM conditioning.** For `small_cnn_film`, the agent computes the same [t, speed, bonus] vector live that train2 computed offline, checks the BONUSTIME banner itself every 0.25s, and hard-fails on missing `cond`/`speed_norm` meta (silent pinning bugs). If the checkpoint was trained with the bonus dim all-0 (`bonus_trained=False`), the live bonus dim is forced to 0 even if this machine has the banner template — the model must never see an input it never trained on (`learned.py:169-199`).

### condition.py — the [t, speed, bonus] vector and BONUSTIME detection

The rationale (`condition.py:1-19`): the plain BC model is a fixed-lead reactive classifier, but the game speeds up as a run deepens and bonus stages have different physics, so a lead correct at 60s is late at 300s (the ~55-65k-collected death band). The three scalars let the policy learn timing *as a function of game state*:

| Dim | Definition | Constants |
|---|---|---|
| `t` | run-elapsed / `t_norm_s`, clamped [0,1] | `T_NORM_S=600` |
| `speed` | phase-correlated horizontal scroll (px/s at model res) on the lower band (`_SPEED_BAND_Y0=0.45` — the sky parallaxes slower and would bias a full-frame match), EMA 0.35, / `speed_norm`, clamped [0,2] | `estimate_scroll` `condition.py:85-109` |
| `bonus` | 1.0 while the banner was seen within `bonus_latch_s` | `BONUS_LATCH_S=3.0`, `BONUS_THRESH=0.30` |

BONUSTIME detection is a TM_CCOEFF_NORMED match of a top-left banner crop (fractions of frame, resolution-independent) against machine-local, gitignored `templates/bonustime_norm.png`; missing template → bonus soft-off at 0 everywhere (train, live, gate scoring degrade identically — `condition.py:71-75, 174-187`). `bonustime_bgr` crops *before* cvtColor because it runs on the hot path (`condition.py:60-68`). The banner pulses; the 3s latch bridges the dips. `SCROLL_V=2` (`condition.py:77-109`) versions the speed estimator: v1 carried phaseCorrelate's constant +0.5px centroid offset; v2 is offset-corrected signed magnitude with reverse motion clamped to 0. The version is stamped into `meta["cond"]["scroll_v"]` at train time and live inference must use the checkpoint's version — mixing versions shifts every speed value, exactly the drift this module exists to prevent. `CondTracker` (`condition.py:112-151`) holds the live EMA/latch state; `run_speeds`/`latch_bonus`/`build_run_cond` are the offline twins.

### HybridPhaseAgent — two models, phase-routed (`policies/hybrid_phase.py`)

WHY (`hybrid_phase.py:4-9`): the 2026-07-12 A/Bs proved a stable trade — `plain_hf4` earns the most coins via jump-spam in normal stages, while `sslfilm_hf4` plays ~35% cleaner and survives the pit-heavy BONUSTIME platform gauntlets (where every model death occurs). The wrapper routes each phase to the model that wins it.

Phase detection reuses `bonustime_bgr` + the 3s latch, checked at most every `check_s=0.25`s (`hybrid_phase.py:60-65`); the latch's decay *is* the switch hysteresis — no thrashing on the banner's pulse. Every `decide()` the **passive** model gets `observe(frame)` so both K-stacks stay warm and a switch never hands control to a model with a degenerate stack; only the active model runs inference (`hybrid_phase.py:70-75`). Decisions are re-tagged `"base/..."` or `"bonus/..."` — the prefix must not contain `:` because ai_farm parses `reason.split(":")[1]` as the class name (`hybrid_phase.py:76-78`). `.explore` fans out to both members; `._device` delegates to base for the startup banner. Missing banner template prints a warning and the base model runs everything.

### HazardTrigger — async learned pit-fall override (`policies/hazard_trigger.py`)

WHY (`hazard_trigger.py:3-7`): M1.1 forensics proved the base policy is BLIND to killing pits (41/46 no-jump falls have jump-conf ~0), and M4.1 proved a small detector recovers 75% of held-out pits from the same pixels. Because the base contributes ~0 confidence, a soft gate-nudge can't lift it over the line — the trigger must fire a jump on its own. It is fully decoupled: it keeps its OWN K-ring (never touches the inner agent's buffers) and only overrides the action; with no `AIFARM_HAZARD` set nothing runs and deployed behaviour is byte-identical.

**Why sync failed / the async design.** LIVE FINDING 2026-07-14: running the head every frame dropped fps 50→37, and low fps is *itself* the dominant fall driver — the detector was net-harmful (`hazard_trigger.py:46-49, 62-65`). Two mitigations coexist: a frame throttle (`check_every=3`; a ~1.5s pit approach ≈ 50 frames, so every 3rd still gives ~17 looks) and, by default (`AIFARM_HAZARD_ASYNC=1`), a **background worker thread**: the decision loop only stores the latest frame ref and reads a cached float — both near-zero cost — while the worker preprocesses, infers, and publishes `self._p_pit` as fast as the GPU allows (`hazard_trigger.py:66-113`). `decide()` consumes each published `p_seq` at most once (`_last_p_seq`, `hazard_trigger.py:191-195`) so a cached result never counts as an extra confirmation read. The sync fallback path (`hazard_trigger.py:196-201`) keeps the every-Nth-frame throttle.

**Firing state machine** (`hazard_trigger.py:202-221`):

| Knob | Default (env) | Behaviour |
|---|---|---|
| `thr` | 0.7 (`AIFARM_HAZARD_THR`) | P(pit) ≥ thr counts as an above read. |
| `confirm_reads` | 1 (`AIFARM_HAZARD_CONFIRM`) | Hysteresis: the FIRST fire of an episode needs N consecutive above-thr reads — false-fire rate falls ~geometrically while a real ~1.5s approach barely notices. 1 = off. |
| `cooldown_s` | 0.25 | Also the DOUBLE-JUMP gap: with P staying ≥ thr, the chained 2nd jump (for wide pits) fires as soon as the cooldown clears, no re-confirmation. |
| `max_per_episode` | 2 (`AIFARM_HAZARD_MAXJUMP`) | Jump budget per sustained-P episode; 2 allows exactly the chained second tap. Three consecutive sub-thr reads end the episode and re-arm the budget. |
| `check_every` | 3 (`AIFARM_HAZARD_EVERY`) | Publish/infer every Nth frame. |

If the inner decision already jumps, the trigger stands down (`hazard_trigger.py:216`). Fired decisions are re-tagged `hazard:jump:<p>`; `self.fires` counts them for the run log. Bonustime belongs to the film dodger: the hybrid tags those frames `bonus/...` and the trigger passes them through untouched (also decaying its episode) (`hazard_trigger.py:184-189`).

**Generation-based reset safety** (`hazard_trigger.py:73, 133-147`): `reset()` bumps `_generation` under the lock and clears the ring, `_latest`, and `_p_pit`. The worker snapshots the generation with each frame and re-checks it both before appending to the ring and before publishing — a pre-reset inference that finishes late can neither poison the new run's buffer nor publish a stale probability, and the worker never infers on a previous run's frame.

The head itself (`_HazardNet`, `hazard_trigger.py:224-240`) is the shared conv trunk + Flatten→128→1 sigmoid, rebuilt to match `scripts/train_hazard.py`'s `hazard.pt`.

### The CV lineage: RuleBasedAgent and HybridAgent

`policies/rule_based.py` holds the pure-CV era and the shared `ActionDecision` dataclass (`rule_based.py:397-402`). Its calibrated detector suite — `_pit_ahead` (floor-brightness/green comparison vs the strip under the cookie; pits are instant death), the priority-ordered `_hazard` taxonomy (scissors→orange pumpkin/trunk→ice tower→hedgehog→rock spikes→inert falling-pins), and the `_bonus_active` lit-letter-prefix gate — is measurement-backed against a 143-frame corpus, with the discipline that false positives (uncontrolled air time) are worse than false negatives (one HP hit) (`rule_based.py:184-191`). `RuleBasedAgent` (`rule_based.py:415`) jumps only with a classified reason; `StreamingRuleBasedAgent` (`rule_based.py:450`) adds duplicate-frame suppression and N-frame confirmation (pits stay immediate). `HybridAgent` (`policies/hybrid.py:29`) wraps the learned model with these CV overrides only when the model is passive (NOOP), throttled to `cv_hz=30` with a 0.35s jump cooldown; the SLIDE override is deliberately *not* cooldowned so it re-asserts every tick and keeps the SlideHold finger down through the whole obstacle (`hybrid.py:16-21`). This is a different class from `HybridPhaseAgent` and is not in the deployed stack.

### The decision loop — `farm.play_until_death` (plus the entry points)

The actual per-frame loop is `farm.py:294` (`play_until_death` → `_run_loop`, `farm.py:317-463`). `agents/controller.py` is the Tkinter GUI ("CookieGame") that runs `farm()` on a daemon thread with stop-event/queue plumbing plus read-only boost/action checks (`controller.py:484-533`); `agents/play.py` is the 15-line headless wrapper around the same `farm()`.

**Cadence.** `tick_s = 1/cfg.decision_hz` (`farm.py:328`), but when the device exposes `wait_frame` the loop runs in STREAMING mode — it reacts to every decoded frame the instant it arrives (scrcpy pushes at display rate) and `tick_s` degrades to the poll-fallback/timeout (`farm.py:330-343`). Expensive full-res template checks are gated to their own 0.25s cadence: the ~80ms HUD check throttled the whole loop to ~12fps when run every frame, starving the learned policy's frame stack (`farm.py:333-338`). HUD absence suppresses inputs immediately (finger lifted) but is *not* death — BONUSTIME washouts bleach the HUD for 10-30s; the loop breaks early only on a menu/result template or after a sustained 30s absence (`farm.py:389-407`). A run-start "Head Start" prompt is tapped at its *settled* position once two consecutive matches agree within 20px, because the button animates in with an elastic bounce and taps at the moving match all failed (`farm.py:356-388`).

**Action execution.** Jumps go through `apply_action` (`gestures.py:104-116`): a tap, or a held press when `jump_hold_ms>0` (holding jumps higher). Tap points and hold durations are jittered (Gaussian position, ±frac duration) because repeating the exact pixel/millisecond hundreds of times per run is a trivial server-side bot tell (`gestures.py:11-31`).

**Slide-hold.** SLIDE is a stateful press-and-hold, not a discrete gesture (`farm.py:418-430`, `gestures.py:34`): touch-DOWN on the first slide prediction, held while the model keeps predicting (`slide_grace_s=0.30` bridges single-frame flickers), UP after grace *and* `slide_min_hold_s=0.45` (anti stutter-slide — popping up after a few frames was observed live). There is deliberately NO time cap: CookieRun allows an indefinite hold (user-confirmed), and a hard cap would blip the finger up mid-tunnel into an obstacle; a stuck prediction can't hang a run because the HUD-absent/stall/run-boundary paths all force-release. This design also killed the old input-queue backlog where per-tick `input swipe ... 500` stacked seconds of queued gestures in the adb shell (`gestures.py:39-46`, `farm.py:418-422`). One-finger discipline: a jump first releases any held slide unless `protecting()` (min-hold) forbids it (`farm.py:425-430`), and `force_release` re-sends UP unconditionally at every run boundary so a silently-rejected UP can never perma-slide the next run (`gestures.py:91-101`, `farm.py:310-314, 320`).

**Death detection.** Frame-diff stillness (6 still ticks past `min_s`) triggers up to 4 centre stall-taps (boost prompts pause the run); after that, a static screen is a real death only if the HUD is also gone — the Head Start dash renders near-identical frames and once false-died an alive boosted run at 14s (`farm.py:435-459`). If a menu/popup template is visible on a stall, the loop stops immediately rather than centre-tapping (which on the menu opens the League leaderboard/Friend popups).

---

## 9. Training pipeline and data flywheel

Every model in this repo is trained offline from recorded runs stored under `data/<run>/`, all sharing one on-disk schema: a folder of timestamped JPEG frames plus two JSON manifests. Human demos come from `scripts/recorder.py`; bot self-play comes from the farm itself via the `AIFARM_RECORD` flywheel; derived `.npy` caches (model-res frames, HP curve, pit indices) make re-training cheap. The current flagship trainer is `scripts/train_iql.py` (offline RL — the first trainer that can exceed imitation), with `scripts/train_hazard.py` (M4 pit detector) and the legacy BC trainer `scripts/train2.py` sharing the same corpus and conv trunk. `scripts/fall_forensics.py` closes the loop by diagnosing *why* falls happen from the same mined data.

### Recording format

One directory per run. Namespaces matter: `demo2+` = 35 fps human demos, `hf2..hf4` = 60 fps human demos (`hifps` mode keeps its own `hf*` namespace "so the normal `demo*` sweep never mixes fps", `recorder.py:13-16`), `demo_self_*` = self-farm runs, `botrun_*` = AIFARM_RECORD flywheel runs.

| File | Contents | Written by |
|---|---|---|
| `frames/NNNNNN.jpg` | Grayscale-later JPEGs; human: quality 88 at 1920 px (or 960 px hifps — the JPEG writer can't encode 1920 px at 60 fps, `recorder.py:13-16,43-44`); bot: the 640×360 quality-80 ring JPEGs (`ai_farm.py:326-327`) | recorder.py / `_RecordingWriter` |
| `frames.json` | `{"frames":[{"idx","t"}…], "save_w", "duration_s", "actual_fps", "hifps"}`; botruns add `"pit_times"` and `"complete"` (`recorder.py:167-169`, `ai_farm.py:415-418`) | same |
| `keys.json` | List of `{t, key, action, dur}` — one entry per real press, `dur` = hold length filled on release so training can label the whole slide/jump *span* (`recorder.py:47-73`). Botrun entries are synthesized from `decision.action` transitions (no `key` field, `ai_farm.py:334-339`) | same |
| `cache_ssl_{H}x{W}_{crop}.npy` | uint8 bank of model-res gray crops. The crop fractions are part of the filename because the length-only staleness check can't distinguish two crops at the same H×W (`pretrain_encoder.py:84-86`). Built lazily by `pretrain_encoder.py:93-106` or `train_iql.py:243-252` | first trainer to touch the run |
| `cache_pits.npy` | Raw frame indices where the "5 for 1 Pit Lift" revive prompt matches `templates/pitlift_norm.png` at ≥0.55 with a 4 s refractory (`train_iql.py:148-174`). Deliberately stores *raw indices*, not rewards, so one cache serves every `--pit-spread` value (`train_iql.py:62-65`) | train_iql |
| `cache_hp.npy` | Per-frame HP fraction from an HSV orange mask over the HP-bar ROI of the color originals (the gray caches crop the bar away, `train_iql.py:177-198`) | train_iql |

Recorder details that protect data integrity: frames dropped by the bounded write queue are *not* listed in `frames.json` ("a dropped frame has no JPEG on disk, so recording its idx would break training loads", `recorder.py:147-150`); the save loop advances on a fixed cadence grid, not gap-after-save (otherwise 60 fps target → 48 fps actual, `recorder.py:138-142`); hifps uses a 4-thread JPEG writer pool (one thread maxed at ~47 fps, `recorder.py:90-92`); auto-stop runs in a background `result_watcher` because the "ok" template only matches at native 2560 px and a 57-90 ms full-res match on the hot path would cap the loop under 60 fps (`recorder.py:106-128`) — a `<rec>/STOP` file forces a clean manual finalize. Never reuse a run folder: the startup wipe would destroy earlier training data (`recorder.py:20-21`).

All corpus loaders gate on `recording_is_complete` (`_runtime.py:15-18`): legacy demos predate the `complete` flag and are accepted; explicit `complete: false` and empty captures are rejected.

### The AIFARM_RECORD flywheel

`AIFARM_RECORD=1` makes `ai_farm.py` persist *every* farm run as `data/botrun_<MMDD_HHMMSS>/` in the demo schema (`ai_farm.py:287-297`). It is nearly free: frames are already JPEG-encoded for the 600-entry death-dump ring, so recording is just writing those bytes on a worker thread. The stated purpose: "each farm run yields on-policy trajectories + auto-mineable pit-fall labels for the next IQL iteration (falls were unmeasurable before; 25 mined examples were too few)". `_RecordingWriter` (`ai_farm.py:49-116`) is a bounded-queue writer whose manifest "contains only completed writes" — short writes raise, errors latch, and `close()` drains with a timeout. `frames.json`/`keys.json` are written atomically (`.tmp` + `os.replace`) so a force-kill mid-dump can't leave a truncated manifest that crashes the next corpus load (`ai_farm.py:411-424`); `complete` = run finished ∧ writer closed ∧ no error ∧ nonempty frames. A run whose recorder failed still exits with code 2 *after* printing its RESULT, so the supervisor counts it and can restart without leaking the writer (`ai_farm.py:496-500`).

### train_iql.py end-to-end

**Why IQL** (`train_iql.py:1-20`): every prior trainer was BC-family and all plateaued at the demonstrator — "BC cannot prefer an action the data never rewarded." IQL learns value from outcomes using the reward signal the farm already produces.

**Corpus** (`train_iql.py:107-114`): default = all `demo_self_*` + `botrun_*` runs with a `frames.json`, plus `hf2/hf3/hf4`; `--runs csv` overrides. States are the deployed champion's exact K-stack geometry (K/H/W/crop from `data/demo/model_meta.json`); actions are the executed key at each frame, spanning `[t, t+max(dur, 0.08)]` (`train_iql.py:253-259`).

**Reward construction** (`train_iql.py:56-65, 260-277`):

| Signal | Value | Where it lands |
|---|---|---|
| LIVE | +0.01 | every frame |
| HIT | −1.0 | confirmed HP hits: >6% drop vs the 1.1 s rolling max, *still* ≥0.05 depressed 0.45 s later (rebound-confirm kills bonus-wash artifacts), 0.6 s refractory, first 4 s excluded (`hit_frames`, `train_iql.py:201-218`) |
| PIT | −4.0 (`--pit-r`), spread over `--pit-spread` 1.0 s | evenly over the frames *leading into* the fall — the prompt appears ~0.5-1 s after the miss and the mistimed/missing jump ~0.5 s earlier, "so credit lands on the decision" (`train_iql.py:271-275`) |
| DEATH | −5.0 | last frame of each run ("run end = death, or the human stopped: close enough") |

The explicit pit penalty exists because the revive setup tanks up to 3 falls — a fall causes *no* HP drop and *no* terminal, so without it "the reward is blind to the exact failure the clean-run objective cares about most (why IQL-1 never learned pits)" (`train_iql.py:57-60`).

**Losses** (`train_iql.py:414-466`, Kostrikov-style, all Adam 3e-4, batch 256, seed 0, CUDA required): V by expectile regression (τ=0.7) toward the min of twin *target* Qs; twin Q by MSE toward `r + γ·bootstrap` (γ=0.995); policy = advantage-weighted BC with `exp(β·A)` (β=3.0) clamped at 100; Polyak targets at 0.995/0.005. Two verified fixes are load-bearing: the terminal mask excludes last-frames from the batch pool, and the 2026-07-14 done-mask fix bootstraps a terminal successor's *own reward* instead of the untrained `V(terminal)` — before it, the −5 death penalty never entered any target (`train_iql.py:429-435`). `--pit-oversample N` duplicates pre-fall-window transitions in the epoch index pool (with `np.unique` so overlapping windows still get exactly N copies), so critics and actor see identical oversampling (`train_iql.py:66-69, 379-386`).

**M2 selective imitation** — actor-loss only, critics untouched, all default no-op (`train_iql.py:70-82`): Turn 2 (more unfiltered bot data) plateaued "because AWBC imitates whatever the recording did, and the bot's recordings are mostly its own flaws."

| Flag | Effect |
|---|---|
| `--mask-hits S` | zero the actor weight on the S seconds before each mined hit (those actions *caused* the hit) |
| `--human-weight W` | multiply the actor weight on hf2/hf3/hf4 so the human prior isn't drowned by self-play |
| `--min-quality` | drop bot runs with >1 pit fall from the corpus entirely (human runs kept) |

**M3 value-side knobs** — both default to bit-identical-to-iql3 (`train_iql.py:83-90`): `--nstep N` precomputes n-step targets that never cross a run boundary (`m = min(N, steps-to-run-end)`; the run-final DEATH_R is captured through the bootstrap, keeping run-boundary semantics identical to 1-step, `train_iql.py:387-413`); `--cql-alpha A` adds `A·(logsumexp_a Q − Q(a_data))` to both critics, reusing the same forward pass (`train_iql.py:446-451`).

**Memory budget** (`train_iql.py:91-99, 116-137, 309-342`): the frame bank is `n·H·W` uint8 — at the deployed 96×224 geometry that is 21,504 bytes ≈ **21.5 KB per transition**, so the full 1.43 M-transition corpus is a ~30.6 GB bank, bigger than this box's 16 GB VRAM *and* its 31 GB RAM (iql5b thrashed the pagefile, then OOM'd the GPU load "and died without saving"). Three defenses:
- `--max-frames N` caps the corpus, keeping **all human demos** plus the **freshest bot runs** by mtime, and *logs every dropped run* — "a silent cap would read as 'trained on everything'".
- `cache_ssl` files are loaded `mmap_mode="r"` so `np.concatenate` copies straight out of the mmap; peak RAM becomes just the destination bank instead of ~2× (`train_iql.py:236-242`).
- Bank auto-placement: GPU-resident when `bank_bytes < 0.5 × free VRAM` (headroom for nets/activations); otherwise it stays in CPU RAM and only the gathered per-batch `(B,K,H,W)` stacks (a few MB) move to the GPU; a failed VRAM probe assumes "won't fit" (`train_iql.py:314-326, 336-342`).

**Export** (`train_iql.py:468-481`): the policy net *is* the small_cnn architecture, so its weights are remapped into the small_cnn `Sequential` layout (conv keys pass through; fc → `9.weight/bias`, head → `12.weight/bias`) and saved as `data/demo/{prefix}.pt` + `_meta.json` with `arch: "small_cnn"` and `cond` popped — deployable through `LearnedAgent` unchanged.

### train_hazard.py — the M4 pit detector

M1.1 forensics proved the dominant failure is **blindness**: "41/46 no-jump falls have the policy's jump-confidence at ~0 over the whole pre-fall window" — so gating/imitation can't fix it; a detector might (`train_hazard.py:1-9`). This script only *measures* whether the pixels carry the signal; if they separate, the head is wired as a jump trigger (`AIFARM_HAZARD` wraps the agent with `HazardTrigger`, `ai_farm.py:240-255`).

- **Model**: a binary head (`Flatten → Linear 128 → ReLU → Dropout 0.4 → Linear 1`) on the *same* conv trunk as the policy, geometry forced from `{--enc-init}_meta.json` (default `iql3`) so on-disk SSL caches line up and the trunk warm-starts 1:1 (`train_hazard.py:50-56, 123-143`).
- **Labels**: `y = 1` for frames within `--hazard-s` (default 1.5 s) *before* each `cache_pits` prompt — the approach window. Out-of-range stale-cache indices are skipped rather than crashing the run (`train_hazard.py:75-82`).
- **Split**: run-level (no frame leakage); requires ≥3 complete pit-positive runs, holds out `max(2, frac·N)` but always leaves ≥1 pit run in training (`train_hazard.py:98-104`). Pit-free runs >20 k frames are excluded so the 65 k-frame demos don't swamp negatives (`train_hazard.py:115-118`).
- **Loss**: `pos_weight`-balanced BCE over a 4:1 negative-subsampled pool; `--focal G` and `--augment` (brightness jitter + 1-in-K temporal drop) are opt-in and bit-identical no-ops when off (`train_hazard.py:196-238`).
- **Metrics**: per-epoch frame P/R at 0.5/0.7/0.9 plus AP with lift over base rate; then held-out **deployment metrics** at 0.7/0.9/0.97, because frame P/R "hides what matters live": per-pit recall (fired ≥1× inside each pit's approach window, `pit_detection_counts`) and false-fire **bursts/min** — counting only fire rising-edges that *begin* on safe ground (bug-hunt #10: the old `fire & safe` edge counted the safe-side tail of a correct detection as a fresh alarm) at each run's *measured* fps (bug-hunt #9: hardcoded 50 fps biased the rate) (`train_hazard.py:269-299`).

Output: `data/demo/{out}.pt` + meta with `arch: "hazard"`, hazard_s, enc_init, focal/augment provenance (`train_hazard.py:301-305`).

### train2.py / SSL / FiLM lineage (brief)

`train2.py` is the BC-family trainer that produced the champion line: class-weighted CE over key-press windows (`win_pre`/`win_post`), "not-yet" negative windows (×4 weight), slide **span** labeling with `--slide-span-cap` (the full 3 s hold labels every crouched frame as slide → live SLIDE-LOCK, `train2.py:197-200`), AWR survival weighting with a human floor so a tanking self-run can never outweigh the expert anchors (`train2.py:477-480`), DAgger corrections mixed in at ×5 weight, optional negatives (`--neg-npz` unlikelihood/jump), `--label-shift-ms`, `--meta-from` arch inheritance for self-farm retrains, FiLM `cond` (t/speed/bonus) calibration, measured-fps write-back into meta, `--save-best` on the sweep's `hit_rate − fam/400` score, and W&B logging. `pretrain_encoder.py` is the SSL leg: predict the frame ~100 ms ahead from a cutout-masked K-stack ("predicting the future forces the encoder to represent scroll speed, obstacle positions and object permanence"), on the exact small_cnn trunk so `--encoder-init` loads 1:1; it also builds the shared `cache_ssl` files and calibrates the scroll-speed med/p90 that FiLM's `speed_norm` uses. `correct.py` is the DAgger labeler over `data/ai_hits/` hit clips (W/S/N/X keys, ~1 s per hit); it snapshots the *exact in-memory frames shown* into `ai_hits/corrections/` so a parallel farm overwriting the source file can never make a stored label point at different pixels, and its k-frames are mtime-coherence-gated because run/hit numbers reset per session (`correct.py:74-90, 154-167`). `src/cookierun_bot/reward.py` (`RewardTracker`) is *not* part of this pipeline — it serves only `env.py`'s planned-RL scaffold (ponytail note at `reward.py:1-2`).

### fall_forensics.py — routing the roadmap

M1.1's no-emulator, no-train diagnostic over every run with a `cache_pits.npy` (`fall_forensics.py:51-63`). Its windows are "deliberately generous and reported as distributions so the thresholds are auditable, not load-bearing" (`fall_forensics.py:32-38`):

| Output | Mechanics | Feeds |
|---|---|---|
| Executed-key classes | Per fall, keys in the 3 s pre-prompt window: `no-jump` (gating/blind), `late-jump` (<0.35 s lead), `fired-in-time` (needs double jump / wrong spot), `wrong-action` (slide ≤1.2 s before — pits need a jump) (`classify_fall`, `fall_forensics.py:74-89`) | M4 vs M1.2 routing |
| Fall-time histogram | 15 s buckets of seconds-into-run; HOT buckets (≥1.5× uniform, min 3) are merged into a suggested `AIFARM_GATE_SCHEDULE="a-b:0.35,…"` string (`fall_forensics.py:129-147`) | M1.2 time-windowed gate schedule |
| Post-revive clustering | Fraction of non-first falls within 8 s of the previous fall; >30% → "add post-revive caution window" (`fall_forensics.py:150-159`) | caution-window decision |
| `--replay` (Part B) | Re-runs the deployed base (default iql3) over each no-jump fall's pre-fall SSL-cache frames and buckets max jump-confidence: GATED (0.30 ≤ conf < 0.45 live gate → a lower gate would have jumped), BLIND (<0.30 → "model never saw the pit (needs data/hazard-head, not gating)"), NEAR (≥ gate → execution/timing gap) (`fall_forensics.py:174-225`) | decided M4 (hazard head) over more gating |

It also writes a machine-readable `fall_forensics.json` at the repo root for downstream steps (`fall_forensics.py:165-171`).

---

## 10. Support tools, tests, and repo layout

Around the live farm loop sits a set of offline support scripts (`scripts/`), a phone-side capture/input app (`android-bridge/`), and a hardware-free test suite (`tests/`). All scripts share a tiny path shim, `scripts/_runtime.py`, which resolves `ROOT`/`DATA`/`CONFIG` relative to the script's own location, puts `src/` on `sys.path`, and exposes `recording_is_complete()` (legacy human demos predate the `complete` flag, so a non-`False` flag plus nonempty frames is the compatibility gate, `_runtime.py:15-18`). Large machine-local assets (`data/`, `templates/`) are gitignored and shared into worktrees via NTFS junctions (see last subsection).

### Auxiliary scripts (`scripts/`)

**`sweep.py` — hyperparameter sweep for the imitation model.** Trains a grid of configs on the recorded human demos and deploys the winner to `data/demo/model.pt` (+`model_meta.json`, `sweep_results.json`). Each demo run is decoded once at the largest swept resolution (crop band `[0.10, 0.20, 1.00, 0.90]`, cached as `cache_{H}x{W}.npy`), then the entire frame bank is moved into VRAM (`sweep.py:93`) — other resolutions are GPU-resized once and every batch is a pure GPU gather, nothing in system RAM. Two hard-won details are encoded in comments: (1) frame-stack spacing is expressed through `meta['fps'] = REC_FPS/stride`, and `REC_FPS` is *measured* from the demos' real timestamps (`sweep.py:104-107`) so `LearnedAgent` stacks live frames at exactly the span it trained on (no OOD drift); (2) checkpointing is best-epoch, because "the last epoch is often not the best". Scoring on the held-out last-15% tail is `score = hits/events − false_fires_per_min/400`, best over conf ∈ {0.5…0.9} — the high end exists because at 60fps the false-fire penalty scales with fps (`sweep.py:197-204`). Config sets are selected by argv: default `CONFIGS_R1`, `r2`/`r3` refinement rounds, `hr` (240×560 max-res), `hifps` (60fps `hf*` demos, output isolated in `data/_hifps_model` so the deployed 35fps model is untouched). The `CONFIGS_HIFPS`/`CONFIGS_R3` comment blocks (`sweep.py:295-329`) are a lab notebook: augmentation is the decisive lever with a single demo; smaller `win_pre` = tighter labels = fewer and *later* fires; final ranking is always live survival. `sweep.py <mode> shard i n` trains `CONFIGS[i::n]` for GPU-parallel sharding (each shard saves `data/_shard_i.pt`; `sweep_par.py` merges + deploys). The deploy guard never overwrites a better deployed model — but only when the previous score came from the *same* demo set; if demos changed, scores aren't comparable and this round's winner deploys (`sweep.py:383-408`). Live per-epoch progress goes to `sweep_progress.jsonl` (tailed by `sweep_dash.py`); `--wandb` logs one W&B run per config. Directories named `*test*` or `*self*` are excluded — self-farm runs belong to `self_farm.py` only (`sweep.py:55-58`).

**`model_score.py` — the shared held-out dodge-quality scorer.** The single measuring stick for the self-farm promotion gate and offline experiments: scores a checkpoint exactly the way `sweep.py` picks its winner — last-15% val tails of the human demos (`demo2/3/4`, "the stable expert ground truth"), canonical `score = hits/events − fam/400`, best over conf {0.5, 0.6, 0.7}. Evaluation is deterministic (argmax + softmax, no sampling), so re-scoring the champion always returns the identical number — the gate can't flap on noise. The pure helpers (`extract_events`, `event_score`, `best_conf_score`, `gate_accepts`) import only numpy so tests exercise them without a GPU; `score_model` lazily imports torch/cv2. Only the val-tail frames a K-stack references are decoded, cached per (demo, geometry) (`model_score.py:88-98`) — demo4 alone is 65k frames, and decoding it fully on every retrain would stall the farm. FiLM checkpoints get their `[t, speed, bonus]` cond vector built with the same helpers `train2.py` used, with hard errors on missing `cond`/`speed_norm` (a wrong scale would score garbage). A missing eval set raises `RuntimeError`, *not* `SystemExit`, deliberately: `self_farm._gate_score`'s `except Exception` catches it and the gate fails closed (`model_score.py:223-225`).

**`learned_check.py` — live single-run smoke test of the deployed model.** Navigates to a real run through the *full* boost gate (3 tiles + Double Coins Multi-Buy — standing user rule: never start a run un-boosted, `learned_check.py:41-44`), drives `LearnedAgent` through `play_until_death`, and logs each HP-drop hit, the action mix, and effective decision fps. Hit counting has a 0.6s refractory because at 70fps one collision's multi-frame HP drain re-triggered the drop check — "110 hits/min was an artifact, not more collisions" (`learned_check.py:61-63`). Deploy conf comes from `sweep_results.json`, overridable as `argv[1]`. Compare against the `dodge_check.py` rule-based baseline (35 hit-events/120s).

**`eval_deploy.py` — offline operating-point sweep, no retraining.** Simulates deployment over the held-out val tail: confidence (0.6–0.95) × N-frame persistence (1–3) × a 0.8s jump-cooldown sim (~28 frames @35fps), printing event hit-rate and false fires/min for each cell. Used to pick the live conf/persistence before burning an emulator run. Note it reads a single-run `data/demo/frames.json` layout directly.

**`analyze_hits.py` — hit-diagnostics triage.** Categorizes `data/ai_hits/hits.jsonl` by the model's decision trace in the pre-impact window [−1.0s, −0.2s]: `blind` (max action prob < 0.30 — data gap), `hesitant` (0.30–0.60 — threshold issue), `fired-but-hit` (an action fired yet HP dropped — timing/wrong action), `cooldown-blocked`, `conf-gate-missed`. The docstring states the method: each category implies a different fix — count them before changing anything. Also reports what fired and the median lead time before impact for `fired-but-hit`.

**`self_farm.py` — the self-improving farm.** Runs the full farm flow (boost gate + Head Start + card solver via `monitor.py` + wedge recovery), banking real coins while recording every run and periodically retraining on its own longest-survival runs. The design guards that matter:

- *Anchoring*: every retrain includes human anchor demos (default `--anchors hf2`) so a bad self-play batch can't drift the model into garbage (`self_farm.py:82-84`).
- *Promotion gate*: a retrain deploys **only** if it beats the deployed champion on the held-out human demos via `model_score`; scoring failure keeps the champion (fail-closed). The comment records why: the old blind hot-swap let survival noise drift the model sideways — adversarially verified flat, never improving (`self_farm.py:92-98`, `deploy_retrain` at `:349-373`, atomic temp-file + `os.replace` swap so the farm never reads a half-written checkpoint).
- *Greedy-only promotion*: explore runs survive by inference luck, not policy quality, so only greedy runs are promoted to `demo_self_*` (`spawn_retrain`, `self_farm.py:379-384`).
- *fps consistency*: recording fps is matched to the deployed model's `meta['fps']`, and the JPEG writer becomes a 4-thread pool at ≥50fps because one writer maxes ~47fps — dropped frames would make recorded fps < model fps, i.e. out-of-distribution training data (`self_farm.py:251-253`).
- *Recording integrity*: a run whose writer failed, stopped dirty, or captured nothing is marked incomplete, excluded from training, and a writer failure stops the farm outright (`self_farm.py:561-572`).
- *Unattended hardening* (header `self_farm.py:22-27`): per-run exception guard, `finally` teardown, escalating recovery (ensure_running failures → game restart → capture-device reopen; repeated degenerate runs → game restart, because the game can crash to the launcher while ensure_running still "succeeds"), hung-retrain kill, launcher-focus self-heal (`_game_foreground`), and calibration-gated capture open (`_calibrated_device`: a game-area < ~1000px means LDPlayer isn't rendering full-size and the model is effectively blind). It refuses to start at all if `monitor.py` can't launch ("refusing to share the emulator"). Clean stop: `touch data/_selffarm/STOP`; on exit it waits up to 600s for an in-flight retrain and gate-deploys it.

| Safety knob (`self_farm.py:101-112`) | Default | Purpose |
|---|---|---|
| `MAX_CRASHES` | 6 | consecutive per-run failures before aborting the night |
| `ENSURE_FAILS_ESCALATE` | 3 | ensure_running failures before a game restart (tier 2 reopens capture) |
| `DEGEN_ESCALATE` | 3 | consecutive degenerate runs before restarting the game app |
| `RETRAIN_TIMEOUT_S` | 1800 | kill a hung background retrain (frees GPU, resumes swaps) |
| `MONITOR_MAX_RESTARTS` | 20 | cap on card-solver relaunches |
| `MIN_DUR_S` / `MIN_FRAMES` | 5.0 / 100 | drop false-start / blind-capture runs from training |
| `RECENT_CAP` | every+keep+4 | bound tmp run dirs so disk can't fill during a long retrain |
| `SHUTDOWN_RETRAIN_WAIT_S` | 600 | grace for the in-flight retrain on clean exit |

### `android-bridge/` — token-authed phone bridge

A small Gradle/Kotlin Android app (`com.cookierun.bridge`: `MainActivity`, `CaptureService`, `TapAccessibilityService`, `BridgeServer`) that lets the PC brain drive a real phone over Wi-Fi. `BridgeServer.kt` is a line-based TCP server with a five-verb protocol (`BridgeServer.kt:12-21`): `AUTH token` must be the first request (compared with constant-time `MessageDigest.isEqual`, `BridgeServer.kt:73`; wrong token = disconnect), then `FRAME` (4-byte big-endian length + JPEG), `TAP x y`, `HOLD x y ms`, `PING`. The PC client (`NetworkDevice` in `src/cookierun_bot/device.py`) refuses to connect without a token — from config or the `COOKIERUN_BRIDGE_TOKEN` env var (`device.py:523-529`; "the bridge token is session-only, so an environment variable is enough"), and rejects whitespace in it. The built `CRBridge-debug.apk` and `.gradle/` output are gitignored. Client behavior is covered by `tests/test_network_device.py`.

### Test suite (`tests/`)

Flat directory, 30 files, **257 tests collected** — all hardware-free. `conftest.py` provides `FakeDevice`, an in-memory `Device` stand-in that records `taps`/`holds` and serves a settable frame, plus `fake_device`/`blank_frame` fixtures (`conftest.py:5-33`). `test_runtime_regressions.py` additionally uses an ast-based `_load_defs` helper that compiles only selected function/class defs from legacy scripts so their hardware-oriented import-time setup never runs (`test_runtime_regressions.py:20-31`, marked `ponytail:`).

The highest-value guards — the ones that stop the bot from spending currency, corrupting training data, or wedging overnight:

| Guard | Tests | What it proves |
|---|---|---|
| Menu confirm-guard | `test_menu.py`, `test_farm.py:319-385` | spend dialogs are never tapped (`revive_crystals` blocked even when absent from the denylist); only allowlisted buttons fire; `ensure_running` blocks otherwise-safe buttons while a denylisted dialog is visible |
| Boost gate | `test_boost_watch.py`, `test_farm.py:571-905` | `ready_to_play` requires all three tiles checked **and** the Double Coins banner; `ensure_running` refuses to Play without the required boosts, attempts the double-coin buy once per cycle, books boost cost only when the gate is ready, and enforces the per-run tile cap |
| Card solver | `test_runtime_regressions.py:45-157` | the farm-side card handler never taps (even with legacy `AIFARM_CARD_AUTO=1`); capture failures leave card protection armed; the device lock refuses a second monitor owner per ADB serial; emulator refresh waits for the solver to release ownership; modal taps stand down during refresh |
| Hazard trigger | `test_hazard_trigger.py` | the async pit-hazard confirmation counts only *distinct* inference results (`_p_seq`), so a stale probability re-read can't satisfy the N-confirm requirement |
| Recording integrity | `test_runtime_regressions.py:282-453` | the recorder never deadlocks nor lists frames whose write failed; metadata is frozen after close-timeout; explicitly-incomplete recordings are rejected by IQL training and by self-farm; pit labels require explicit recorded pit evidence |

The rest of the suite covers the module map: capture/device/input (`test_capture.py`, `test_device.py`, `test_win_input.py`, `test_network_device.py`), template detection and gestures, policies (`test_rule_based.py`, `test_learned_arch.py`, `test_condition.py`/`test_film.py`, `test_hybrid.py`/`test_hybrid_phase.py`), scoring and training metrics (`test_model_score.py`, `test_reward.py`, `test_metrics.py`, `test_train_hazard_metrics.py`), farm agents (`test_action_watch.py`, `test_coin_watch.py`, `test_overlay_watch.py`), gift drawing (stops on a card bonus without tapping, `test_gift_draw.py:257`), and config/env/sandbox plumbing.

### `data/`, `templates/`, and worktree junctions

Both directories are machine-local and gitignored (`.gitignore:12` `templates/`, `.gitignore:15` `data/`): `data/` holds recordings, models, diagnostics and self-farm state; `templates/` holds the screen-match PNGs, which are capture-setup-specific (HDR-off, LDPlayer window geometry). Because Claude worktrees live under `.claude/worktrees/<name>/`, each worktree carries NTFS junctions (reparse tag `0xa0000003`, mount point) `data -> C:/Users/singh/Desktop/cookierun-bot/data` and `templates -> C:/Users/singh/Desktop/cookierun-bot/templates`, so every branch's code sees the *single* shared model/demo/template store — `scripts/_runtime.py` resolves `DATA = ROOT / "data"` relative to the worktree and lands in the shared store through the junction. Practical consequences: a model deployed by a sweep in one worktree is immediately live for a farm launched from another, and a fresh worktree needs its junctions recreated before any script touching `DATA` or `templates/` will run.

---

## 11. Reference: environment variables and config.yaml

### Environment variables

| Variable | Default | Where read | Effect |
|---|---|---|---|
| `AIFARM_JUMP_CAP` | `0.60` | `scripts/ai_farm.py:183` | Overrides the jump-confidence gate cap for live A/Bs (the agent's gate is `min(conf, cap)`); lower caps let film models fire pit-jumps they predict at 0.61-0.75. |
| `AIFARM_SLIDE_CONF` | unset (falls back to config `gestures.slide_conf` / LearnedAgent default) | `scripts/ai_farm.py:187` (applied at `:204`) | Overrides the slide-confidence gate for A/Bs; printed as an OVERRIDE when set. |
| `AIFARM_FPS_MIN` | `45` | `scripts/ai_farm.py:197` | If measured decision fps stays below this for 2 consecutive runs, ai_farm exits with code 17 (`REFRESH_EXIT`) so monitor.py does an ldconsole emulator refresh; `0` disables the check (monitor.py:532 relaunches with `AIFARM_FPS_MIN=0` after the refresh cap is hit). |
| `AIFARM_HYBRID` | unset (falls back to `data/demo/hybrid.json` if present) | `scripts/ai_farm.py:212` | `"base,bonus"` model-name pair that activates the phase-aware `HybridPhaseAgent` (base earns in normal stages, bonus dodges BONUSTIME pit gauntlets). |
| `AIFARM_HYBRID_CONFS` | `""` → both gates default to `min(conf, jump_cap)` (setdefault'd from hybrid.json `confs` at `:218`) | `scripts/ai_farm.py:232` | Per-model jump gates for the hybrid as `"0.60,0.45"`; must be exactly two numbers in (0,1] or ai_farm exits (`:43-45`). |
| `AIFARM_HAZARD` | unset (setdefault'd from hybrid.json `hazard` key at `:224`) | `scripts/ai_farm.py:244` | Model name or `.pt` path (`"hazard"` → `data/demo/hazard.pt`) that wraps the agent in `HazardTrigger`, the learned pit detector that forces a jump when P(pit) crosses the threshold outside BONUSTIME. |
| `AIFARM_HAZARD_THR` | `0.7` | `scripts/ai_farm.py:250` (setdefault from hybrid.json `hazard_thr` at `:226`) | P(pit) probability threshold at which the hazard trigger forces a jump. |
| `AIFARM_HAZARD_EVERY` | `3` | `scripts/ai_farm.py:251` | `check_every`: run hazard-detector inference only every N decision frames. |
| `AIFARM_HAZARD_CONFIRM` | `1` | `scripts/ai_farm.py:252` | `confirm_reads`: consecutive above-threshold reads required before the forced jump fires. |
| `AIFARM_HAZARD_MAXJUMP` | `2` | `scripts/ai_farm.py:253` | Max forced jumps per sustained hazard episode (caps repeated firing on one obstacle). |
| `AIFARM_HAZARD_ASYNC` | `1` (on) | `src/cookierun_bot/policies/hazard_trigger.py:66` | `1` runs hazard inference on a background thread (fps-safe; the sync path dropped fps 50→37 and was net-harmful); anything else = synchronous per-frame forward. |
| `AIFARM_RECORD` | unset (off) | `scripts/ai_farm.py:293` | `1` persists every farm run as training data (`data/botrun_*` in the demo recording schema) — the IQL data flywheel; consumed by `scripts/train_iql.py:111`. |
| `MONITOR_CARD_THRESH` | `0.85` | `scripts/monitor.py:58` | Template-match threshold for detecting the card mini-game screen (arms the autonomous card solver). |
| `MONITOR_CARD_THRESH_ACT` | `0.80` | `scripts/monitor.py:60` | Lower per-round recheck threshold used while actively solving the card game. |
| `MONITOR_MARGIN_OK` | `3.0` | `scripts/monitor.py:62` | Score margin at/above which a card tap counts as confident; below = low-confidence handling. |
| `COOKIERUN_BRIDGE_TOKEN` | none — **required** for the `network` capture backend (RuntimeError if empty) | `src/cookierun_bot/device.py:524` | Auth token for the on-device phone bridge (`AUTH <token>` handshake); fallback when no token is passed in code, must contain no whitespace. |
| `ADBUTILS_ADB_PATH` | unset (falls back to `shutil.which("adb")` then adbutils' bundled adb) | read `src/cookierun_bot/agents/controller.py:118` (`discover_adb_path`); saved/set/restored around farm runs at `src/cookierun_bot/farm.py:588-681` and `controller.py:403-432`, `:448-480` from config `device.adb_path` | Path to the adb binary honored by the adbutils library; farm/controller export it (from `cfg.adb_path`) for the duration of a run and restore the old value afterward. |
| `FORCE_CUDA` (alias `CUDA_ONLY`) | `0` (off) | `src/cookierun_bot/policies/learned.py:149` | `1`/`true`/`yes` requires a CUDA device — raises RuntimeError instead of silently falling back to CPU inference (the 7fps CPU-torch trap). |
| `AIFARM_GATE_SCHEDULE` | n/a — **not read anywhere yet** | only emitted as a suggestion by `scripts/fall_forensics.py:147` | Planned M1 knob (time-windowed jump-gate schedule, e.g. `"150-165:0.35,225-315:0.35"`); fall_forensics prints a suggested value but no code consumes it. |
| `AIFARM_CARD_AUTO` | n/a — **no reader in current code** | set only by `tests/test_runtime_regressions.py:66` | Vestigial: the test sets it before calling `farm_cards._cardgame`, but nothing in `scripts/` or `src/` reads it anymore. |

### config.yaml reference

Parsed by `src/cookierun_bot/config.py::load_config` (defaults shown are the loader's; `config.example.yaml` values noted where they differ). Sections `loop`, `gestures`, `reward`, `menu`, and all five `regions` are **required** (ConfigError if missing).

| Key | Type / default | Effect |
|---|---|---|
| `device.serial` | str or null (default null) | ADB device serial; null = auto/first device (`127.0.0.1:5555` fallback in monitor.py). |
| `device.capture` | str, default `scrcpy`; one of `scrcpy`, `adb`, `ldplayer`, `bluestacks`, `network` (example: `ldplayer`) | Screen-capture backend selection. |
| `device.max_fps` | int, default 60 (example: 120) | Capture frame-rate cap. |
| `device.adb_path` | str, default `""` | Explicit adb binary path; exported as `ADBUTILS_ADB_PATH` during farm runs. |
| `window.title` | str, default `"BlueStacks App Player"` (example: `"LDPlayer"`) | Emulator window title for Windows-native capture/input. |
| `window.top_bar` | int, default 40 | Pixels of emulator window chrome to exclude at the top. |
| `window.right_bar` | int, default 40 | Pixels of emulator window chrome (toolbar) to exclude at the right. |
| `loop.decision_hz` | int, **required** (example: 60) | Agent decision-loop frequency. |
| `loop.target_stage` | str, **required** (example: `"Episode 1"`) | Stage the bot farms. |
| `regions.play_area` | `[x,y,w,h]`, **required** | Gameplay crop on the captured frame (model input). |
| `regions.coin_counter` | `[x,y,w,h]`, **required** | In-run coin counter OCR region. |
| `regions.mystery_box_counter` | `[x,y,w,h]`, **required** | In-run mystery-box counter OCR region. |
| `regions.results_coins` | `[x,y,w,h]`, **required** | Result-screen coins OCR region. |
| `regions.results_ingredients` | `[x,y,w,h]`, **required** | Result-screen ingredients OCR region. |
| `gestures.jump_button` | `[x,y]`, **required** | Jump button tap coordinate. |
| `gestures.slide_button` | `[x,y]`, **required** | Slide button tap coordinate. |
| `gestures.slide_hold_ms` | int, **required** (example: 500) | Base slide hold duration. |
| `gestures.jump_hold_ms` | int, default 250 | Jump press hold duration (held press = higher/longer jump). |
| `gestures.tap_jitter_px` | int, default 0 (example: 55) | Gaussian tap-scatter radius around each button (anti-detection humanization; 0 = deterministic). |
| `gestures.hold_jitter_frac` | float, default 0.0 (example: 0.15) | +/- fractional jitter on hold durations (anti-detection). |
| `gestures.slide_grace_s` | float, default 0.40 | Slide stays held this long after the model stops predicting slide (proven R3 value). |
| `gestures.slide_min_hold_s` | float, default 1.5 | Minimum total hold once a slide starts; a jump does not cut it short inside this window. |
| `gestures.jump_cooldown_s` | float, default 0.25 | Minimum time between jumps. |
| `gestures.slide_conf` | float, default 0.35 | Min softmax probability for the model to slide (overridable via `AIFARM_SLIDE_CONF`). |
| `reward.w_coin` | float, **required** (example: 1.0) | Reward weight per coin. |
| `reward.w_box` | float, **required** (example: 50.0) | Reward weight per mystery box. |
| `reward.w_survive` | float, **required** (example: 0.01) | Per-step survival reward weight. |
| `reward.death_penalty` | float, **required** (example: 10.0) | Penalty on death. |
| `spending.allow_coin_boosts` | bool, default false | Master switch for spending coins on pre-run boosts (Double Coins etc.). |
| `spending.max_boost_cost_per_run` | int, default 0 (example: 12000) | Per-run coin-spend budget for boosts. |
| `spending.double_coins_first_cost` | int, default 1200 | Expected cost of the first Double Coins buy. |
| `spending.double_coins_reroll_cost` | int, default 600 | Expected cost of each Double Coins reroll. |
| `spending.max_double_coin_rolls` | int, default 3 | Max Double Coins reroll attempts per run. |
| `spending.forbid_crystals` | bool, default true | Never spend crystals (hard safety rail). |
| `menu.allowlist` | list[str], **required** (example: `[restart, replay, collect, ok, start]`) | Menu-template names the bot may tap. |
| `menu.denylist` | list[str], **required** (example: `[revive_crystals, buy, purchase, watch_ad]`) | Menu-template names the bot must never tap. |
| `phone.host` | str, default `""`; **required when `device.capture: network`** | On-device bridge app host (phone IP). |
| `phone.port` | int, default 8080 | On-device bridge app port. |
| `templates_dir` | str, default `templates` | Directory of template images for menu/screen matching. |

---

## Document map

| Doc | Purpose |
|---|---|
| [README.md](../README.md) | Fresh-machine setup, install, first run |
| [RUNBOOK.md](../RUNBOOK.md) | Record / train / farm command reference |
| [docs/MILESTONES.md](MILESTONES.md) | The M0–M5 no-human improvement roadmap + 20-idea backlog |
| **docs/PROJECT.md** (this file) | Architecture, subsystems, ops catalog, full reference |

*Generated 2026-07-20 from a fresh read of the working tree (multi-agent documentation pass)
merged with the project's operational history. When code and this document disagree, trust the
code — and fix the document.*
