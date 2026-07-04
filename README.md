# CookieRun Bot

Local CookieRun Classic farming controller for an Android emulator.

## Current Local Setup

- ADB auto-selects the single ready emulator if the configured serial is stale.
- Capture size is `2560x1440`.
- `config.yaml` is set to `capture: ldplayer`, `max_fps: 120`, `decision_hz: 60`, and coin boosts enabled.
- Button, result, digit, and Double Coins boost templates are present under `templates/`.
- Live proof on 2026-07-03: the settled Result-screen coin crop read matched the
  screenshot at `108,452` coins.

## Run Path

Install Python deps:

```powershell
.\install.ps1
```

`install.ps1` creates/uses `.venv`, installs `requirements.txt`, installs this project
editably, then installs `scrcpy-client` with `--no-deps` because its published dependency
pin conflicts with the `adbutils` 2.x API used by this bot.

The commands below assume the venv Python. Use `.\.venv\Scripts\python.exe -m ...` if
you have not activated `.venv`.

Read-only calibration screenshot:

```powershell
python -m cookierun_bot.calibrate config.yaml
```

Read-only coin watcher:

```powershell
python -m cookierun_bot.agents.coin_watch config.yaml --frames 10 --interval 1 --stable-reads 2
```

Read-only jump/slide advisor:

```powershell
python -m cookierun_bot.agents.action_watch config.yaml --frames 120 --hz 60
```

Read-only pre-run boost gate check:

```powershell
python -m cookierun_bot.agents.boost_watch config.yaml --frames 10
```

Click-through read-only overlay for manual play:

```powershell
python -m cookierun_bot.agents.overlay_watch config.yaml --interval-ms 100 --hold-ms 220
```

Use `--hold-ms` to compensate for human reaction/capture delay. During particle-heavy
skill/channeling frames the overlay shows `SKILL` and suppresses jump/slide prompts;
tune that with `--channeling-threshold` or disable it with `--no-channeling`.

Offline sandbox that actually executes the jump/slide policy:

```powershell
python -m cookierun_bot.sandbox
```

Start the bot only when you actually want it to press controls:

```powershell
python -m cookierun_bot.agents.play config.yaml
```

One-click controller UI:

```powershell
python -m cookierun_bot.agents.controller
```

From Explorer you can also launch:

```powershell
.\CookieGame.bat
```

`CookieGame.bat` prefers `.venv\Scripts\python.exe`, so the one-click app uses the same
dependencies installed by `install.ps1`.

The controller wraps the same `farm.py` engine, shows run totals, gross/net coins,
ingredients/hr, ADB path/device/run-count controls, device/boost/action checks, and live
logs. Stop uses the farm loop's stop event, so it exits cleanly without killing Python.

## Boost Spending

The target farming flow verifies the three required run boosts before every round:
HP potion, pocket watch, and x2 Point Booster. With coin boosts enabled, it also buys
Random Boost through Multi-Buy with `Double Coins` selected. The bot will not press
Play from the boost screen until the three run boosts are checked and the red
`Double Coins` banner is verified active.

This is off unless explicitly enabled:

```yaml
spending:
  allow_coin_boosts: true
  max_boost_cost_per_run: 12000
  forbid_crystals: true
```

Net coins are reported as result-screen coins minus assumed boost spend.

Result coins are read after the result screen value is stable; the bot does not use the
early animated counter value.
