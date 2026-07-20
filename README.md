# CookieRun Bot

Local CookieRun Classic farming controller for an Android emulator.

> **Full project reference:** [docs/PROJECT.md](docs/PROJECT.md) — architecture, all
> subsystems, model lineage with measured results, the operations/failure catalog, and the
> complete env-var + config reference. Commands live in [RUNBOOK.md](RUNBOOK.md); the
> improvement roadmap in [docs/MILESTONES.md](docs/MILESTONES.md).

## Expected Local Setup

- A blank ADB serial auto-selects the first ready emulator. An explicit serial fails closed
  when that device is missing; it is never replaced with a different connected device.
- LDPlayer capture and local templates are calibrated for `2560x1440`.
- The recommended LDPlayer settings are `capture: ldplayer`, `max_fps: 120`, and
  `decision_hz: 60`. Coin spending remains opt-in.

`config.yaml`, `templates/`, and `data/` are machine-local and ignored by Git. A clean clone
does **not** include calibrated templates, recordings, or a trained model. Before farming,
copy your local `config.yaml` and `templates/` into the repo. Learned-agent runs also need
`data/demo/model.pt` plus `data/demo/model_meta.json`; restore those files or record and train
them with the runbook.

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

For the optional Android Wi-Fi bridge, copy the token displayed by the bridge app before
starting the PC client:

```powershell
$env:COOKIERUN_BRIDGE_TOKEN = "token-shown-on-phone"
```

The token is required for every connection. Treat it as a session password and use the
bridge only on a trusted local network.

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
