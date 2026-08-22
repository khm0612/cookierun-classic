"""farm.py — ONE command that runs the whole bot. Everything, unattended.

    python scripts/farm.py              # 15 runs
    python scripts/farm.py 30           # 30 runs
    python scripts/farm.py 10 --boot    # cold-boot the emulator first (closed/stuck game)
    python scripts/farm.py 5  --attended    # YOU solve card games (no auto-solver)
    python scripts/farm.py 10 --no-record   # don't write training recordings

What this owns, so you never have to remember another command:

  preflight   adb reachable (auto `adb connect`), emulator alive, optional cold boot,
              GPU-contention warning (external games starve the hazard detector = more falls)
  models      the deployed stack straight from data/demo/hybrid.json — printed before launch
  per run     Double Coins Multi-Buy + the 3 boost tiles + Head Start, then the learned dodger
  between     card-game solver, reward-popup dismissers (Mystery Box / Level Up / Congrats),
              adb reconnect, emulator refresh on fps decay, farm relaunch if it dies
  output      live console + a timestamped logs/farm_*.log of the entire session

WHY this still spawns children instead of being one flat process: the farm is deliberately
supervised — monitor watches supervisor watches ai_farm — so a crashed/wedged run is killed
and relaunched without losing the batch, and the card solver keeps working while the farm is
dead. Collapsing that into a single process would trade an overnight-proof stack for a single
point of failure. This file is the one entry point; the supervision below it stays.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from _runtime import DATA, ROOT      # also puts src/ on sys.path — never import-order dependent

MONITOR = str(ROOT / "scripts" / "monitor.py")
SUPERVISOR = str(ROOT / "scripts" / "supervisor.py")
HYBRID = DATA / "demo" / "hybrid.json"
SERIAL = os.environ.get("COOKIERUN_SERIAL", "127.0.0.1:5555")

# Games/apps that measurably cost the bot falls: they hold VRAM and starve the async hazard
# detector's worker (live 2026-07-20: hazard fires collapsed 15 -> 1-6 and falls spiked while
# fps sat at 36-39 instead of ~48). Warned about, never killed — the desktop is the user's.
GPU_HOGS = ("StarRail", "League of Legends", "VALORANT", "csgo", "cs2", "obs64", "obs32")


def _adb(*args, timeout=20):
    return subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout)


def _device_ready() -> bool:
    try:
        out = _adb("devices").stdout
    except Exception:
        return False
    return any(ln.split("\t")[-1].strip() == "device"
               for ln in out.splitlines()[1:] if ln.strip())


def preflight(force_boot: bool) -> bool:
    """Make the emulator reachable, or say precisely why we can't farm."""
    print("=" * 68)
    try:
        cfg = open(HYBRID, encoding="utf-8").read().strip() if HYBRID.exists() else "(none)"
    except Exception as exc:
        cfg = f"(unreadable: {exc})"
    print(f"  deployed stack : {cfg}")

    busy = []
    try:
        tl = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=20).stdout
        busy = [g for g in GPU_HOGS if g.lower() in tl.lower()]
    except Exception:
        pass
    if busy:
        print(f"  !! GPU contention: {', '.join(busy)} running — expect lower fps and MORE "
              f"falls.\n     Closing them is the single cheapest cleanliness win.")

    if not force_boot and _device_ready():
        print(f"  emulator       : ready ({SERIAL})")
        print("=" * 68, flush=True)
        return True

    # not ready (or a boot was demanded): try a cheap reconnect before the 2-minute cold boot
    if not force_boot:
        print("  emulator       : not ready — reconnecting adb...")
        try:
            _adb("kill-server"); time.sleep(1)
            _adb("start-server"); time.sleep(1)
            _adb("connect", SERIAL); time.sleep(2)
        except Exception:
            pass
        if _device_ready():
            print(f"  emulator       : ready after reconnect ({SERIAL})")
            print("=" * 68, flush=True)
            return True

    print("  emulator       : cold-booting LDPlayer (ldconsole quit/launch, ~1-2 min)...")
    print("=" * 68, flush=True)
    import monitor                     # reuse the proven boot path (boot -> game -> reposition)
    if monitor.refresh_emulator(print):
        return True
    print("!! emulator did not come up — start LDPlayer manually, then re-run.", flush=True)
    return False


def main(argv) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    runs = int(args[0]) if args and args[0].isdigit() else 15
    attended = "--attended" in flags

    # Production defaults, overridable from the environment:
    #   record  -> every run becomes training data (the flywheel that produced iql5b)
    #   fps_min -> two sub-45fps runs trigger an automatic emulator refresh (0 = off)
    os.environ.setdefault("AIFARM_RECORD", "0" if "--no-record" in flags else "1")
    os.environ.setdefault("AIFARM_FPS_MIN", "45")

    if not preflight("--boot" in flags):
        return 1

    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    logpath = logdir / f"farm_{time.strftime('%Y%m%d_%H%M%S')}.log"
    mode = "ATTENDED (you solve card games)" if attended else "UNATTENDED (auto card solver)"
    cmd = ([sys.executable, "-u", SUPERVISOR, str(runs)] if attended
           else [sys.executable, "-u", MONITOR, "supervise", str(runs)])

    with open(logpath, "w", encoding="utf-8", buffering=1) as logf:
        def out(line: str) -> None:
            print(line, flush=True)
            try:
                logf.write(line + "\n")
            except Exception:
                pass

        out(f">> FARM {runs} run(s) | {mode}")
        out(f">> record={os.environ['AIFARM_RECORD']} fps_min={os.environ['AIFARM_FPS_MIN']}"
            f" | log -> {logpath}")
        out(">> Ctrl+C stops the whole stack cleanly\n")

        try:
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as exc:
            out(f"!! could not start the stack: {exc}")
            return 1
        try:
            for line in p.stdout:                      # live tee: console + file
                out(line.rstrip())
            rc = p.wait()
        except KeyboardInterrupt:
            # Ctrl+C already reached the whole console group; let the stack run its own
            # cleanup (it kills any orphaned farm), then hard-kill only if it hangs.
            out("\n>> Ctrl+C — letting the stack clean up...")
            try:
                p.wait(timeout=30)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            rc = 130
        out(f"\n>> finished (rc={rc}) | full log: {logpath}")
        return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
