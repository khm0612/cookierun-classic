"""One-command wrapper that runs the WHOLE farm stack together and tees a combined log.

Starts a single owning process (`monitor.py supervise N`) that concurrently runs:
  * the MODEL            — LearnedAgent dodger, via the supervisor -> ai_farm child
  * the boost pipeline   — Double-Coins gate + the 3 boost tiles + Head Start every run
  * the CARD-GAME SOLVER — auto-solves the post-run "Find the card" bonus (no human needed)
  * ADB auto-recovery    — reconnects the device on capture drops
  * SUPERVISOR relaunch  — restarts the farm if it dies (bounded; never two farms at once)

Everything streams LIVE to the console AND to a timestamped logs/run_*.log, so "the model,
the log, the monitor, and the card solver" are all running — and visible — at the same time.

Usage:
  python scripts/run_all.py            # 15 runs (default)
  python scripts/run_all.py 30         # 30 runs
Stop with Ctrl+C — the stack cleans up after itself (any orphaned farm is killed on exit).
To run it unattended/detached instead, launch this same command with PowerShell Start-Process.
"""
from __future__ import annotations
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MONITOR = str(ROOT / "scripts" / "monitor.py")


def main(n: int = 15) -> int:
    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    logpath = logdir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    banner = [
        "=" * 72,
        f"  CookieRun auto-farm - unified run  ({n} runs)",
        "  model (LearnedAgent) + Double-Coins/boost gate + Head Start",
        "  + card-game solver + adb auto-recovery + supervisor relaunch",
        f"  combined log -> {logpath}",
        "  Ctrl+C to stop (any orphaned farm is cleaned up automatically)",
        "=" * 72,
    ]

    with open(logpath, "w", encoding="utf-8", buffering=1) as logf:
        def out(line: str) -> None:
            print(line, flush=True)
            try:
                logf.write(line + "\n")
            except Exception:
                pass

        for b in banner:
            out(b)

        try:
            p = subprocess.Popen(
                [sys.executable, "-u", MONITOR, "supervise", str(n)],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except Exception as exc:
            out(f"[run_all] could not start the stack: {exc}")
            return 1

        rc = 0
        try:
            for line in p.stdout:              # live tee: console + file
                out(line.rstrip())
            rc = p.wait()
        except KeyboardInterrupt:
            # Ctrl+C reached the whole console group, so the monitor is already running its
            # own cleanup; just wait for it to finish, then hard-kill if it hangs.
            out("\n[run_all] Ctrl+C - letting the stack clean up (killing any orphaned farm)...")
            try:
                p.wait(timeout=30)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            rc = 130

        out(f"[run_all] finished (rc={rc}); full log at {logpath}")
        return rc


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 15))
