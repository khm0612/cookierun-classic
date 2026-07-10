"""Autonomous unattended monitor for the farm marathon.

Runs ALONGSIDE (or, in `supervise` mode, OWNS) the farm. It is fully independent of the
bot's GPU capture: it grabs the screen with `adb exec-out screencap` (NOT dxcam), so it
never contends with the running bot. Three jobs:

  1. CARD GAMES (critical) — the bot deliberately stands down and waits for a human at the
     post-run "Find the card" bonus, which freezes an unattended batch. This monitor solves
     it: shape-based pair detection (the SAME verified `farm_cards._card_pair` heuristic the
     bot uses) + taps. Sensitive path (earlier in-loop auto-tapping spam was "very
     dangerous"), so it is heavily gated:
       * acts only when `cardgame` is present on TWO consecutive polls (settle gate),
       * re-checks `cardgame` present on the SAME frame it acts on, every round,
       * taps each pair-card exactly ONCE/round, waits ~4s, re-checks (no spamming),
       * taps ONLY the 6 known card centers, ONLY while cardgame is present,
       * on a LOW-confidence board it still taps its best guess — a wrong pick just wins a
         lesser prize and 3 tries/round absorb it; stalling the marathon for hours is worse.
     Every board it sees is saved to data/ai_hits/ for offline heuristic tuning.
  2. ADB RECOVERY — repeated screencap failures trigger `adb reconnect` (device-offline has
     stalled unattended runs before).
  3. SUPERVISOR RESILIENCE (supervise mode) — owns the supervisor process and relaunches it
     (bounded) with the remaining-run count if it ever dies, so an overnight batch finishes.

Usage:
  python scripts/monitor.py                 # watch a separately-launched supervisor (cards + adb)
  python scripts/monitor.py supervise 15    # OWN the whole batch: launch+watch supervisor 15,
                                            #   solve cards, recover adb, relaunch on death
  python scripts/monitor.py test PATH       # dry-run: report detection + pair on a saved frame
"""
from __future__ import annotations
import sys
import os
import time
import subprocess
import threading

import numpy as np
import cv2

from _runtime import CONFIG, DATA, ROOT
from cookierun_bot.config import load_config                    # noqa: E402
from cookierun_bot.detect import TemplateMatcher                # noqa: E402
from cookierun_bot.farm_cards import _card_pair, _CARD_CENTERS  # noqa: E402

try:
    SERIAL = load_config(str(CONFIG)).device_serial or "127.0.0.1:5555"
except Exception:
    SERIAL = "127.0.0.1:5555"
TEMPLATES = str(ROOT / "templates")
OUT_DIR = str(DATA / "ai_hits")
SUP_LOG = str(ROOT / "supervisor.log")
SUP_SCRIPT = str(ROOT / "scripts" / "supervisor.py")
POLL_S = 3.0
CARD_THRESH = 0.85          # detection (arming) threshold -- high, to avoid false positives
CARD_THRESH_ACT = 0.80      # per-round recheck threshold while solving
MARGIN_OK = 3.0             # >= this = confident; below = flagged as a guess (still tapped)
GRAB_FAILS_BEFORE_RECONNECT = 3
MAX_SUP_RELAUNCH = 3        # bounded so a truly-broken emulator can't loop forever

# supervise-mode shared state (updated by the supervisor pump thread, read by the watch loop)
_sup = {"target": 0, "done": 0, "finished": False, "proc": None}
_STOP = threading.Event()   # set on monitor shutdown so the pump won't relaunch mid-exit


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _adb(*args, timeout=15):
    return subprocess.run(["adb", "-s", SERIAL, *args], capture_output=True, timeout=timeout)


def reconnect_adb() -> None:
    log("adb: repeated grab failures -> reconnecting")
    try:
        subprocess.run(["adb", "reconnect"], capture_output=True, timeout=15)
        time.sleep(1.0)
        subprocess.run(["adb", "connect", SERIAL], capture_output=True, timeout=15)
    except Exception as exc:
        log(f"adb reconnect error: {exc}")


def grab():
    """adb screencap -> BGR ndarray (2560x1440), or None on failure (retries once)."""
    for _ in range(2):
        try:
            raw = _adb("exec-out", "screencap", "-p").stdout
            if raw:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception:
            pass
        time.sleep(0.5)
    return None


def tap(x, y) -> None:
    try:
        _adb("shell", "input", "tap", str(int(x)), str(int(y)), timeout=10)
    except Exception:
        pass


def save_card(frame, rnd) -> None:
    """Audit every card board so the aspect heuristic can be tuned offline on real data."""
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"cardgame_mon_{int(time.time())}_{rnd}.jpg")
        cv2.imwrite(path, cv2.resize(frame, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass


def solve_cardgame(matcher: TemplateMatcher) -> None:
    """Solve rounds until the cardgame template is gone (or a safety cap is hit)."""
    for rnd in range(1, 7):                       # a normal game is <=3 rounds; cap generously
        f = grab()
        if f is None:
            time.sleep(1.0)
            continue
        if not matcher.present(f, "cardgame", CARD_THRESH_ACT):
            log("card game gone -- solved/closed; run resuming")
            return
        save_card(f, rnd)
        i, j, margin = _card_pair(f)
        ci, cj = _CARD_CENTERS[i], _CARD_CENTERS[j]
        tag = "OK" if margin >= MARGIN_OK else "LOW-guess"
        log(f"round {rnd}: pair = cards {i + 1} & {j + 1} (margin {margin:.1f} {tag}) "
            f"-> tap {ci} then {cj}")
        tap(*ci)
        time.sleep(0.6)
        tap(*cj)
        time.sleep(4.0)                            # user rule: don't hurry, wait ~4s / round
    log("card solve: hit round cap -- leaving it; the run continues regardless")


def _kill_stray_farm() -> None:
    """Kill any orphaned supervisor.py / ai_farm.py python processes before a (re)launch.
    Windows does not reap a dead parent's children, so if the supervisor ever dies with its
    ai_farm child still alive (or a prior monitor was killed / double-started), relaunching
    would put TWO farms on one emulator over one adb+capture -- the catastrophic double-tap
    case. We clean up at every launch boundary so that can never happen. (monitor.py is not
    matched by the pattern, so this never kills the watcher itself.)"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'ai_farm\\.py|supervisor\\.py' } | "
             "ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } "
             "catch {} }"],
            capture_output=True, timeout=20)
    except Exception as exc:
        log(f"stray-farm cleanup skipped: {exc}")


def _pump_supervisor(target: int) -> None:
    """Own the supervisor as a child; relaunch it (bounded) with the remaining count if it
    dies before TARGET runs complete. Completed runs are counted from the '>> RESULT:' lines
    it emits (an unread run still emits one, so it still counts). Mirrors output to SUP_LOG.
    The relaunch budget counts CONSECUTIVE deaths WITHOUT progress (it resets whenever runs
    complete), so a slow-but-advancing batch is never abandoned prematurely."""
    _sup["target"] = target
    no_progress = 0
    try:
        logf = open(SUP_LOG, "a", buffering=1)
    except Exception:
        logf = None

    def emit(line: str) -> None:
        print(line, flush=True)
        if logf is not None:
            try:
                logf.write(line + "\n")
            except Exception:
                pass

    try:
        while _sup["done"] < target and not _STOP.is_set():
            _kill_stray_farm()                        # never allow two farms on one emulator
            remaining = target - _sup["done"]
            done_before = _sup["done"]
            emit(f"[mon-sup] launching supervisor for {remaining} run(s) "
                 f"({_sup['done']}/{target} done, no-progress relaunch {no_progress}/{MAX_SUP_RELAUNCH})")
            try:
                p = subprocess.Popen([sys.executable, "-u", SUP_SCRIPT, str(remaining)],
                                     cwd=str(ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            except Exception as exc:
                emit(f"[mon-sup] could not launch supervisor: {exc}")
                break
            _sup["proc"] = p
            for line in p.stdout:
                line = line.rstrip()
                emit(line)
                if line.startswith(">> RESULT:"):
                    _sup["done"] += 1
            rc = p.wait()
            _sup["proc"] = None
            if _sup["done"] >= target or _STOP.is_set():
                break
            no_progress = 0 if _sup["done"] > done_before else no_progress + 1
            if no_progress > MAX_SUP_RELAUNCH:
                emit(f"[mon-sup] supervisor made no progress across {no_progress} relaunch(es) "
                     f"(rc={rc}), {_sup['done']}/{target} done -- giving up. Check the emulator.")
                break
            emit(f"[mon-sup] supervisor exited rc={rc}; {target - _sup['done']} run(s) left "
                 "-- relaunching in 5s")
            _STOP.wait(5)
        emit(f"[mon-sup] DONE supervising: {_sup['done']}/{target} run(s)")
    finally:
        # ALWAYS mark finished, even if the pump crashes (e.g. a stdout UnicodeEncodeError in
        # emit): main()'s only supervise-mode exit is _sup['finished'], so without this a dead
        # pump would leave the monitor heartbeating forever against no running farm.
        _sup["finished"] = True
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass


def test(path: str) -> None:
    f = cv2.imread(path)
    if f is None:
        log(f"could not read {path}")
        return
    matcher = TemplateMatcher(TEMPLATES)
    present = matcher.present(f, "cardgame", CARD_THRESH)
    log(f"frame {f.shape} | cardgame present(>= {CARD_THRESH}) = {present}")
    i, j, margin = _card_pair(f)
    log(f"_card_pair -> cards {i + 1} & {j + 1}  (0-based {i},{j})  margin={margin:.2f}")
    log(f"would tap: {_CARD_CENTERS[i]} then {_CARD_CENTERS[j]}")


def main(supervise_target: "int | None" = None) -> None:
    matcher = TemplateMatcher(TEMPLATES)
    if not matcher.has("cardgame"):
        log("FATAL: no `cardgame` template loaded -- cannot detect the card game.")
        sys.exit(1)
    if supervise_target:
        log(f"monitor: SUPERVISING {supervise_target} runs + card/adb watch | serial={SERIAL}")
        threading.Thread(target=_pump_supervisor, args=(supervise_target,),
                         daemon=True).start()
    else:
        log(f"monitor armed | serial={SERIAL} | poll={POLL_S}s | cardgame template OK")
    seen = 0
    grab_fails = 0
    hb = time.monotonic()
    try:
        while True:
            try:
                if supervise_target and _sup["finished"]:
                    log("supervisor finished -- monitor exiting")
                    return
                f = grab()
                if f is None:
                    grab_fails += 1
                    if grab_fails >= GRAB_FAILS_BEFORE_RECONNECT:
                        reconnect_adb()
                        grab_fails = 0
                    time.sleep(POLL_S)
                    continue
                grab_fails = 0
                if matcher.present(f, "cardgame", CARD_THRESH):
                    seen += 1
                    if seen >= 2:                      # settle gate: 2 consecutive detections
                        log("CARD GAME detected (settled) -- taking over")
                        solve_cardgame(matcher)
                        seen = 0
                        hb = time.monotonic()
                else:
                    seen = 0
                if time.monotonic() - hb > 120:
                    hb = time.monotonic()
                    extra = f" | farm {_sup['done']}/{_sup['target']}" if supervise_target else ""
                    log(f"heartbeat: watching, no card game{extra}")
                time.sleep(POLL_S)
            except Exception as exc:
                # a transient per-frame blip (cv2/adb/decode) must NOT fall through to the
                # finally below, which stops the supervisor and kills the running farm. The
                # watcher has to be at least as crash-tolerant as what it watches — log and
                # keep going. (KeyboardInterrupt/SystemExit aren't Exception, so a real
                # Ctrl+C / shutdown still reaches the finally and cleans up.)
                log(f"watch-loop error (continuing): {type(exc).__name__}: {exc}")
                time.sleep(POLL_S)
    finally:
        # on ANY exit in supervise mode (finished / Ctrl+C / error), stop the pump from
        # relaunching and don't leave an orphaned farm running unattended.
        if supervise_target:
            _STOP.set()
            p = _sup.get("proc")
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
            _kill_stray_farm()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        test(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "supervise":
        main(supervise_target=int(sys.argv[2]))
    else:
        main()
