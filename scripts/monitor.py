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
       * on a LOW-confidence board it DOES NOT TAP; it keeps the farm paused and waits for
         manual resolution. Wrong card taps cost more than pausing.
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
CARD_THRESH = float(os.environ.get("MONITOR_CARD_THRESH", "0.85"))
# per-round recheck threshold while solving
CARD_THRESH_ACT = float(os.environ.get("MONITOR_CARD_THRESH_ACT", "0.80"))
# >= this = confident tap; below = low-confidence
MARGIN_OK = float(os.environ.get("MONITOR_MARGIN_OK", "3.0"))
GRAB_FAILS_BEFORE_RECONNECT = 3
MAX_SUP_RELAUNCH = 3        # bounded so a truly-broken emulator can't loop forever

# --- emulator refresh (fps-degradation recovery; see refresh_emulator) ---
REFRESH_RC = 17             # ai_farm/supervisor "emulator degraded" exit code
MAX_EMU_REFRESH = 2         # refreshes per batch: a machine still slow after 2 fresh
                            # boots has a different problem — stop cycling the emulator
LDCONSOLE = r"C:\LDPlayer\LDPlayer14\ldconsole.exe"
LD_INDEX = "0"              # `ldconsole list2` -> `0,LDPlayer,...`
GAME_ACTIVITY = "com.devsisters.crg/com.devsisters.CookieRunForKakao.OvenbreakX"
BOOT_TIMEOUT_S = 240        # sys.boot_completed poll cap (~60-90s typical). NEVER use
                            # `adb wait-for-device` on the 5555 transport — it can hang.
LD_WINDOW_POS = (3520, 60, 1600, 930)   # known-good capture geometry (fix_window.py)
NEWS_X_TAP = (2255, 148)    # News-popup X (adb coords). With NO popup up this exact
                            # spot is the friends-list heart — so the tap is GATED, see
                            # _dismiss_startup_popups (never blind-tap it).

# While we're solving a card, drop a flag so farm.py's nav (which sometimes fails to match the
# card template on its dxcam frame) never sends BACK on the card screen — a stray BACK can forfeit
# the card and walk the app out to the Android launcher (observed 2026-07-05, 'sliding card' wedge).
_CARD_FLAG = str(DATA / "_selffarm" / "card_active")
def _set_card_flag() -> None:
    try:
        os.makedirs(os.path.dirname(_CARD_FLAG), exist_ok=True)
        with open(_CARD_FLAG, "w") as fh:
            fh.write("1")                              # content unused; the FRESH mtime is the signal
    except OSError:
        pass
def _clear_card_flag() -> None:
    try:
        os.remove(_CARD_FLAG)
    except OSError:
        pass

# supervise-mode shared state (updated by the supervisor pump thread, read by the watch loop)
_sup = {"target": 0, "done": 0, "finished": False, "proc": None}
_STOP = threading.Event()   # set on monitor shutdown so the pump won't relaunch mid-exit


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _alert_user() -> None:
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass


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


def median_grab(n: int = 5, gap: float = 0.10):
    """Grab n frames and return the pixel-wise MEDIAN — DE-ANIMATES the card sprites (the shimmer/
    sparkles are transient, the pose is static), giving the pose heuristic a clean, noise-free frame.
    The documented root cause of low-confidence rounds is 'sprites are animated'; this removes it."""
    fr = []
    for _ in range(n):
        g = grab()
        if g is not None:
            fr.append(g)
        time.sleep(gap)
    if not fr:
        return None
    if len(fr) == 1:
        return fr[0]
    return np.median(np.stack(fr), axis=0).astype(np.uint8)


def tap(x, y) -> None:
    try:
        _adb("shell", "input", "tap", str(int(x)), str(int(y)), timeout=10)
    except Exception:
        pass


def save_card(frame, rnd) -> None:
    """Audit every card board (FULL 2560x1440 res) so the solver can be tuned/learned offline on
    real data — half-res JPGs blur the subtle pose difference that separates the answer pair."""
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"cardgame_mon_{int(time.time())}_{rnd}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    except Exception:
        pass


def _wait_for_manual_card_clear(matcher: TemplateMatcher) -> None:
    """Hold the farm on a low-confidence card board until the user clears it."""
    last_ping = 0.0
    while True:
        _set_card_flag()
        f = grab()
        if f is None:
            time.sleep(1.0)
            continue
        if not matcher.present(f, "cardgame", CARD_THRESH_ACT):
            _clear_card_flag()
            log("low-confidence card cleared manually -- run resuming")
            return
        now = time.monotonic()
        if now - last_ping > 20.0:
            last_ping = now
            _alert_user()
            log("LOW-confidence card still up -- waiting; no card taps will be made")
        time.sleep(1.0)


def solve_cardgame(matcher: TemplateMatcher) -> None:
    """Solve rounds until the cardgame template is gone (or a safety cap is hit).

    High-confidence boards are tapped once per round. Low-confidence boards are fail-closed:
    keep the farm paused and wait for manual resolution instead of gambling wrong cards.
    """
    for rnd in range(1, 7):                       # a normal game is <=3 rounds; cap generously
        f = grab()
        if f is None:
            time.sleep(1.0)
            continue
        if not matcher.present(f, "cardgame", CARD_THRESH_ACT):
            log("card game gone -- solved/closed; run resuming")
            _clear_card_flag()                     # card gone -> let nav resume normal BACK handling
            return
        # A real card game is a full-screen overlay with NO menu Play button. If Play IS visible we
        # mis-detected the menu / Friends leaderboard as a card -> NEVER restart or tap: restarting
        # force-stops the game and spawns unrecognized transition screens that nav BACK-spams out to
        # the launcher (the observed 2026-07-05 "keeps exiting the game" wedge). Stand down instead.
        if matcher.present(f, "play", 0.72):
            log("menu Play visible -- NOT a real card game (mis-detect); standing down, no restart/tap")
            _clear_card_flag()
            return
        _set_card_flag()                           # card present -> nav must NOT BACK this screen
        f = median_grab()                          # de-animated frame (sparkles removed) for solving
        if f is None:
            continue
        save_card(f, rnd)
        i, j, margin = _card_pair(f)
        if margin < MARGIN_OK:
            log(f"round {rnd}: margin {margin:.1f} < {MARGIN_OK} = low confidence -- STANDING DOWN "
                f"(heuristic guess cards {i + 1} & {j + 1}); waiting for manual solve")
            _wait_for_manual_card_clear(matcher)
            return
        ci, cj = _CARD_CENTERS[i], _CARD_CENTERS[j]
        log(f"round {rnd}: pair = cards {i + 1} & {j + 1} (margin {margin:.1f} OK) "
            f"-> tap {ci} then {cj}")
        tap(*ci)
        time.sleep(0.6)
        tap(*cj)
        time.sleep(4.0)                            # user rule: don't hurry, wait ~4s / round
    _clear_card_flag()                             # gave up -> don't freeze nav on a stuck flag
    log("card solve: hit round cap -- leaving it; the run continues regardless")


def dismiss_modal(matcher, template: str, confirm_xy, label: str) -> None:
    """Tap the known Confirm on a BENIGN blocking popup that nav's dxcam template misses (e.g. the
    weekly 'League Results' screen, which nav BACK-spammed into a wedge 2026-07-05). This is
    BANNER-GATED — we only tap `confirm_xy` while `template` (a distinctive, spend-free banner) is
    confirmed present, so we can NEVER tap a purchase/crystal-spend dialog. Drops card_active while
    dismissing so nav stands down instead of BACK-spamming."""
    log(f"{label} popup detected -- dismissing via Confirm {confirm_xy} (banner-gated, no spend)")
    _set_card_flag()
    try:
        for _ in range(4):                         # a couple taps in case the first misses / re-shows
            tap(*confirm_xy)
            time.sleep(2.0)
            g = grab()
            if g is None or not matcher.present(g, template, 0.85):
                log(f"{label} dismissed -- run resuming")
                return
        log(f"{label} still present after 4 taps -- leaving it")
    finally:
        _clear_card_flag()


def _reposition_ld_window(emit) -> bool:
    """Move the LDPlayer window to the known-good capture spot and front it (inlined from
    the manual fix_window.py routine). A fresh LDPlayer boots at a drifted position/size;
    dxcam captures the DESKTOP window, so geometry must be restored BEFORE the relaunched
    ai_farm recalibrates its game-area. Also jiggles the mouse first — a sleeping display
    starves dxcam of frames entirely."""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value == "LDPlayer":     # EnumWindows by exact title (FindWindow flaked)
                    found.append(hwnd)
            return True

        user32.EnumWindows(_cb, 0)
        if not found:
            emit("[refresh] no visible 'LDPlayer' window found to reposition")
            return False
        user32.mouse_event(1, 1, 0, 0, 0)       # wake the display
        user32.mouse_event(1, -1, 0, 0, 0)
        user32.MoveWindow(found[0], *LD_WINDOW_POS, True)
        user32.SetForegroundWindow(found[0])
        time.sleep(0.5)
        emit(f"[refresh] LDPlayer window -> {LD_WINDOW_POS} + fronted")
        return True
    except Exception as exc:
        emit(f"[refresh] window reposition failed: {exc!r}")
        return False


def _find_close_multiscale(frame, matcher):
    """Find an event/News-popup X on an ADB frame. The close/close2 templates were cropped
    from nav's dxcam frames (~0.5-0.7x the 2560px adb width), so the matcher's single-scale
    match misses them on adb grabs (the league_results lesson) — search a few downscales of
    the frame and map the best hit back to adb coords. Returns (score, x, y) or None."""
    best = None
    for name in ("close", "close2"):
        tpl = matcher._templates.get(name)
        if tpl is None:
            continue
        for s in (0.44, 0.5, 0.56, 0.62, 0.7, 0.8, 1.0):
            small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
                continue
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv >= 0.80 and (best is None or mv > best[0]):
                best = (float(mv), int((ml[0] + tpl.shape[1] / 2) / s),
                        int((ml[1] + tpl.shape[0] / 2) / s))
    return best


def _dismiss_startup_popups(matcher, emit) -> bool:
    """Conditionally clear the post-launch News/event popup — NEVER blind-tap: with no
    popup up, the News-X spot (2255,148) is the friends-list heart button. Ladder per
    grab: menu Play visible = clean, done; a card = leave it (the watch loop solves
    cards); League Results = the existing banner-gated dismiss; a matched close-X = tap
    the MATCH; only when the screen has settled unrecognized (attempt >= 3, Play absent)
    fall back to ONE fixed-point News-X tap. Anything still left is fine — the relaunched
    ai_farm's ensure_running clears reward-popup gauntlets as part of normal nav."""
    tapped_fixed = False
    for attempt in range(6):
        f = grab()
        if f is None:
            time.sleep(2.0)
            continue
        if matcher.present(f, "play", 0.72):
            emit("[refresh] menu Play visible — popup layer clear")
            return True
        if matcher.present(f, "cardgame", CARD_THRESH):
            emit("[refresh] card game up — leaving it to the watch loop")
            return True
        if matcher.present(f, "league_results", 0.85):
            dismiss_modal(matcher, "league_results", (1280, 1210), "LEAGUE RESULTS")
            continue
        hit = _find_close_multiscale(f, matcher)
        if hit is not None:
            emit(f"[refresh] popup X matched at ({hit[1]},{hit[2]}) score {hit[0]:.2f} — tapping it")
            tap(hit[1], hit[2])
        elif attempt >= 3 and not tapped_fixed:
            tapped_fixed = True
            save_card(f, f"refresh_{attempt}")     # audit frame: what did we tap on?
            emit(f"[refresh] settled unrecognized + Play absent — single gated News-X tap at {NEWS_X_TAP}")
            tap(*NEWS_X_TAP)
        time.sleep(2.5)
    emit("[refresh] not on a clean menu after popup pass — leaving the rest to ai_farm's nav")
    return False


def refresh_emulator(emit) -> bool:
    """Full unattended LDPlayer refresh — the proven RAM-overflow recovery procedure
    (2026-07-06) + tonight's manual fps fix, automated: ldconsole quit -> launch -> poll
    sys.boot_completed via `adb connect` + getprop (NEVER `adb wait-for-device`: it hangs
    on the 5555 transport) -> `am start` the game (it does NOT auto-start; boot focus is
    the LDPlayer launcher) -> reposition + front the window -> conditionally dismiss the
    News popup. Uses ONLY ldconsole/adb/win32 — never dxcam: the relaunched ai_farm
    re-initializes capture in its fresh process, and that is what resumes play. Returns
    False if the emulator never reached boot_completed (caller falls back to the existing
    bounded failure machinery)."""
    emit(f"[refresh] EMULATOR REFRESH: ldconsole quit/launch --index {LD_INDEX}")
    try:
        subprocess.run([LDCONSOLE, "quit", "--index", LD_INDEX], capture_output=True, timeout=30)
    except Exception as exc:
        emit(f"[refresh] ldconsole quit failed: {exc!r}")
    time.sleep(6.0)                                # Ld9BoxHeadless exits ~3s after quit
    try:
        subprocess.run([LDCONSOLE, "launch", "--index", LD_INDEX], capture_output=True, timeout=30)
    except Exception as exc:
        emit(f"[refresh] ldconsole launch failed: {exc!r}")
        return False
    t0 = time.time()
    booted = False
    while time.time() - t0 < BOOT_TIMEOUT_S:
        try:
            subprocess.run(["adb", "connect", SERIAL], capture_output=True, timeout=15)
            out = _adb("shell", "getprop", "sys.boot_completed", timeout=10)
            if out.stdout and out.stdout.strip() == b"1":
                booted = True
                break
        except Exception:
            pass
        time.sleep(5.0)
    if not booted:
        emit(f"[refresh] emulator did NOT boot within {BOOT_TIMEOUT_S}s — refresh FAILED")
        return False
    emit(f"[refresh] booted in {time.time() - t0:.0f}s — starting the game")
    try:
        _adb("shell", "am", "start", "-n", GAME_ACTIVITY, timeout=20)
    except Exception as exc:
        emit(f"[refresh] am start failed: {exc!r}")
        return False
    _reposition_ld_window(emit)
    emit("[refresh] waiting 40s for the game to reach the title/menu")
    time.sleep(40.0)
    _dismiss_startup_popups(TemplateMatcher(TEMPLATES), emit)
    _reposition_ld_window(emit)                    # re-assert geometry+focus right before relaunch
    emit("[refresh] refresh complete — relaunching the farm")
    return True


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
    refreshes = 0            # emulator refreshes used this batch (capped MAX_EMU_REFRESH)
    fps_check_off = False    # after the cap: relaunch with AIFARM_FPS_MIN=0 so the batch
                             # finishes degraded-but-whole instead of churning 2-run chunks
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
            env = dict(os.environ, AIFARM_FPS_MIN="0") if fps_check_off else None
            try:
                p = subprocess.Popen([sys.executable, "-u", SUP_SCRIPT, str(remaining)],
                                     cwd=str(ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env=env)
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
            if rc == REFRESH_RC:
                # fps-degradation handoff from ai_farm (via supervisor). A refresh
                # relaunch is NOT a failure: skip the no-progress accounting entirely
                # (the >= 2 runs that tripped the check were already counted above).
                if refreshes < MAX_EMU_REFRESH:
                    refreshes += 1
                    emit(f"[mon-sup] FPS-DEGRADED exit (rc={rc}) -> emulator refresh "
                         f"{refreshes}/{MAX_EMU_REFRESH}")
                    if not refresh_emulator(emit):
                        emit("[mon-sup] refresh FAILED (emulator never booted) — "
                             "relaunching anyway; a dead emulator yields zero-run "
                             "attempts and the existing bounded budgets stop the batch")
                    emit(f"[mon-sup] resuming batch: {target - _sup['done']} run(s) left")
                else:
                    fps_check_off = True
                    emit(f"[mon-sup] refresh budget ({MAX_EMU_REFRESH}) exhausted — "
                         "relaunching with AIFARM_FPS_MIN=0 (finish the batch degraded "
                         "rather than cycling the emulator)")
                continue
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
    if margin < MARGIN_OK:
        log(f"would STAND DOWN: margin {margin:.2f} < {MARGIN_OK}")
    else:
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
                if matcher.present(f, "cardgame", CARD_THRESH) and not matcher.present(f, "play", 0.72):
                    _set_card_flag()                   # card seen -> veto nav BACK NOW, before the
                                                       # 2-poll settle gate (closes the cold-start race
                                                       # where nav could BACK a just-appeared card).
                                                       # Play-veto: never arm on a menu/leaderboard frame.
                    seen += 1
                    if seen >= 2:                      # settle gate: 2 consecutive detections
                        log("CARD GAME detected (settled) -- taking over")
                        solve_cardgame(matcher)
                        seen = 0
                        hb = time.monotonic()
                elif matcher.present(f, "league_results", 0.85):
                    dismiss_modal(matcher, "league_results", (1280, 1210), "LEAGUE RESULTS")
                    seen = 0
                    hb = time.monotonic()
                else:
                    seen = 0
                    _clear_card_flag()                 # no card in view -> release the BACK veto
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
