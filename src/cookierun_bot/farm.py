"""Unattended farm loop for CookieRun Classic on LDPlayer (guest coords 2560x1440).

Flow (Episode 1): menu Play -> pre-run boost screen Play -> run -> rule-based agent
plays until death (screen stops scrolling) -> Result screen -> OK -> back to menu -> repeat.

Currency guardrail: after death we WAIT out any 'revive with crystals' prompt (it has a
countdown and auto-dismisses) before tapping OK, and we never tap buy/revive spots.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import os
import sys
import time

import numpy as np

from .config import CAPTURE_BACKENDS, ConfigError, load_config
from .detect import TemplateMatcher, read_int, read_results
from .device import open_device, select_adb_serial
from .gift_draw import GIFT_TEMPLATES, draw_gifts, gift_button_visible
from .gestures import ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE, SlideHold, apply_action
from .metrics import Metrics, RunResult
from .policies.rule_based import ActionDecision, StreamingRuleBasedAgent

from .farm_common import (
    _BUTTONS, _RESULT_BUTTON, _SAFE_ACTIONS,
    _diff, _find_stable, _in_run, _nav_read, _safe_to_back,
    _scrolling, _sleep_interruptible, _sleep_remaining, _snapshot,
    _stop_requested, _tap_template, _tile_checked, _tile_checked_stable,
    _visible_safe_action, _wait_for_change,
    BoostGateStatus, BoostResult, BoostTileStatus,
    read_run_result, read_wallet, wait_for_result_frame,
)
from .farm_cards import _CARD_CENTERS, _CARD_HALF, _alert_user, _card_pair, _cardgame
from .farm_boosts import (
    _BOOST_TEMPLATES, _HS_STOCK_BOX, _RUN_BOOST_TILES, _TILE_CENTERS, _Box,
    _watch_headstart, buy_double_coins, ensure_run_boosts,
    read_boost_gate_status, format_boost_gate_status,
)


def ensure_running(dev, matcher, cfg=None, tries: int = 240, log=print,
                   should_stop=None, cycle=None, gift_state=None) -> bool:
    """Drive any settled post-run screen back into a live run, tapping ONLY buttons we
    can see by image (never a spend/revive button):
      Mystery Box  -> 'Open all' (banks farmed ingredients),
      Result       -> 'OK',
      menu / boost -> 'Play!'  (the two Play taps happen across successive iterations;
                      the run auto-starts after the boost Play, so we never blind-tap).
    Returns True once the background is scrolling (a run is in progress).

    We only act on a SETTLED (static) screen: post-run modals (Result, Mystery Box) ignore
    taps while their intro / coin-counter animation plays, so tapping then just burns the
    budget. scrcpy stops sending frames once a screen is static, so 'settled' = the frame
    stopped changing. This makes each screen take ~1 tap instead of 8-9."""
    cycle = cycle if cycle is not None else {}
    cycle.setdefault("boost_cost", 0)
    cycle.setdefault("required_boost_cost", 0)
    cycle.setdefault("double_coin_cost", 0)
    gift_state = gift_state if gift_state is not None else {}
    gift_state.setdefault("depleted", False)
    prev = None
    action_name = None
    action_seen = 0
    last_tap = None
    tap_repeats = 0
    boost_seen = False        # once we've seen the boost screen, never bare-tap Play again
    unknown_settled = 0       # consecutive settled frames with nothing recognized
    for _ in range(tries):
        if _stop_requested(should_stop):
            return False
        if _in_run(dev, matcher):
            return True
        f = _nav_read(dev)
        if f is None:
            _sleep_interruptible(0.2, should_stop)
            continue
        snap = _snapshot(f)
        settled = prev is not None and snap is not None and _diff(snap, prev) < 2.5
        prev = snap
        visible_action = _visible_safe_action(f, matcher)
        if visible_action and visible_action == action_name:
            action_seen += 1
        else:
            action_name = visible_action
            action_seen = 1 if visible_action else 0
        if not settled:
            if action_seen < 3:
                _sleep_interruptible(0.2, should_stop) # screen still animating — poll it
                continue
            # ponytail: some screens never become pixel-static because button gleams keep
            # moving. Seeing the same safe template three times is enough to act.
        if matcher.present(f, "cardgame", 0.8):
            _cardgame(dev, matcher, log=log, should_stop=should_stop)
            prev = None
            continue
        if matcher.present(f, "boostprompt", 0.75):
            dev.tap(*_STALL_TAP)                       # activate the run-start Fast Start /
            _wait_for_change(dev, f, should_stop=should_stop); prev = None
            continue
        if not gift_state.get("depleted") and gift_button_visible(f, matcher):
            result = draw_gifts(dev, matcher, log=log, should_stop=should_stop)
            if result.opened:
                gift_state["depleted"] = result.depleted
                if result.draws:
                    log(f"[gift] opened {result.draws} gift(s); depleted={result.depleted}")
                elif result.depleted:
                    log("[gift] no draw points available")
                prev = None
                continue
        # Pre-run boost screen: keyed on the always-visible tile grid, NOT on multibtn —
        # the right-hand panel cycles (HP Upgrade / Random Boost / ...) and the pink
        # Multi pill is only on the Random Boost panel, so a multibtn-only gate let the
        # bare play-tap fallthrough start runs with NO boosts (observed live).
        boostish = (matcher.find(f, "tile_hp", 0.80) is not None
                    or matcher.present(f, "multibtn", 0.80)
                    or matcher.present(f, "chesttile", 0.80)
                    or matcher.present(f, "dblbanner", 0.80))
        if boostish:
            boost_seen = True
        if boostish:
            spending = getattr(cfg, "spending", None)
            if spending is not None and spending.allow_coin_boosts:
                status = read_boost_gate_status(f, matcher)
                log("[boost] " + format_boost_gate_status(status))
                if getattr(cfg, "templates_dir", None):
                    hs_stock = read_int(f, _HS_STOCK_BOX, cfg.templates_dir)
                    if hs_stock is not None:
                        log(f"[boost] Head Start stock: {hs_stock}")
                # Already ready (all three tiles checked AND Double Coins banner up)?
                # Press Play NOW. Re-running the tile/buy gauntlet on a ready screen only
                # risks a re-check flaking and a stray tap toggling a tile back off — and
                # both tile checks and the Double Coins boost persist for this run.
                if status.ready_to_play:
                    cycle.setdefault("boost_cost", cycle.get("required_boost_cost", 0)
                                     + cycle.get("double_coin_cost", 0))
                    _tap_template(dev, matcher, "play", 0.80)
                    # arm the Head Start watch NOW: the prompt fires the instant the
                    # run starts and its window is too short for post-detection watchers
                    _watch_headstart(dev, matcher, log=log, should_stop=should_stop)
                    prev = None
                    continue
                required_boosts = ensure_run_boosts(
                    dev, matcher, spending, log=log, should_stop=should_stop)
                if not required_boosts.active:
                    log("[boost] required three boost tiles not verified; not pressing Play")
                    _sleep_interruptible(0.5, should_stop)
                    continue
                cycle["required_boost_cost"] = required_boosts.spent
                # buy_double_coins short-circuits to (True, 0) if the banner is already up.
                # The failed-latch only holds while the banner is genuinely absent: a slow
                # auto-roll can land AFTER our poll ceiling, and the banner also survives a
                # game restart — in both cases fall through so the verified banner unwedges
                # the cycle without any new Multi-Buy tap.
                if cycle.get("double_coin_failed") and not matcher.present(f, "dblbanner", 0.80):
                    log("[boost] Double Coins was already attempted and not verified; not retrying")
                    _sleep_interruptible(0.5, should_stop)
                    continue
                result = buy_double_coins(
                    dev, matcher, spending, log=log, should_stop=should_stop)
                cycle["double_coin_cost"] += result.spent
                if not result.active:
                    cycle["double_coin_failed"] = True
                    log("[boost] Double Coins banner not verified; not pressing Play")
                    _sleep_interruptible(0.5, should_stop)
                    continue
                cycle["double_coin_failed"] = False
                cycle["boost_cost"] = cycle["required_boost_cost"] + cycle["double_coin_cost"]
            _tap_template(dev, matcher, "play", 0.80)  # start the run
            # arm the Head Start watch NOW (see ready_to_play branch)
            _watch_headstart(dev, matcher, log=log, should_stop=should_stop)
            prev = None
            continue
        # Green/teal button templates use a HIGH threshold (0.82): the real reward buttons
        # match ~1.0, and this avoids false-matching the League leaderboard's green message
        # buttons (which was opening Friend's Info / Medal Shop popups mid-navigation).
        tapped = None
        for name, thr in (("openall", 0.82),   # collect mystery boxes
                          ("confirm", 0.82),   # box-reveal / relic (teal)
                          ("confirm2", 0.82),  # daily / reward popup
                          ("ok", 0.82),        # dismiss a Result screen
                          ("close", 0.82),     # close event-popup X
                          ("close2", 0.82),    # close Friend's Info / Medal Shop X (leaderboard wedge)
                          ("play", 0.80)):     # menu Play! (boost Play goes via the branch)
            # 'close'/'close2' on the boost screen would X the Buy-Upgrades panel and HIDE
            # the tile grid the boost gate must verify (observed live) — never tap there.
            if name in ("close", "close2") and boostish:
                continue
            if name == "play":
                # The menu Play! and the boost-screen Play! BOTH match 'play'. Tapping the
                # menu Play lands on the boost screen a beat later, whose Play! then matches
                # too — bare-tapping it would start a boost-LESS run (the exact failure that
                # kept Double Coins off). So before any generic Play tap, poll for the boost
                # screen (its tiles/chest render just after the slide-in). If it's there,
                # hand off to the boost branch instead of tapping Play.
                if boost_seen:
                    continue
                bpt, _bf = _find_stable(dev, matcher, "tile_hp", 0.85, tries=5,
                                        should_stop=should_stop)
                if bpt is None:
                    bpt, _cf = _find_stable(dev, matcher, "chesttile", 0.80, tries=3,
                                            should_stop=should_stop)
                if bpt is not None:
                    boost_seen = True
                    break                      # tapped stays None -> re-enter boost branch
            if _tap_template(dev, matcher, name, thr):
                tapped = name
                break
        if tapped:
            log(f"[nav] {tapped}")
            if tapped == last_tap:
                tap_repeats += 1
            else:
                last_tap, tap_repeats = tapped, 1
            # Repeated NO-EFFECT taps have two known causes, so recover from both:
            # (a) the persistent input shell's REMOTE session died silently (fire-and-
            #     forget writes vanish; poll() still looks healthy — observed live:
            #     verified Play taps never landing for 10+ min) -> respawn the shell;
            # (b) a tap-deaf restored modal (Result/Mystery Box swallow taps for minutes
            #     but KEYCODE_BACK dismisses instantly). Never send BACK for play — on
            #     the menu it opens the quit-game dialog.
            if tap_repeats % 4 == 0:
                if hasattr(dev, "reset_shell"):
                    log("[nav] taps having no effect — respawning input shell")
                    dev.reset_shell()
                if tapped != "play" and hasattr(dev, "back") and _safe_to_back(dev, matcher):
                    log("[nav] tap-deaf modal — sending BACK")
                    dev.back()
            # Pace nav taps to the device's input-processing rate. Taps go out
            # fire-and-forget (persistent adb shell, ~0ms), so without a floor the loop
            # laps a still-animating modal and queues taps that land on the NEXT screen —
            # the reward-popup gauntlet then never converges (observed live). In-run dodges
            # keep their own low latency; only menu navigation is paced here.
            _wait_for_change(dev, f, should_stop=should_stop)
            _sleep_interruptible(0.35, should_stop)
            prev = None
            unknown_settled = 0
        elif settled or action_seen >= 3:
            # A settled screen with NOTHING recognized: a stray tap landed on the League
            # leaderboard and opened Friend's Info / Medal Shop, whose X matches `close`
            # too weakly to tap (0.77 < 0.82) — the classic wedge. KEYCODE_BACK closes
            # those popups; on the bare menu BACK opens the quit dialog, so gate it behind
            # a few consecutive unknown-settled frames (a real menu shows a Play match and
            # never lands here).
            unknown_settled += 1
            if unknown_settled >= 3 and hasattr(dev, "back"):
                if _safe_to_back(dev, matcher):
                    log("[nav] unrecognized settled modal — sending BACK")
                    dev.back()
                else:
                    log("[nav] unrecognized but Play-visible/capture-broken — NOT sending BACK")
                unknown_settled = 0
            _sleep_interruptible(0.4, should_stop)
            prev = None
        else:
            _sleep_interruptible(0.4, should_stop)     # still animating — keep polling
    return _in_run(dev, matcher)



_STALL_TAP = (1280, 640)


_ACTION_NAMES = {
    ACTION_NOOP: "noop",
    ACTION_JUMP: "jump",
    ACTION_SLIDE: "slide",
}


def _decide_action(agent, frame) -> ActionDecision:
    if hasattr(agent, "decide"):
        return agent.decide(frame)
    return ActionDecision(agent.act(frame), "legacy")


def play_until_death(dev, cfg, agent, matcher=None, max_s: float = 3600.0,
                     min_s: float = 8.0, should_stop=None, log=print,
                     on_step=None) -> float:
    """Run the rule-based agent until the run truly ends. A static screen is NOT assumed
    to be death: the run pauses at "Tap to activate ... Boost!" prompts, so on a stall we
    tap to activate/continue and only declare death if the screen stays static after
    several taps. `min_s` ignores the brief static run-start.

    On a stall, if a MENU/popup button (Play/OK/Confirm) is visible we are NOT in a run
    (e.g. a false start left us on the menu) — stop immediately rather than centre-tapping,
    which on the menu lands on the League leaderboard and opens Friend's Cookie popups."""
    slide_ctl = SlideHold(grace_s=getattr(cfg.gestures, "slide_grace_s", 0.30),
                          min_hold_s=getattr(cfg.gestures, "slide_min_hold_s", 0.45))
    try:
        return _run_loop(dev, cfg, agent, matcher, max_s, min_s, should_stop, log,
                         on_step, slide_ctl)
    finally:
        # NEVER leak a held finger past the run: a dangling touch-DOWN would perma-slide
        # the next run and confuse every menu tap in between. force_ (not plain release)
        # so a lost/rejected UP earlier — which already cleared `held` — is still lifted.
        slide_ctl.force_release(dev, cfg.gestures)


def _run_loop(dev, cfg, agent, matcher, max_s, min_s, should_stop, log,
              on_step, slide_ctl) -> float:
    agent.reset()
    slide_ctl.force_release(dev, cfg.gestures)   # clear any finger orphaned by a crashed run
    prev = None
    still = 0
    stall_taps = 0
    hud_missing = 0
    last_action_reason = None
    last_action_log = 0.0
    t0 = time.monotonic()
    tick_s = 1.0 / cfg.decision_hz
    # STREAMING mode when the device supports it: react to every decoded frame the
    # instant it arrives (scrcpy pushes at the display rate) instead of sleeping on a
    # fixed decision tick. tick_s remains the poll fallback / static-screen timeout.
    wait_frame = getattr(dev, "wait_frame", None)
    # The full-res HUD template check costs ~80ms — running it EVERY frame throttled the
    # whole decision loop to ~12fps (measured live; it starved the learned policy's frame
    # stack). Run-end detection only needs a few Hz (30s absence backstop), so gate it.
    hud_check_s = 0.25
    hud_next = 0.0
    hud_absent = False
    headstart_done = False
    hs_prev = None
    while time.monotonic() - t0 < max_s and not _stop_requested(should_stop):
        tick_start = time.monotonic()
        f = wait_frame(timeout=tick_s) if wait_frame is not None else dev.last_frame()
        if f is None:
            if wait_frame is None:
                _sleep_remaining(tick_start, tick_s)
            continue
        menu_up = False
        if matcher is not None and tick_start >= hud_next:
            hud_next = tick_start + hud_check_s
            hud_absent = not matcher.present(f, "slide", 0.60)
            # menu/result probe shares the same cadence: 3 more full-res template finds
            # per frame would crawl the loop to ~4fps through every BONUSTIME washout
            if hud_absent:
                menu_up = any(matcher.find(f, n, 0.82) for n in ("play", "ok", "confirm2"))
            elif not headstart_done and tick_start - t0 < 25.0:
                # run-start "Tap to activate Head Start Boost!" pause: activate it NOW
                # (user directive) — the stall-tap backstop only fired after ~8s of
                # frozen screen, wasting the boost's head start every round.
                # THRESHOLD 0.60: the prompt only lives ~2s (measured in the demo:
                # frames 910-975 @35fps), so THIS 60fps path is the only one fast
                # enough — and window-grab softening puts scores right at 0.75-0.85,
                # which missed live. The strict POSITION GATE below makes a low
                # threshold safe: the prompt button sits near screen centre
                # (~1220,690), while every measured false match (the >> icon in the
                # bottom HUD boost bar etc.) lands far outside the centre box.
                # FIXED-POINT tap: the button animates in with an elastic slide/bounce
                # (live matches at x=1439/1334/1315/1277/1141 — every tap AT the moving
                # match failed to activate). The template near centre only proves the
                # prompt is UP; the button always comes to rest at (1220, 687), so wait
                # a beat for the animation and tap the rest position itself.
                pt = matcher.find(f, "headstart", 0.60)
                # y-gate <130 (was 260): excludes the bottom-HUD ⏩ boost icon (~y869) that the loose
                # gate false-matched + dead-tapped; the real run-start prompt rests at y~687-737.
                if pt is not None and abs(pt[0] - 1220) < 400 and abs(pt[1] - 690) < 130:
                    # STABLE-POINT tap: taps at the historical rest point (1220,687)
                    # and blanket bursts there all failed — today's matches settle at
                    # ~(1290-1345, 700-737), i.e. the button (and its hitbox) moved.
                    # Trust the LIVE match: once two consecutive detections agree
                    # within 20px the button has stopped animating — tap THERE.
                    if hs_prev and abs(pt[0] - hs_prev[0]) < 20 and abs(pt[1] - hs_prev[1]) < 20:
                        log(f"[run] Head Start settled at {pt} — tapping it")
                        slide_ctl.release(dev, cfg.gestures)   # one finger: lift slide first
                        dev.tap(*pt)
                        time.sleep(0.35)
                        dev.tap(*pt)
                        headstart_done = True
                    hs_prev = pt
        if matcher is not None and hud_absent:
            # HUD hidden/washed: suppress inputs NOW, but absence alone is not death —
            # BONUSTIME washouts and bonus scooter rides hide or bleach the HUD for
            # 10-30s stretches (measured live: 0.5s and 3s debounces both false-died
            # mid-run, detaching the bot from an alive run). A true death lands on the
            # Result screen within seconds, so break EARLY on a menu/result template
            # and otherwise only after a sustained 30s absence as the backstop.
            # (0.60 not 0.72: rides bleach the live HUD down to ~0.74 while menus
            # measure <=0.39, so 0.60 keeps washed-but-present HUDs in-run.)
            if hud_missing == 0.0:
                hud_missing = time.monotonic()
            slide_ctl.release(dev, cfg.gestures)   # inputs suppressed => lift the finger
            if menu_up:
                break                       # Result / menu is up => the run truly ended
            if time.monotonic() - hud_missing >= 30.0:
                break
            if wait_frame is None:
                _sleep_remaining(tick_start, tick_s)
            continue
        hud_missing = 0.0
        decision = _decide_action(agent, f)
        if decision.action != ACTION_NOOP:
            now = time.monotonic()
            # streaming mode re-decides per frame (60fps+): log a repeated action at
            # most twice a second
            if decision.reason != last_action_reason or now - last_action_log >= 0.5:
                log(f"[action] {_ACTION_NAMES.get(decision.action, decision.action)} "
                    f"reason={decision.reason} confirmed={decision.confirmed}")
                last_action_reason, last_action_log = decision.reason, now
        # SLIDE is a stateful press-and-hold (SlideHold): DOWN on the first slide
        # prediction, held while the model keeps predicting, UP when it stops — this is
        # what the span-labelled training expresses. It ALSO kills the old input-queue
        # backlog: per-tick `input swipe ... 500` re-fires stacked seconds of queued
        # gestures in the adb shell (LDPlayer has no scrcpy-style one-finger throttle).
        slide_ctl.update(dev, cfg.gestures, decision.action == ACTION_SLIDE)
        if decision.action == ACTION_JUMP:
            if slide_ctl.held:
                slide_ctl.release(dev, cfg.gestures)   # one finger: end slide, then jump
            apply_action(dev, decision.action, cfg.gestures)
        if on_step is not None:               # observer hook (diagnostics): never mutates
            on_step(time.monotonic() - t0, f, decision)
        snap = _snapshot(f)
        if prev is not None and snap is not None and _diff(snap, prev) < 2.5:
            still += 1
            if still >= 6 and (time.monotonic() - t0) > min_s:
                if matcher is not None and any(
                        matcher.find(f, n, 0.82) for n in ("play", "ok", "confirm2")):
                    break                       # a menu/popup is up => not a run => stop
                if stall_taps < 4:              # try to clear a boost prompt / continue
                    slide_ctl.release(dev, cfg.gestures)   # one finger: lift slide first
                    dev.tap(*_STALL_TAP)
                    stall_taps += 1
                    still = 0
                    _wait_for_change(dev, f, timeout_s=0.5, should_stop=should_stop)
                    continue
                # Static after taps: only a real death if the in-run HUD is GONE. With the
                # HUD still visible this is a paused prompt / low-motion stretch (the Head
                # Start dash renders near-identical frames — it false-died at 14s live,
                # abandoning a boosted run that was still going). Real deaths land on the
                # Result screen (HUD absent), which the hud_absent path breaks on.
                if matcher is None or hud_absent:
                    break                       # still static after taps => real death
                still = 0                       # HUD present => alive; keep watching
        else:
            still = 0
            stall_taps = 0                      # motion resumed => alive, reset budget
        prev = snap
        if wait_frame is None:
            _sleep_remaining(tick_start, tick_s)   # streaming mode paces on frame arrival
    return time.monotonic() - t0


def _restart_game(cfg, log=print, should_stop=None) -> None:
    """Force-restart CookieRun to recover a wedge: some post-run popups (e.g. the Mystery
    Box screen) intermittently ignore injected taps during their intro animation, and a
    relaunch clears that and lands back on the menu. No run is in progress when we do this
    (ensure_running just failed), so nothing is lost."""
    import subprocess
    if _stop_requested(should_stop):
        return
    adb = cfg.adb_path or "adb"
    base = [adb] + (["-s", cfg.device_serial] if cfg.device_serial else [])
    pkg = "com.devsisters.crg"
    log("!! stuck - restarting game to recover")
    subprocess.run(base + ["shell", "am", "force-stop", pkg])
    _sleep_interruptible(1.5, should_stop)
    if _stop_requested(should_stop):
        return
    subprocess.run(base + ["shell", "am", "start", "-n",
                           pkg + "/com.devsisters.CookieRunForKakao.OvenbreakX"])
    # ponytail: don't burn a fixed splash delay; ensure_running already polls real frames.
    _sleep_interruptible(3.0, should_stop)


def _runtime_config(cfg, device_serial: str | None = None,
                    capture_backend: str | None = None,
                    adb_path: str | None = None):
    if capture_backend:
        if capture_backend not in CAPTURE_BACKENDS:
            raise ConfigError(f"unknown capture backend: {capture_backend}")
        cfg = replace(cfg, capture_backend=capture_backend)
    if device_serial is not None:
        cfg = replace(cfg, device_serial=device_serial.strip() or None)
    if adb_path is not None:
        cfg = replace(cfg, adb_path=adb_path.strip())
    return cfg


def _adb_path(adb_path: str = "") -> str:
    if adb_path:
        return adb_path
    try:
        import adbutils
        return adbutils.adb_path()
    except Exception:
        return "adb"


def _ready_adb_devices(adb_path: str = "") -> list[str]:
    import subprocess
    out = subprocess.run(
        [_adb_path(adb_path), "devices"],
        capture_output=True,
        text=True,
        timeout=6,
        check=False,
    )
    if out.returncode != 0:
        return []
    devices = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def _auto_serial_config(cfg, log=print):
    import subprocess
    serial = cfg.device_serial or ""
    if ":" in serial:
        # a host:port endpoint (LDPlayer TCP) is only listed after an explicit connect,
        # so a cold emulator start would otherwise fall back to whatever alias exists
        try:
            subprocess.run([_adb_path(cfg.adb_path), "connect", serial],
                           capture_output=True, text=True, timeout=6, check=False)
        except Exception:
            pass
    devices = _ready_adb_devices(cfg.adb_path)
    selected, status = select_adb_serial(cfg.device_serial or "", devices)
    if status == "ready" and selected and selected != (cfg.device_serial or ""):
        log(f"[adb] using {selected} instead of {cfg.device_serial or 'auto'}")
        return replace(cfg, device_serial=selected)
    return cfg


def _reopen_device(dev, cfg, log=print):
    try:
        dev.stop()
    except Exception:
        pass
    dev = open_device(cfg)
    dev.start()
    log(f"device reopened; game-area {getattr(dev, '_ga', '?')} guest {dev.resolution}")
    return dev


def farm(cfg_path: str = "config.yaml", max_runs: int | None = None,
         stop_event=None, log=print, on_result=None,
         allow_coin_boosts: bool | None = None,
         device_serial: str | None = None,
         capture_backend: str | None = None,
         adb_path: str | None = None) -> Metrics:
    dev = None
    cfg = load_config(cfg_path)
    cfg = _runtime_config(cfg, device_serial, capture_backend, adb_path)
    if allow_coin_boosts is not None:
        spending = cfg.spending
        if allow_coin_boosts and spending.max_boost_cost_per_run <= 0:
            spending = replace(spending, max_boost_cost_per_run=12000)
        cfg = replace(
            cfg,
            spending=replace(spending, allow_coin_boosts=allow_coin_boosts),
        )
    cfg = _auto_serial_config(cfg, log=log)
    should_stop = stop_event.is_set if stop_event is not None else None
    old_adb_path = os.environ.get("ADBUTILS_ADB_PATH")
    if cfg.adb_path:
        os.environ["ADBUTILS_ADB_PATH"] = cfg.adb_path
    metrics = Metrics()
    run = 0
    try:
        dev = open_device(cfg)
        dev.start()
        matcher = TemplateMatcher(cfg.templates_dir)
        log(f"device ready; game-area {getattr(dev, '_ga', '?')} guest {dev.resolution}")
        if cfg.spending.allow_coin_boosts:
            missing = [name for name in _BOOST_TEMPLATES if not matcher.has(name)]
            if missing:
                log("[boost] missing templates: " + ", ".join(missing))
        missing_gift = [name for name in GIFT_TEMPLATES if not matcher.has(name)]
        if missing_gift:
            log("[gift] missing templates: " + ", ".join(missing_gift))
        agent = StreamingRuleBasedAgent(cfg)
        crashes = 0
        gift_state = {"depleted": False}
        while (max_runs is None or run < max_runs) and not _stop_requested(should_stop):
            cycle_t0 = time.monotonic()
            cycle = {"boost_cost": 0}
            # Unattended containment: a transient adb/capture error inside one cycle must
            # not kill the whole overnight loop — recover the device + game and go again.
            # Three consecutive crashed cycles means something is truly broken: stop.
            try:
                if not ensure_running(dev, matcher, cfg, log=log,
                                      should_stop=should_stop, cycle=cycle,
                                      gift_state=gift_state):
                    _restart_game(cfg, log=log, should_stop=should_stop)
                    if _stop_requested(should_stop):
                        break
                    dev = _reopen_device(dev, cfg, log=log)
                    # a fresh launch shows a gauntlet of reward popups (daily / run-challenge /
                    # events) before the menu, so give the restart extra patience to clear them.
                    if not ensure_running(dev, matcher, cfg, tries=240, log=log,
                                          should_stop=should_stop, cycle=cycle,
                                          gift_state=gift_state):
                        log("!! could not start a run after restart - stopping")
                        break
                if _stop_requested(should_stop):
                    break
                dur = play_until_death(
                    dev, cfg, agent, matcher, should_stop=should_stop, log=log)
                results = read_run_result(dev, cfg, matcher, should_stop=should_stop)
            except Exception as exc:
                crashes += 1
                log(f"!! cycle crashed ({type(exc).__name__}: {exc}) - "
                    f"recovering ({crashes}/3)")
                if crashes >= 3:
                    raise
                if _stop_requested(should_stop):
                    break
                try:
                    dev = _reopen_device(dev, cfg, log=log)
                except Exception as reopen_exc:
                    log(f"!! device reopen failed ({type(reopen_exc).__name__}: "
                        f"{reopen_exc}) - stopping")
                    raise exc
                _restart_game(cfg, log=log, should_stop=should_stop)
                continue
            crashes = 0
            cycle_s = time.monotonic() - cycle_t0
            run += 1
            result = RunResult(
                results["coins"], results["ingredients"], cycle_s,
                boost_cost=cycle["boost_cost"],
            )
            metrics.add(result)
            log(f"[run {run}] gross={result.coins} boost={result.boost_cost} "
                f"net={result.net_coins} ingredients={result.ingredients} "
                f"survived={dur:.0f}s cycle={cycle_s:.0f}s | {metrics.summary()}")
            if on_result is not None:
                on_result(run, result, metrics)
    finally:
        try:
            if dev is not None:
                dev.stop()
        finally:
            if cfg.adb_path:
                if old_adb_path is None:
                    os.environ.pop("ADBUTILS_ADB_PATH", None)
                else:
                    os.environ["ADBUTILS_ADB_PATH"] = old_adb_path
            log(f"FINAL: {metrics.summary()}")
    return metrics


if __name__ == "__main__":
    args = sys.argv[1:]
    farm(args[0] if args else "config.yaml",
         int(args[1]) if len(args) > 1 else None)
