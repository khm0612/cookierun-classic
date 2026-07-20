"""Pre-run boost gate: the three mandated tiles, Double Coins Multi-Buy, and the
Head Start activation, plus the boost-screen status read and stock badge."""
from __future__ import annotations
import time

import numpy as np

from .detect import read_int
from .farm_common import (
    _boost_read_fast, _sleep_interruptible, _stop_requested, _wait_for_change, _find_stable,
    _tile_checked, _tile_checked_stable,
    BoostResult, BoostTileStatus, BoostGateStatus,
)

__all__ = [
    "_watch_headstart", "read_boost_gate_status", "format_boost_gate_status",
    "ensure_run_boosts", "buy_double_coins", "_HS_STOCK_BOX", "_TILE_CENTERS",
    "_RUN_BOOST_TILES", "_BOOST_TEMPLATES", "_Box",
]


_BOOST_TEMPLATES = ("multibtn", "pickboosts", "dblcheck", "dblrow", "multibuy", "dblbanner",
                    "tile_hp", "tile_watch", "tile_x2", "tilecheck", "close")

# The three per-run boost tiles the user mandates CHECKED before every run ("always check
# this three options"): extra HP potion (800 coins/run), pocket watch (800 coins/run), and
# the x2 Point Booster (consumes owned stock, no coins). Costs are estimates for net-coin
# accounting; the game charges at Play, not at tap.
_RUN_BOOST_TILES = (("tile_hp", 800), ("tile_watch", 800), ("tile_x2", 0))

# x2 Point Booster runs off finite OWNED stock (no coins). When it depletes the tile can no
# longer be checked — but it earns no coins, so it must NOT be allowed to halt the farm. HP
# and watch are the coin-relevant boosts and stay hard-required; x2 is best-effort.
_OPTIONAL_TILES = frozenset({"tile_x2"})


def _watch_headstart(dev, matcher, timeout_s: float = 15.0, log=print,
                     should_stop=None) -> bool:
    """Watch for the run-start 'Tap to activate Head Start ⏩' prompt and PRESS it.
    USER-confirmed: the prompt appears RIGHT at run start with a VERY SHORT window, so
    this must be armed BEFORE the run begins (called immediately after the boost-screen
    Play tap — the post-detection watchers consistently missed the window). 60fps frames;
    position gate = centre box (the ⏩ icon in the bottom HUD must never be tapped);
    two consecutive agreeing matches (~33ms apart) = the elastic slide-in has settled."""
    wait_frame = getattr(dev, "wait_frame", None)
    prev = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s and not _stop_requested(should_stop):
        f = wait_frame(0.1) if wait_frame else dev.last_frame()
        if f is None:
            continue
        pt = matcher.find(f, "headstart", 0.60)
        # y-gate TIGHTENED to <130 (was 260): the run-start prompt rests at y~687-737, but the
        # loose gate also matched the bottom-HUD ⏩ boost icon at y~869 and tapped THAT (dead tap =
        # Head Start never activated, observed live). <130 keeps the real prompt, excludes the HUD icon.
        if pt is not None and abs(pt[0] - 1220) < 400 and abs(pt[1] - 690) < 130:
            if prev is not None and abs(pt[0] - prev[0]) < 20 and abs(pt[1] - prev[1]) < 20:
                dev.tap(*pt)
                time.sleep(0.25)
                dev.tap(*pt)
                log(f"[boost] Head Start ⏩ pressed at {pt}")
                return True
            prev = pt
    return False


# Fixed tile-grid centers on the 2560x1440 boost screen. Tile ICON templates rot (the
# tile's art wobbles with stock-count/price state — tile_x2 dropped below threshold twice
# in one day), but the GREEN CHECK badge is tight/background-independent and scores
# 0.96-1.0 — so when an icon can't be found, read the badge at the tile's known spot.
# READ-only fallback: we never tap based on these coords.
_TILE_CENTERS = {"tile_hp": (440, 865), "tile_watch": (760, 865), "tile_x2": (1070, 875)}


class _Box:
    """Minimal region shim for detect.read_int (matches config.Region's crop contract)."""
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def crop(self, frame):
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]


# Head Start ⏩ stock badge on the boost screen (2nd-row-left tile, count at its top-right;
# calibrated live: reads exactly the badge, wider boxes bleed in the '800' price above).
# Stock DECREMENTS on a successful activation — the eyes-free proof that ⏩ was pressed:
# consecutive gates differing by 1 mean the run between them consumed a Head Start.
_HS_STOCK_BOX = _Box(480, 1050, 140, 60)


def read_boost_gate_status(frame, matcher) -> BoostGateStatus:
    tiles = []
    for name, _ in _RUN_BOOST_TILES:
        pt = matcher.find(frame, name, 0.80)
        if pt is None and _tile_checked(matcher, frame, _TILE_CENTERS[name]):
            tiles.append(BoostTileStatus(name, True, True))   # badge proves tile + checked
            continue
        tiles.append(BoostTileStatus(name, pt is not None,
                                     bool(pt and _tile_checked(matcher, frame, pt))))
    return BoostGateStatus(
        required_tiles=tuple(tiles),
        double_coin_banner=matcher.present(frame, "dblbanner", 0.80),
        random_boost_button=matcher.present(frame, "multibtn", 0.80),
        pick_boosts_dialog=matcher.present(frame, "pickboosts", 0.80),
        multi_buy_button=matcher.find(frame, "multibuy", 0.80) is not None,
    )


def format_boost_gate_status(status: BoostGateStatus) -> str:
    tiles = " ".join(
        f"{tile.name}={'checked' if tile.checked else ('visible' if tile.visible else 'missing')}"
        for tile in status.required_tiles
    )
    return (
        f"ready={status.ready_to_play} required={status.required_tiles_checked} "
        f"double_banner={status.double_coin_banner} random_boost={status.random_boost_button} "
        f"pick_dialog={status.pick_boosts_dialog} multi_buy={status.multi_buy_button} "
        f"{tiles}"
    )


def ensure_run_boosts(dev, matcher, spending, log=print, should_stop=None) -> BoostResult:
    """On the pre-run boost screen, verify the three user-mandated boost tiles are CHECKED
    (HP potion+ 800c, pocket watch 800c, x2 Point Booster from stock) and tap any UNCHECKED
    tile once to enable it. USER-AUTHORIZED spend ("always check this three options").
    Never taps a tile it can't see or one already checked (a tap would toggle it OFF).
    Returns whether all three required tiles are verified checked plus the estimated
    per-run coin cost of the priced tiles that end up checked."""
    fixed_cost = sum(cost for _name, cost in _RUN_BOOST_TILES)
    if int(spending.max_boost_cost_per_run) < fixed_cost:
        log(f"[boost] fixed tile cost {fixed_cost} exceeds per-run cap; skipping")
        return BoostResult(False, 0)
    cost = 0
    for name, tile_cost in _RUN_BOOST_TILES:
        # FAST PATH (the common case): the three checks PERSIST across runs, so each tile is almost
        # always ALREADY checked. Verify the green-check BADGE at the tile's fixed grid centre FIRST,
        # on a fast (~3ms dxcam) frame — it's capture-robust (~0.99) and needs NO icon match. This
        # skips the doomed 8-poll icon search: the tile_watch/tile_x2 ICON templates have rotted
        # below 0.80, so _find_stable burned ~7s per tile before falling back to this same badge
        # (measured: ~19s for an all-checked gate). Any tile whose badge isn't confirmed here still
        # drops to the full sharp-adb check + enable-tap below, so nothing is weakened.
        ff = _boost_read_fast(dev)
        if ff is not None and _tile_checked(matcher, ff, _TILE_CENTERS[name]):
            cost += tile_cost
            continue
        pt, f = _find_stable(dev, matcher, name, 0.80, should_stop=should_stop)
        if pt is None:
            # Icon templates rot with tile state (stock count / price art) — but the gate
            # only needs "is it CHECKED": read the badge at the tile's fixed grid spot.
            # If checked, done (checks persist). If not, refuse — never tap unseen tiles.
            if f is not None and _tile_checked(matcher, f, _TILE_CENTERS[name]):
                cost += tile_cost
                continue
            if name in _OPTIONAL_TILES:
                log(f"[boost] {name} not verifiable (stock depleted?); continuing best-effort")
                continue
            log(f"[boost] {name} tile not visible after polling; skipping")
            return BoostResult(False, cost)
        if _tile_checked(matcher, f, pt):
            cost += tile_cost
            continue
        # Unchecked: ONE enable tap, then poll-verify. Only one tap ever — a second blind
        # tap after a false-negative re-read would toggle a just-checked tile back OFF.
        dev.tap(*pt)
        _wait_for_change(dev, f, timeout_s=1.0, should_stop=should_stop)
        if not _tile_checked_stable(dev, matcher, name, should_stop=should_stop):
            if name in _OPTIONAL_TILES:
                log(f"[boost] {name} still unchecked after tap (stock depleted?); "
                    f"continuing best-effort")
                continue
            log(f"[boost] {name} still unchecked after tap+retries")
            return BoostResult(False, cost)
        cost += tile_cost
    return BoostResult(True, cost)


def buy_double_coins(dev, matcher, spending, log=print,
                     should_stop=None) -> BoostResult:
    """Activate the Double Coins doubler via the Random-Boost Multi-Buy (USER DIRECTIVE
    2026-07-17: "always activate the coin booster ... no rerolls cap").

    Live-mapped flow (2026-07-18, every step template-gated so a missing screen just skips):
      chesttile (the pink ? Random-Boost tile, ~1,200c) selects the Random Boost PANEL ->
      multibtn (pink "Multi" pill on that panel) opens the "Pick desired Boosts!" dialog ->
      the Double Coins row persists CHECKED (dblcheck; re-check via dblrow if not) ->
      multibuy starts the game's own roll loop, which SPENDS coins (first 1,200 / reroll 600)
      and STOPS ITSELF the moment a selected boost lands -> the dialog closes and the red
      dblbanner shows above Play. Rolls that land other boosts are owned boosts, not waste.

    The old refusal ("no enforceable spend cap") is superseded by the user directive: the
    game's own stop-at-success bounds the spend in practice (~9k coins observed vs ~45k
    earned/run pre-doubling). A 75s watchdog stops WAITING (never the game) so a hung
    dialog cannot stall farming; the caller plays without the doubler on any failure."""
    if not spending.allow_coin_boosts:
        return BoostResult(False, 0)
    est = int(getattr(spending, "double_coins_first_cost", 1200))
    ff = _boost_read_fast(dev)
    if ff is not None and matcher.present(ff, "dblbanner", 0.80):
        log("[boost] Double Coins banner already active")
        return BoostResult(True, 0)
    pt, _f = _find_stable(dev, matcher, "chesttile", 0.80, should_stop=should_stop)
    if pt is None:
        log("[boost] Random-Boost tile not visible; playing without Double Coins")
        return BoostResult(False, 0)
    dev.tap(*pt)
    pt, _f = _find_stable(dev, matcher, "multibtn", 0.80, should_stop=should_stop)
    if pt is None:
        log("[boost] Multi pill not visible; playing without Double Coins")
        return BoostResult(False, 0)
    dev.tap(*pt)
    pt, f = _find_stable(dev, matcher, "pickboosts", 0.80, should_stop=should_stop)
    if pt is None or f is None:
        log("[boost] Pick-Boosts dialog did not open; playing without Double Coins")
        return BoostResult(False, 0)
    if not matcher.present(f, "dblcheck", 0.80):
        row = matcher.find(f, "dblrow", 0.80)
        if row is None:
            log("[boost] Double Coins row not found in dialog; playing without it")
            return BoostResult(False, 0)
        dev.tap(*row)                        # one tap checks the row (selection persists)
        _sleep_interruptible(0.5, should_stop)
    pt, _f = _find_stable(dev, matcher, "multibuy", 0.80, should_stop=should_stop)
    if pt is None:
        log("[boost] Multi-Buy button not found; playing without Double Coins")
        return BoostResult(False, 0)
    dev.tap(*pt)
    log("[boost] Multi-Buy pressed — game rolls until Double Coins lands "
        "(self-stopping; no reroll cap per user directive)")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 75.0 and not _stop_requested(should_stop):
        fr = _boost_read_fast(dev)
        if fr is not None and matcher.present(fr, "dblbanner", 0.80):
            log(f"[boost] Double Coins ACTIVE ({time.monotonic() - t0:.0f}s)")
            return BoostResult(True, est)
        _sleep_interruptible(1.0, should_stop)
    log("[boost] Double Coins banner not seen within 75s — playing without it")
    return BoostResult(False, est)
