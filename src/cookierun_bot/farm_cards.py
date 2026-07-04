"""Post-run "Find the X card!" bonus: card-pair detection + the stand-down policy
(the bot NEVER taps cards; it alerts and waits for the user/Claude to solve)."""
from __future__ import annotations
import os
import time

import numpy as np

from .farm_common import _nav_read, _sleep_interruptible, _stop_requested

__all__ = ["_CARD_CENTERS", "_CARD_HALF", "_alert_user", "_card_pair", "_cardgame"]


_CARD_CENTERS = [(883, 602), (1280, 602), (1677, 602),
                 (883, 1088), (1280, 1088), (1677, 1088)]
_CARD_HALF = (150, 190)          # half-size of the card crop (w, h) at 2560x1440


def _alert_user() -> None:
    """Audible attention ping (Windows system exclamation); silent no-op elsewhere."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass


def _card_pair(frame):
    """Return (i, j, margin) of the answer pair. USER RULE (refined on a live round
    2026-07-04): FOUR of the six cards show one identical decoy pose; the answer is the
    pose that exists as exactly a PAIR. Raw pixel diffs cannot separate the groups
    (animated sparkles + small sprites), but the sprite's SHAPE can: mask the sprite
    against the card's uniform background and use its bbox aspect (a sliding pose is
    wide/low ~0.73; upright ~1.05 — measured live, cards 2&6 vs 1/3/4/5). The two cards
    whose aspect deviates most from the median are the pair. Crop is LEFT-biased because
    the gingerbread mascot can occlude a right-column card's right edge. margin = gap
    between the 2nd and 3rd largest deviations (small = ambiguous round)."""
    import cv2
    devs = []
    aspects = []
    for cx, cy in _CARD_CENTERS:
        c = frame[cy - 170:cy + 170, cx - 135:cx + 60]
        bg = np.median(c.reshape(-1, 3), axis=0)
        dist = np.abs(c.astype(int) - bg).sum(2)
        m = (dist > 60).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(m)
        if len(ys) < 20:
            aspects.append(1.0)
            continue
        aspects.append((ys.max() - ys.min()) / max(xs.max() - xs.min(), 1))
    a = np.array(aspects)
    dev = np.abs(a - np.median(a))
    order = np.argsort(-dev)
    margin = float((dev[order[1]] - dev[order[2]]) * 10)   # ~aspect gap, scaled
    return int(order[0]), int(order[1]), margin


def _cardgame(dev, matcher, log=print, should_stop=None,
              user_grace_s: float = 30.0) -> None:
    """Card bonus, hybrid flow (user directive evolution 2026-07-04): alert the user and
    give them `user_grace_s` to pick (they're best at it); if they don't act, tap the
    outlier-heuristic guess — wrong picks still award a lesser prize, so guessing beats
    stalling an unattended session. Card sprites are ANIMATED (sparkles), so the pixel
    heuristic is genuinely unreliable — every appearance saves an audit frame to keep
    improving it offline."""
    f = _nav_read(dev)
    if f is None:
        return
    try:
        import cv2
        cv2.imwrite(os.path.join("data", "ai_hits", f"cardgame_{int(time.time())}.jpg"),
                    cv2.resize(f, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass
    i, j, margin = _card_pair(f)
    # USER DIRECTIVE (final, 2026-07-04): the bot NEVER taps cards. Two agents acting on
    # the same card screen (the child's grace-expiry guess + Claude/user solving) caused
    # random-looking spam across rounds 2-3 — dangerous. The child's ONLY job here is to
    # stand down completely, announce, and wait; Claude (notified via the log monitor) or
    # the user does the careful 3-round solve externally.
    log(f">> card game up — MODEL STOPPED, WAITING FOR YOU/Claude to solve all rounds. "
        f"(heuristic reference only: cards {i+1} & {j+1}, margin {margin:.1f})")
    _alert_user()
    last_ping = time.monotonic()
    while not _stop_requested(should_stop):
        f = _nav_read(dev)
        if f is None or not matcher.present(f, "cardgame", 0.8):
            log(">> card game cleared — resuming.")
            return
        now = time.monotonic()
        if now - last_ping > 20.0:
            last_ping = now
            _alert_user()
        _sleep_interruptible(0.5, should_stop)
