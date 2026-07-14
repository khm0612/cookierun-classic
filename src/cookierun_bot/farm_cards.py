"""Post-run card detection for the farm-side stand-down path.

Only ``scripts/monitor.py`` owns card taps. The farm process keeps navigation paused until
that monitor (or a user) clears the board.
"""
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
    """Return (i, j, margin) of the answer pair. USER-CONFIRMED structure (2026-07-05): the board
    is 4 IDENTICAL decoy cards + 2 IDENTICAL answer cards (same picture within each group). So the
    2 answers are near-identical to EACH OTHER and different from the 4 decoys -> they have the
    LOWEST average PICTURE similarity to the group. We compute pairwise normalized cross-correlation
    on the card crops and pick the 2 lowest-avg-similarity cards; margin = how cleanly those 2
    separate from the 4-decoy cluster. Feed a TEMPORAL-MEDIAN frame (monitor.median_grab) so the
    animated sparkles don't corrupt the match — that was the documented reason raw pixels failed.
    Crop is LEFT-biased because the gingerbread mascot occludes a right-column card's right edge.

    (Superseded the bbox-ASPECT heuristic: aspect throws away the picture and can't separate subtle
    same-aspect poses; on de-animated boards pairwise similarity is a far stronger signal, and it
    agreed with the old solver on 13/15 of its confident boards even on animated data.)"""
    import cv2
    crops = []
    for cx, cy in _CARD_CENTERS:
        c = frame[cy - 185:cy + 185, cx - 158:cx + 78]
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, (72, 108))
        g = cv2.GaussianBlur(g, (3, 3), 0)                 # robust to sub-pixel sprite jitter
        g -= g.mean()
        crops.append(g / (float(np.linalg.norm(g)) + 1e-6))  # unit-norm zero-mean -> dot == NCC
    n = len(crops)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i][j] = float((crops[i] * crops[j]).sum())
    avg = S.sum(1) / (n - 1)                                # mean similarity to the other 5 cards
    order = np.argsort(avg)                                 # ascending: 2 answers (lowest) first
    margin = float((avg[order[2]] - avg[order[1]]) * 20)    # gap: 2-answer cluster -> 4-decoy cluster
    return int(order[0]), int(order[1]), margin


def _cardgame(dev, matcher, log=print, should_stop=None) -> None:
    """Pause farm navigation until the independent monitor clears the card bonus."""
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
    # ponytail: one tap owner is safer than coordinating two independent solvers.
    log(f">> card game up — MODEL STOPPED, waiting for monitor/user to solve all rounds. "
        f"(heuristic reference only: cards {i+1} & {j+1}, margin {margin:.1f})")
    _alert_user()
    last_ping = time.monotonic()
    while not _stop_requested(should_stop):
        f = _nav_read(dev)
        if f is None:
            # a capture stall (None frame) is NOT a cleared screen — declaring "cleared"
            # here emits a false resume signal during a known capture hiccup. Keep waiting.
            _sleep_interruptible(0.5, should_stop)
            continue
        if not matcher.present(f, "cardgame", 0.8):
            log(">> card game cleared — resuming.")
            return
        now = time.monotonic()
        if now - last_ping > 20.0:
            last_ping = now
            _alert_user()
        _sleep_interruptible(0.5, should_stop)
