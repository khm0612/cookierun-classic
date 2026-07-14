"""Conditioning features for the FiLM dodge policy: run-time, scroll speed, bonus phase.

WHY: the plain BC model is a fixed-lead reactive classifier — it learned "fire ~win_pre
seconds before the human did" from mostly early-run frames. But the game SPEEDS UP as a
run goes deeper, and bonus stages are pit-heavy with different physics, so a fixed lead
that is right at 60s is late at 300s (the ~55-65k-collected death band). These three
scalars let the policy learn timing AS A FUNCTION of game state instead:

  t      — run-elapsed seconds / t_norm_s, clamped to [0,1]
  speed  — horizontal scroll speed (px/sec at model input resolution, phase-correlated
           between consecutive stack frames), EMA-smoothed, / speed_norm, clamped [0,2]
  bonus  — 1.0 while the BONUSTIME banner has been seen within the last bonus_latch_s
           (same detector + latch as scripts/ai_farm.py; the banner pulses, the latch
           bridges the dips)

The SAME functions build the vector offline (from recorded demo frames + timestamps in
scripts/train2.py) and live (inside LearnedAgent), so train and inference cannot drift.
All normalisation constants live in model_meta.json under "cond".
"""
from __future__ import annotations
import cv2
import numpy as np

# defaults recorded into meta["cond"] at train time; live code reads them from meta
COND_DIMS = ["t", "speed", "bonus"]
T_NORM_S = 600.0
BONUS_LATCH_S = 3.0
SPEED_EMA = 0.35
BONUS_THRESH = 0.30           # TM_CCOEFF_NORMED on the scale-normalized banner crop
# banner crop as FRACTIONS of the full frame (resolution-independent: works on 960x540
# recordings and 2560x1440 live frames alike) — identical to scripts/ai_farm.bonustime
_BT_ROWS = 0.083
_BT_COL0, _BT_COL1 = 0.04, 0.31
# scroll is measured on the LOWER band of the model-res frame (ground + obstacles move at
# game speed; the sky/background layers parallax slower and would bias a full-frame match)
_SPEED_BAND_Y0 = 0.45

_hann_cache: dict = {}


def _banner_crop(frame) -> "np.ndarray":
    h, w = frame.shape[:2]
    return frame[0:int(h * _BT_ROWS), int(w * _BT_COL0):int(w * _BT_COL1)]


def _banner_match(gray_crop, tpl) -> bool:
    if gray_crop.size == 0:
        return False
    c = cv2.resize(gray_crop, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(c, tpl, cv2.TM_CCOEFF_NORMED)[0, 0]) >= BONUS_THRESH


def bonustime_gray(gray_full, tpl) -> bool:
    """BONUSTIME banner present on a FULL grayscale frame (any resolution)."""
    if tpl is None:
        return False
    return _banner_match(_banner_crop(gray_full), tpl)


def bonustime_bgr(frame_bgr, tpl) -> bool:
    """Crop FIRST, then convert — this runs on the live decide() hot path where a
    full-frame 2560x1440 cvtColor every check would be wasted work on ~98% of pixels."""
    if tpl is None:
        return False
    crop = _banner_crop(frame_bgr)
    if crop.size == 0:
        return False
    return _banner_match(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), tpl)


def load_bonus_template(templates_dir) -> "np.ndarray | None":
    """The machine-local banner template (gitignored); None => bonus stays 0 (soft-off)."""
    import os
    return cv2.imread(os.path.join(str(templates_dir), "bonustime_norm.png"),
                      cv2.IMREAD_GRAYSCALE)


# Estimator version, stamped into meta["cond"]["scroll_v"] at TRAIN time. Live inference
# and gate scoring must use the version a checkpoint was trained with — the v1->v2 offset
# shifts every speed value, so mixing versions across train/live is exactly the drift this
# module exists to prevent. New trainings always use the current version.
SCROLL_V = 2


def estimate_scroll(prev_small, cur_small, scroll_v: int = 1) -> "float | None":
    """Horizontal shift in PX between two consecutive model-res grayscale frames via phase
    correlation on the ground band. Returns None when the correlation peak is too weak to
    trust (menu frames, scene cuts) — the caller keeps its running estimate.

    v1 (legacy, what pre-2026-07-12 film checkpoints trained on): abs(dx), which carries
    cv2.phaseCorrelate's constant +0.5px centroid offset (identical frames read 0.5, a
    3px scroll reads 2.49). v2: offset-corrected signed magnitude — game scroll (content
    moving LEFT) is negative dx, so true px = -(dx - 0.5); reverse motion (never
    legitimate in a runner) clamps to 0, and identical frames read exactly 0."""
    h = prev_small.shape[0]
    y0 = int(h * _SPEED_BAND_Y0)
    a = np.asarray(prev_small[y0:], np.float32)
    b = np.asarray(cur_small[y0:], np.float32)
    if a.shape != b.shape or a.size == 0:
        return None
    key = a.shape
    if key not in _hann_cache:
        _hann_cache[key] = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, _dy), resp = cv2.phaseCorrelate(a, b, _hann_cache[key])
    if resp < 0.10:
        return None
    if scroll_v >= 2:
        return float(min(max(-(dx - 0.5), 0.0), a.shape[1] * 0.25))
    return float(min(abs(dx), a.shape[1] * 0.25))


class CondTracker:
    """Live-side conditioning state for LearnedAgent. Feed it stack-slot transitions via
    on_slot(); read the vector per decision via vector(now). Run boundaries reset explicitly
    through ``LearnedAgent.reset()``; idle phases inside one run preserve this state."""

    def __init__(self, t_norm_s=T_NORM_S, speed_norm=1.0, bonus_latch_s=BONUS_LATCH_S,
                 ema=SPEED_EMA, scroll_v=1):
        self.t_norm_s = float(t_norm_s)
        self.speed_norm = max(float(speed_norm), 1e-6)
        self.bonus_latch_s = float(bonus_latch_s)
        self.ema = float(ema)
        self.scroll_v = int(scroll_v)
        self.reset()

    def reset(self) -> None:
        self._t0 = None
        self._speed = None            # px/sec EMA, None until first measurement
        self._bonus_seen = -1e9

    def on_slot(self, prev_small, new_small, dt: float) -> None:
        """Called when the agent appends a NEW frame-stack slot (prev/new at model res)."""
        if prev_small is None or dt <= 1e-3 or dt > 0.5:
            return                    # unusable gap: keep the running EMA
        px = estimate_scroll(prev_small, new_small, self.scroll_v)
        if px is None:
            return
        inst = px / dt
        self._speed = inst if self._speed is None else \
            self.ema * inst + (1.0 - self.ema) * self._speed

    def bonus_seen(self, now: float) -> None:
        self._bonus_seen = now

    def vector(self, now: float) -> np.ndarray:
        if self._t0 is None:
            self._t0 = now
        t = min((now - self._t0) / self.t_norm_s, 1.0)
        sp = 0.0 if self._speed is None else min(self._speed / self.speed_norm, 2.0)
        b = 1.0 if now - self._bonus_seen < self.bonus_latch_s else 0.0
        return np.array([t, sp, b], np.float32)


def run_speeds(ts, imgs_small, scroll_v: int = 1) -> np.ndarray:
    """RAW px/sec scroll speed per frame of one recorded run (EMA-smoothed, un-normalised
    so the trainer can calibrate speed_norm over the whole corpus). imgs_small = (N,H,W)
    uint8 at model resolution, ts = recording timestamps."""
    n = len(ts)
    out = np.zeros(n, np.float32)
    s = None
    for i in range(1, n):
        dt = float(ts[i] - ts[i - 1])
        if 1e-3 < dt <= 0.5:
            px = estimate_scroll(imgs_small[i - 1], imgs_small[i], scroll_v)
            if px is not None:
                inst = px / dt
                s = inst if s is None else SPEED_EMA * inst + (1.0 - SPEED_EMA) * s
        out[i] = 0.0 if s is None else s
    if n > 1:
        out[0] = out[1]
    return out


def latch_bonus(ts, bt_raw, latch_s=BONUS_LATCH_S) -> np.ndarray:
    """Apply the live 3s latch to raw per-frame banner detections (bool array). bt_raw=None
    (no banner template on this machine) => all-0, the same soft-off that train2's WARNING
    path and LearnedAgent's missing-template path produce — scoring must not crash where
    training and live both degrade gracefully."""
    if bt_raw is None:
        return np.zeros(len(ts), np.float32)
    out = np.zeros(len(ts), np.float32)
    seen = -1e9
    for i, t in enumerate(ts):
        if bt_raw[i]:
            seen = t
        out[i] = 1.0 if t - seen < latch_s else 0.0
    return out


def build_run_cond(ts, speeds_raw, bonus_latched, t_norm_s, speed_norm) -> np.ndarray:
    """Assemble the (N,3) float32 cond array for one run from precomputed pieces."""
    ts = np.asarray(ts, np.float64)
    cond = np.zeros((len(ts), 3), np.float32)
    cond[:, 0] = np.clip((ts - ts[0]) / t_norm_s, 0.0, 1.0)
    cond[:, 1] = np.clip(np.asarray(speeds_raw, np.float32) / max(speed_norm, 1e-6), 0.0, 2.0)
    cond[:, 2] = bonus_latched
    return cond
