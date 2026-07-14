import numpy as np
import pytest

from cookierun_bot.policies import condition


def _textured(h=96, w=224, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), np.uint8)
    # smooth it a bit so phase correlation has broad structure, not per-pixel noise
    import cv2
    return cv2.GaussianBlur(img, (7, 7), 2)


def test_estimate_scroll_recovers_shift():
    img = _textured()
    shifted = np.roll(img, -5, axis=1)          # content moves LEFT like the game scroll
    px = condition.estimate_scroll(img, shifted)
    assert px is not None
    assert abs(px - 5.0) < 1.0


def test_estimate_scroll_rejects_unrelated_frames():
    a = _textured(seed=1)
    b = _textured(seed=2)                        # independent noise: no coherent shift
    px = condition.estimate_scroll(a, b)
    assert px is None or px < 3.0                # low response -> None (or a tiny spurious peak)


def test_estimate_scroll_clamps_to_quarter_width():
    img = _textured()
    px = condition.estimate_scroll(img, np.roll(img, -200, axis=1))
    assert px is None or px <= 224 * 0.25


def test_estimate_scroll_v2_removes_offset():
    # v2 corrects phaseCorrelate's constant +0.5px centroid offset: identical frames read
    # EXACTLY 0 (v1 reads 0.5), and real shifts read true magnitude (v1 reads true-0.5).
    img = _textured()
    assert condition.estimate_scroll(img, img, scroll_v=2) < 1e-9
    px3 = condition.estimate_scroll(img, np.roll(img, -3, axis=1), scroll_v=2)
    assert px3 is not None and abs(px3 - 3.0) < 0.3
    # reverse motion (never legitimate in a runner) clamps to 0
    rev = condition.estimate_scroll(img, np.roll(img, 3, axis=1), scroll_v=2)
    assert rev == 0.0
    # v1 legacy behavior unchanged (deployed checkpoints depend on it)
    assert abs(condition.estimate_scroll(img, img) - 0.5) < 0.05


def test_latch_bonus_bridges_gaps():
    ts = np.arange(0.0, 10.0, 1.0)
    raw = np.zeros(10, bool)
    raw[2] = True
    out = condition.latch_bonus(ts, raw, latch_s=3.0)
    assert out.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_build_run_cond_shapes_and_clipping():
    ts = np.arange(0.0, 700.0, 100.0)            # runs past t_norm_s -> t clamps at 1
    speeds = np.array([0, 50, 100, 200, 400, 800, 1600], np.float32)
    bonus = np.zeros(7, np.float32)
    cond = condition.build_run_cond(ts, speeds, bonus, t_norm_s=600.0, speed_norm=400.0)
    assert cond.shape == (7, 3) and cond.dtype == np.float32
    assert cond[:, 0].max() == 1.0 and cond[:, 0].min() == 0.0
    assert cond[:, 1].max() == 2.0                # speed clamps at 2x norm
    assert (cond[:, 2] == 0).all()


def test_run_speeds_scales_with_scroll_rate():
    # phaseCorrelate's centroid step underestimates absolute px by ~15% — irrelevant here,
    # because speed_norm is calibrated with the SAME estimator (bias cancels). What the
    # cond dim needs is CONSISTENCY: double scroll => ~double estimate, stable over a run.
    base = _textured()
    ts = np.arange(0.0, 1.0, 1 / 60)

    def speeds(px_per_frame):
        imgs = np.stack([np.roll(base, -px_per_frame * i, axis=1) for i in range(len(ts))])
        return condition.run_speeds(ts, imgs)

    s3, s6 = speeds(3), speeds(6)
    assert 100.0 < s3[-1] < 220.0                 # 3 px/frame @60fps = 180 px/s nominal
    assert 1.6 < s6[-1] / s3[-1] < 2.4            # monotone: 2x scroll ~ 2x estimate
    assert s3[0] == s3[1]                         # first frame backfilled


def test_cond_tracker_lifecycle():
    tr = condition.CondTracker(t_norm_s=100.0, speed_norm=100.0, bonus_latch_s=3.0)
    v0 = tr.vector(1000.0)
    assert v0.tolist() == [0.0, 0.0, 0.0]
    img = _textured()
    tr.on_slot(img, np.roll(img, -2, axis=1), dt=1 / 60)   # 2px/frame -> 120 px/s
    tr.bonus_seen(1001.0)
    v1 = tr.vector(1001.0)
    assert v1[0] == pytest.approx(0.01, abs=1e-3)          # 1s / 100
    assert 0.8 < v1[1] < 1.6                               # ~120/100
    assert v1[2] == 1.0
    v2 = tr.vector(1004.5)                                 # latch expired
    assert v2[2] == 0.0
    # Decision gaps are normal mid-run; only an explicit reset marks a new run.
    v3 = tr.vector(1024.5)
    assert v3[0] > 0.2 and v3[1] > 0.0
    v4 = tr.vector(1100.0)                         # long mid-run phase gaps keep state
    assert v4[0] > v3[0] and v4[1] > 0.0
    tr.reset()                                     # run boundaries reset explicitly
    assert tr.vector(1101.0).tolist() == [0.0, 0.0, 0.0]


def test_cond_tracker_does_not_wipe_new_bonus_after_long_decision_gap():
    tr = condition.CondTracker(t_norm_s=100.0, speed_norm=100.0, bonus_latch_s=3.0)
    tr.vector(0.0)

    tr.bonus_seen(61.0)
    v = tr.vector(61.0)

    assert v[0] == pytest.approx(0.61)
    assert v[2] == 1.0


def test_bonustime_gray_soft_off_without_template():
    img = _textured()
    assert condition.bonustime_gray(img, None) is False
    assert condition.bonustime_bgr(np.dstack([img] * 3), None) is False
    assert condition.load_bonus_template("definitely_missing_dir") is None


def test_latch_bonus_none_raw_is_all_zero_soft_off():
    # missing machine-local banner template => bt_raw=None => all-0, NOT a crash — the
    # promotion gate must degrade exactly like train2/LearnedAgent do (fail-soft, not
    # TypeError-fail-closed-forever)
    ts = np.arange(0.0, 5.0, 1.0)
    out = condition.latch_bonus(ts, None)
    assert out.shape == (5,) and (out == 0).all()
