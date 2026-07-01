import numpy as np
from cookierun_bot.config import Region
from cookierun_bot.capture import preprocess, FrameStack


def test_preprocess_shape_and_dtype():
    frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    out = preprocess(frame, Region(0, 200, 1080, 1200), size=(84, 84))
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8


def test_framestack_reset_repeats_then_push_shifts():
    fs = FrameStack(k=4)
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    pa = Region(0, 0, 1080, 1920)
    stacked = fs.reset(preprocess(frame, pa))
    assert stacked.shape == (4, 84, 84)
    bright = np.full((1920, 1080, 3), 255, dtype=np.uint8)
    stacked2 = fs.push(preprocess(bright, pa))
    assert stacked2.shape == (4, 84, 84)
    assert stacked2[-1].mean() > stacked2[0].mean()   # newest frame is the bright one
