import numpy as np
import cv2
import pytest
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.detect import (
    TemplateMatcher, read_int, read_mystery_boxes,
)


def _digit_image(text, size=(200, 60)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(img, text, (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 3, cv2.LINE_AA)
    return img


def _cfg(regions):
    return Config(None, "scrcpy", 60, 15, "Episode 1", regions,
                  Gestures((0, 0), (0, 0), 300),
                  RewardWeights(1, 50, 0.01, 10), ["ok"], ["buy"], "templates")


def test_template_matcher_finds_known_template(tmp_path):
    tpl = np.zeros((30, 30, 3), dtype=np.uint8)
    tpl[:15, :] = 200                               # patterned so TM_CCOEFF_NORMED is well-defined
    cv2.imwrite(str(tmp_path / "blob.png"), tpl)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[100:130, 50:80] = tpl                     # place identical patch
    m = TemplateMatcher(str(tmp_path))
    assert m.present(frame, "blob", threshold=0.9) is True
    assert m.find(frame, "blob", threshold=0.9) is not None
    assert m.present(np.zeros((200, 200, 3), np.uint8), "blob") is False


@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="tesseract not installed"
)
def test_read_int_reads_digits():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[0:60, 0:200] = _digit_image("1234")
    val = read_int(frame, Region(0, 0, 200, 60))
    assert val == 1234


def test_read_mystery_boxes_zero_on_unreadable():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    cfg = _cfg({"mystery_box_counter": Region(0, 0, 50, 50),
                "coin_counter": Region(0, 0, 50, 50),
                "results_coins": Region(0, 0, 50, 50),
                "results_ingredients": Region(0, 0, 50, 50),
                "play_area": Region(0, 0, 50, 50)})
    assert read_mystery_boxes(frame, cfg) == 0     # blank -> 0, never crashes
