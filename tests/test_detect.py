import sys
import types
import numpy as np
import cv2
import pytest
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot import detect
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
    assert m.has("blob") is True
    assert m.has("missing") is False
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


def test_read_int_can_use_digit_templates_without_tesseract(tmp_path, monkeypatch):
    digits_dir = tmp_path / "digits"
    digits_dir.mkdir()
    for d in "0123456789":
        cv2.imwrite(str(digits_dir / f"{d}.png"), _digit_image(d, size=(60, 80)))
    frame = np.zeros((100, 260, 3), dtype=np.uint8)
    frame[0:80, 0:260] = _digit_image("1203", size=(260, 80))
    fake = types.SimpleNamespace(image_to_string=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no binary")))
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    assert read_int(frame, Region(0, 0, 260, 80), str(tmp_path)) == 1203


def test_read_int_prefers_cached_digit_templates(tmp_path, monkeypatch):
    digits_dir = tmp_path / "digits"
    digits_dir.mkdir()
    for d in "0123456789":
        cv2.imwrite(str(digits_dir / f"{d}.png"), _digit_image(d, size=(60, 80)))
    frame = np.zeros((100, 260, 3), dtype=np.uint8)
    frame[0:80, 0:260] = _digit_image("1203", size=(260, 80))
    detect._load_digit_templates.cache_clear()
    monkeypatch.setattr(
        detect,
        "_read_int_tesseract",
        lambda crop: (_ for _ in ()).throw(AssertionError("slow OCR should not run")),
    )

    assert read_int(frame, Region(0, 0, 260, 80), str(tmp_path)) == 1203
    before = detect._load_digit_templates.cache_info().hits
    assert read_int(frame, Region(0, 0, 260, 80), str(tmp_path)) == 1203
    assert detect._load_digit_templates.cache_info().hits == before + 1


def test_read_mystery_boxes_zero_on_unreadable():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    cfg = _cfg({"mystery_box_counter": Region(0, 0, 50, 50),
                "coin_counter": Region(0, 0, 50, 50),
                "results_coins": Region(0, 0, 50, 50),
                "results_ingredients": Region(0, 0, 50, 50),
                "play_area": Region(0, 0, 50, 50)})
    assert read_mystery_boxes(frame, cfg) == 0     # blank -> 0, never crashes


def test_digit_boxes_splits_touching_digits_instead_of_dropping_them():
    """Regression: a bold comma-grouped balance renders leading digits (e.g. '43' in
    438,651) as ONE tall+wide connected component. The old round-icon filter dropped it,
    truncating the number (438,651 -> 8651; the ~20% result-screen misreads). It must be
    SPLIT into its digits instead."""
    crop = np.zeros((80, 210, 3), np.uint8)
    cv2.rectangle(crop, (46, 25), (46 + 49, 25 + 31), (255, 255, 255), -1)   # merged "43" (w=49)
    for x, w in [(98, 21), (131, 24), (157, 21), (181, 11)]:                 # singles 8,6,5,1
        cv2.rectangle(crop, (x, 25), (x + w, 25 + 30), (255, 255, 255), -1)
    boxes = detect._digit_boxes(crop)
    assert len(boxes) == 6, f"wide blob was not split into 2: {boxes}"       # 2 split + 4 singles
    xs = sorted(x for x, _, _, _ in boxes)
    assert xs[0] < 60 and 60 <= xs[1] < 96                                   # two halves of the merged blob
