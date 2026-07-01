from __future__ import annotations
import os
import re
import glob
import cv2
import numpy as np


class TemplateMatcher:
    def __init__(self, templates_dir: str):
        self._templates: dict[str, np.ndarray] = {}
        for path in glob.glob(os.path.join(templates_dir, "*.png")):
            name = os.path.splitext(os.path.basename(path))[0]
            self._templates[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _match(self, frame, name):
        tpl = self._templates.get(name)
        if tpl is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
            return None
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc, tpl.shape

    def present(self, frame, name, threshold: float = 0.8) -> bool:
        m = self._match(frame, name)
        return bool(m and m[0] >= threshold)

    def find(self, frame, name, threshold: float = 0.8):
        m = self._match(frame, name)
        if not m or m[0] < threshold:
            return None
        (max_val, (mx, my), (th, tw)) = m
        return (mx + tw // 2, my + th // 2)   # center point


def read_int(frame, region) -> "int | None":
    try:
        import pytesseract
    except ImportError:
        return None
    # ponytail: OCR is a best-effort screen read at a trust boundary — any failure
    # (missing tesseract binary, bad crop, decode error) must degrade to "unknown"
    # (None) rather than crash the running bot, so we catch broadly here.
    try:
        crop = region.crop(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
    except Exception:
        return None
    digits = re.sub(r"\D", "", txt)
    return int(digits) if digits else None


def detect_death(frame, matcher: TemplateMatcher) -> bool:
    return matcher.present(frame, "results", 0.8) or matcher.present(frame, "gameover", 0.8)


def read_coins(frame, cfg) -> "int | None":
    return read_int(frame, cfg.regions["coin_counter"])


def read_mystery_boxes(frame, cfg) -> int:
    """Parse the 'n/3' box counter; return n, or 0 if unreadable."""
    val = read_int(frame, cfg.regions["mystery_box_counter"])
    if val is None:
        return 0
    return min(val, 3)


def read_results(frame, cfg) -> dict:
    coins = read_int(frame, cfg.regions["results_coins"]) or 0
    ingredients = read_int(frame, cfg.regions["results_ingredients"]) or 0
    return {"coins": coins, "ingredients": ingredients}
