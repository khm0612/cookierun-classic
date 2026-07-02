from __future__ import annotations
import os
import re
import glob
import cv2
import numpy as np

_DIGIT_SIZE = (32, 48)


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


def _digit_mask(img) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else None
    if hsv is not None:
        light = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([179, 95, 255]))
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 135]))
        dark_frac = float((dark > 0).mean())
        light_frac = float((light > 0).mean())
        if 0.02 <= dark_frac <= 0.70 and (float(np.median(hsv[:, :, 2])) > 120 or light_frac > 0.35):
            mask = dark
        else:
            mask = light
    else:
        mask = cv2.inRange(img, 180, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _normalize_digit(mask) -> "np.ndarray | None":
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 3 or h < 8:
        return None
    return cv2.resize(mask[y:y + h, x:x + w], _DIGIT_SIZE, interpolation=cv2.INTER_AREA)


def _load_digit_templates(templates_dir: str) -> dict[str, list[np.ndarray]]:
    templates = {}
    for d in "0123456789":
        paths = glob.glob(os.path.join(templates_dir, "digits", f"{d}*.png"))
        paths += [os.path.join(templates_dir, f"digit_{d}.png")]
        for path in paths:
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            norm = _normalize_digit(_digit_mask(img))
            if norm is not None:
                templates.setdefault(d, []).append(norm)
    return templates


def _digit_boxes(crop) -> list[tuple[int, int, int, int]]:
    mask = _digit_mask(crop)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    min_h = max(14, int(crop.shape[0] * 0.40))
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if h < min_h or area < 40:
            continue
        if w / max(h, 1) > 1.15:       # skip round icons; digits are narrower
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes)


def _read_int_digit_templates(frame, region, templates_dir: str) -> "int | None":
    templates = _load_digit_templates(templates_dir)
    if not templates:
        return None
    crop = region.crop(frame)
    mask = _digit_mask(crop)
    digits = []
    for x, y, w, h in _digit_boxes(crop):
        norm = _normalize_digit(mask[y:y + h, x:x + w])
        if norm is None:
            continue
        best_digit = None
        best_score = -1.0
        for digit, variants in templates.items():
            for template in variants:
                score = 1.0 - float(cv2.absdiff(norm, template).mean()) / 255.0
                if score > best_score:
                    best_digit = digit
                    best_score = score
        if best_digit is None or best_score < 0.55:
            if not digits:
                continue
            break
        digits.append(best_digit)
    return int("".join(digits)) if digits else None


def _read_int_tesseract(crop) -> "int | None":
    try:
        import pytesseract
    except ImportError:
        return None
    # ponytail: OCR is a best-effort screen read at a trust boundary — any failure
    # (missing tesseract binary, bad crop, decode error) must degrade to "unknown"
    # (None) rather than crash the running bot, so we catch broadly here.
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
    except Exception:
        return None
    digits = re.sub(r"\D", "", txt)
    return int(digits) if digits else None


def read_int(frame, region, templates_dir: str | None = None) -> "int | None":
    crop = region.crop(frame)
    val = _read_int_tesseract(crop)
    if val is not None:
        return val
    if templates_dir:
        return _read_int_digit_templates(frame, region, templates_dir)
    return None


def detect_death(frame, matcher: TemplateMatcher) -> bool:
    return matcher.present(frame, "results", 0.8) or matcher.present(frame, "gameover", 0.8)


def read_coins(frame, cfg) -> "int | None":
    return read_int(frame, cfg.regions["coin_counter"], cfg.templates_dir)


def read_mystery_boxes(frame, cfg) -> int:
    """Parse the 'n/3' box counter; return n, or 0 if unreadable."""
    val = read_int(frame, cfg.regions["mystery_box_counter"], cfg.templates_dir)
    if val is None:
        return 0
    return min(val, 3)


def read_results(frame, cfg) -> dict:
    coins = read_int(frame, cfg.regions["results_coins"], cfg.templates_dir) or 0
    ingredients = read_int(frame, cfg.regions["results_ingredients"], cfg.templates_dir) or 0
    return {"coins": coins, "ingredients": ingredients}
