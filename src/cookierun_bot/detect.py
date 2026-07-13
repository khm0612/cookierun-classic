from __future__ import annotations
import os
import re
import glob
from functools import lru_cache
import cv2
import numpy as np

_DIGIT_SIZE = (32, 48)

# Digit-classification gates for the template OCR path (TM_CCOEFF_NORMED on the normalized
# 32x48 glyph bitmap). Deliberately lenient: a genuinely below-threshold OR near-tie digit
# makes the WHOLE number untrustworthy (returns None -> Tesseract) rather than emitting a
# silently-wrong partial read. Validate against real result crops before tightening these.
_DIGIT_ACCEPT = 0.35
_DIGIT_MARGIN = 0.03

# Result-screen coin-field hardening (2026-07-13 offline audit of the result_r*.jpg
# corpus): ~20% of live run results misread — reads-as-0 plus wrong numbers (observed
# live: 11,411 for a ~6-digit total). Reproduced offline: a fixed crop that CLIPS the
# digit row (panel displaced mid entry/bounce animation, or a longer total) still
# "reads" as a confident smaller/different number (108,963 -> 8963 / 408963 / 1113063
# on shifted ROIs). Defenses: read a slightly GROWN crop, and VETO any read whose ink
# touches the crop border (a clipped row must be unreadable, never a wrong number).
_RESULT_COINS_MAX = 400_000     # > the ~280k observed per-run coin ceiling (x2 incl.)
_RESULT_EDGE_PX = 3
_RESULT_GROW = (40, 12, 8, 24)  # left, right, up, down — measured on the audit corpus:
                                # stays inside the bonus icons (~x1810) and the dashed
                                # separators (~y710 / ~y890) around the coins row


class TemplateMatcher:
    def __init__(self, templates_dir: str):
        self._templates: dict[str, np.ndarray] = {}
        for path in glob.glob(os.path.join(templates_dir, "*.png")):
            name = os.path.splitext(os.path.basename(path))[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:               # unreadable file => has() must say missing
                self._templates[name] = img

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

    def has(self, name: str) -> bool:
        return name in self._templates

    def find(self, frame, name, threshold: float = 0.8):
        m = self._match(frame, name)
        if not m or m[0] < threshold:
            return None
        (max_val, (mx, my), (th, tw)) = m
        return (mx + tw // 2, my + th // 2)   # center point


def _light_digit_mask(hsv) -> np.ndarray:
    return cv2.inRange(hsv, np.array([0, 0, 160]), np.array([179, 160, 255]))


def _dark_digit_mask(hsv) -> np.ndarray:
    return cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 135]))


def _digit_mask(img) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else None
    if hsv is not None:
        if float(np.median(hsv[:, :, 2])) > 150 and float(np.median(hsv[:, :, 1])) < 100:
            mask = _dark_digit_mask(hsv)
        else:
            mask = _light_digit_mask(hsv)
    else:
        mask = cv2.inRange(img, 180, 255)
    return mask


def _normalize_digit(mask) -> "np.ndarray | None":
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 3 or h < 8:
        return None
    return cv2.resize(mask[y:y + h, x:x + w], _DIGIT_SIZE, interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=8)
def _load_digit_templates(templates_dir: str) -> dict[str, tuple[np.ndarray, ...]]:
    templates = {}
    seen = set()
    for d in "0123456789":
        paths = glob.glob(os.path.join(templates_dir, "digits", f"{d}*.png"))
        paths += [os.path.join(templates_dir, f"digit_{d}.png")]
        for path in paths:
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else None
            masks = [_digit_mask(img)]
            if hsv is not None:
                masks.extend([_light_digit_mask(hsv), _dark_digit_mask(hsv)])
            for mask in masks:
                norm = _normalize_digit(mask)
                if norm is not None and norm.tobytes() not in seen:
                    templates.setdefault(d, []).append(norm)
                    seen.add(norm.tobytes())
    return {digit: tuple(variants) for digit, variants in templates.items()}


def _digit_boxes(crop) -> list[tuple[int, int, int, int]]:
    mask = _digit_mask(crop)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_h = max(14, int(crop.shape[0] * 0.35))
    singles, wide = [], []
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if h < min_h or area < 40:
            continue
        if w / max(h, 1) > 1.15:
            # A TALL but wide blob is almost always TOUCHING DIGITS (bold comma-grouped
            # balances render e.g. "43" as one component), NOT a round icon (icons are
            # ~square, w/h<=1.15). The old code DROPPED these -> silently truncated leading
            # digits (438,651 read as 8651; the ~20% result "0"/misreads). Split instead.
            wide.append((x, y, w, h))
        else:
            singles.append((x, y, w, h))
    # reference single-digit width, to infer how many digits a wide blob holds
    ref_w = int(np.median([w for _, _, w, _ in singles])) if singles else max(1, int(min_h * 0.62))
    boxes = list(singles)
    for x, y, w, h in wide:
        n = max(2, int(round(w / max(ref_w, 1))))     # e.g. 49px / 24px -> 2 digits
        step = w / n
        boxes += [(int(x + k * step), y, int(round(step)), h) for k in range(n)]
    return sorted(boxes)


def _read_int_digit_templates(frame, region, templates_dir: str,
                              edge_px: int = 0) -> "int | None":
    templates = _load_digit_templates(templates_dir)
    if not templates:
        return None
    crop = region.crop(frame)
    mask = _digit_mask(crop)
    boxes = _digit_boxes(crop)
    if edge_px and boxes:
        # Ink touching the crop border inside the digit row's band means the value may
        # be CLIPPED (panel mid-animation / more digits than the ROI fits). A clipped
        # row read "successfully" is a silently-wrong number -> refuse instead. Checked
        # on the raw MASK, not the boxes: a near-edge sliver of a clipped digit can fall
        # below the component-size floor and vanish from `boxes` while its ink still
        # hugs the border.
        ch, cw = mask.shape[:2]
        top = min(y for _, y, _, _ in boxes)
        bot = max(y + h for _, y, _, h in boxes)
        lo = min(x for x, _, _, _ in boxes)
        hi = max(x + w for x, _, w, _ in boxes)
        yband = mask[max(0, top - 2):min(ch, bot + 2)]
        xband = mask[:, max(0, lo - 2):min(cw, hi + 2)]
        if (yband[:, :edge_px].any() or yband[:, cw - edge_px:].any()
                or xband[:edge_px].any() or xband[ch - edge_px:].any()):
            return None
    digits = []
    for x, y, w, h in boxes:
        norm = _normalize_digit(mask[y:y + h, x:x + w])
        if norm is None:
            continue
        best_digit, best_score, second_score = None, -1.0, -1.0
        for digit, variants in templates.items():
            # TM_CCOEFF_NORMED on the normalized bitmap is far more discriminative of glyph
            # SHAPE than mean-abs-diff, which also rewards matching background pixels.
            dscore = max(
                float(cv2.matchTemplate(norm, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                for template in variants)
            if dscore > best_score:
                best_digit, best_score, second_score = digit, dscore, best_score
            elif dscore > second_score:
                second_score = dscore
        # A digit we can't classify confidently (low score OR a near-tie between glyphs)
        # makes the whole field untrustworthy: bail to None so read_int falls through to
        # Tesseract, instead of the old continue/break that silently dropped a leading digit
        # or truncated the tail (10x/1000x under-reads that passed downstream unchecked).
        if best_digit is None or best_score < _DIGIT_ACCEPT or best_score - second_score < _DIGIT_MARGIN:
            return None
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
        # Tesseract wants ~30px-tall DARK digits on a LIGHT background; game counters are
        # small and light-on-dark, so upscale ~3x and flip polarity when the Otsu field
        # comes out mostly dark (light glyphs on a dark ground).
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(thresh.mean()) < 127.0:
            thresh = cv2.bitwise_not(thresh)
        txt = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
    except Exception:
        return None
    digits = re.sub(r"\D", "", txt)
    return int(digits) if digits else None


def read_int(frame, region, templates_dir: str | None = None) -> "int | None":
    crop = region.crop(frame)
    if templates_dir:
        val = _read_int_digit_templates(frame, region, templates_dir)
        if val is not None:
            return val
    return _read_int_tesseract(crop)


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


def _grown_results_region(region, frame):
    """The coins ROI grown by _RESULT_GROW (clamped to the frame) so a result panel
    displaced by its entry/bounce animation — or a 7-digit total — still lands fully
    inside the crop; the edge-clip veto then guarantees a clipped row is never read."""
    from .config import Region
    gl, gr, gu, gd = _RESULT_GROW
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, region.x - gl), max(0, region.y - gu)
    x1 = min(fw, region.x + region.w + gr)
    y1 = min(fh, region.y + region.h + gd)
    return Region(x0, y0, x1 - x0, y1 - y0)


def read_results(frame, cfg) -> dict:
    # Coins: template-only read on the GROWN crop with the edge-clip veto + a
    # plausibility cap. No Tesseract fallback for this field: it cannot apply the veto,
    # and a clipped crop is exactly when it returns a confident wrong number (the
    # binary is absent on the farm box anyway, so the fallback only ever produced 0s
    # here). An unreadable/implausible coins field returns 0 -> read_run_result's
    # settle loop keeps polling and flags UNREAD, never counts garbage.
    grown = _grown_results_region(cfg.regions["results_coins"], frame)
    coins = _read_int_digit_templates(frame, grown, cfg.templates_dir,
                                      edge_px=_RESULT_EDGE_PX)
    if coins is not None and coins > _RESULT_COINS_MAX:
        coins = None
    ingredients = read_int(frame, cfg.regions["results_ingredients"], cfg.templates_dir) or 0
    return {"coins": coins or 0, "ingredients": ingredients}
