from __future__ import annotations
from dataclasses import dataclass
import zlib
import cv2
import numpy as np

from ..gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE

# Geometry for CookieRun Classic. The cookie runs at a fixed spot on the left (~x/W 0.23,
# feet on the ground line ~y/H 0.80); obstacles and coins scroll in from the right. Bands
# are expressed as fractions of frame (W,H) so they hold at any capture resolution. Tuned
# from recorded 2560x1440 footage.
_GROUND_BAND = (0.297, 0.680, 0.461, 0.903)   # just ahead at ground/body height -> jump
_OVER_BAND = (0.289, 0.389, 0.422, 0.597)     # ahead at head height -> slide under
_COIN_BAND = (0.289, 0.299, 0.586, 0.680)     # ahead + above: airborne coin arcs -> jump

# Edge-fraction is a weak signal in this cluttered game (measured range 0.01-0.08: clear
# track ~0.02-0.03, obstacle/platforming-dense sections ~0.06-0.07). Thresholds sit just
# above the clear-track median so jump/slide fire at the busy/obstacle stretches — the old
# 0.16 was unreachable, so the agent never dodged at all.
_OBSTACLE_FRAC = 0.060    # only the busiest/obstacle-dense frames -> jump (naive lower values
_OVER_FRAC = 0.090        # over-reacted and DIED faster; slide is risky so keep it rare)
_COIN_FRAC = 0.010        # gold-coin fraction in the coin band worth jumping for
# Diagnosis data: 360 jumps in a 201s run = the cookie was airborne ~half the time, landing
# on obstacles as often as clearing them. Jumps must be RARE and REASONED; the cooldown must
# outlast a full jump arc so we never chain blind arcs.
_JUMP_COOLDOWN_S = 0.8


def _crop(frame, box):
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = box
    return frame[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)]


def _frame_signature(frame) -> tuple[tuple[int, ...], int]:
    """Cheap content signature for duplicate-frame suppression.

    Some capture/read paths reuse the same ndarray object while replacing its pixels.
    Object identity would mark every later frame as a duplicate, so sample pixels instead.
    """
    h, w = frame.shape[:2]
    sy = max(1, h // 48)
    sx = max(1, w // 86)
    sample = frame[::sy, ::sx]
    return tuple(frame.shape), zlib.crc32(sample.tobytes())


def _edge_fraction(band) -> float:
    """Obstacles/enemies have strong silhouettes; the flat scrolling ground/sky does not.
    Edge density is a far better 'something solid is here' signal than raw darkness."""
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    edges = cv2.Canny(gray, 80, 200)
    return float((edges > 0).mean())


def _gold_fraction(band) -> float:
    """Fraction of bright gold/yellow pixels (coins) in the band."""
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([18, 90, 150]), np.array([40, 255, 255]))
    return float((mask > 0).mean())


# Floor strips at the very bottom (fractions of W,H): the cookie stands on the REF strip;
# AHEAD is where a pit would appear. Falling into a pit is INSTANT death (HP doesn't save
# you), so this is the top-priority jump trigger.
_FLOOR_AHEAD = (0.33, 0.80, 0.46, 0.90)   # floor just ahead of the cookie
_FLOOR_REF = (0.12, 0.80, 0.25, 0.90)     # floor under/behind the cookie (the reference)


def _green_floor_fraction(region) -> float:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 60, 55]), np.array([95, 255, 255]))
    return float((mask > 0).mean())


def _pit_ahead(frame) -> bool:
    """True if the floor AHEAD is a pit: much darker than, and a very different colour
    from, the solid floor under the cookie. Measured on real footage — pits: ratio<0.5 &
    colourdist>200; solid track: ratio 0.85-1.06 & colourdist<100. Comparing to the floor
    under the cookie makes it adapt to each zone's floor colour. Dark Forest floors can be
    too dark for the brightness gate, so also compare the green grass/floor strip."""
    ahead = _crop(frame, _FLOOR_AHEAD)
    ref = _crop(frame, _FLOOR_REF)
    if ahead.size == 0 or ref.size == 0:
        return False
    ref_green = _green_floor_fraction(ref)
    if ref_green > 0.015:
        ahead_green = _green_floor_fraction(ahead)
        return ahead_green < ref_green * 0.35 and (ref_green - ahead_green) > 0.01
    ab = float(cv2.cvtColor(ahead, cv2.COLOR_BGR2GRAY).mean())
    rb = float(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).mean())
    cd = float(np.abs(ahead.reshape(-1, 3).mean(0) - ref.reshape(-1, 3).mean(0)).sum())
    return rb > 60 and ab < rb * 0.55 and cd > 150


# Ahead-band for solid ground obstacles (fractions of W,H). Tuned from a gallery of real
# HP-hit frames (hit_logger): the dominant damage source is large saturated-orange blobs at
# ground level in DARK zones — spiky pumpkin enemies and trunk walls in the Dark Forest.
# Everything orange at ground level wants the same response: JUMP (dodge pumpkins/trunks,
# bounce on potato/burger skill items, collect coin stacks). The dark-background gate stops
# false fires on the oven's red bricks / desert sand (whole-background orange).
# Hazard scan region: full height, from just ahead of the cookie to far right (lead time).
# Geometry decides the dodge (taxonomy from real hit footage): the SAME orange palette is a
# spiky pumpkin when it's a ground blob (-> JUMP) but a hanging trunk wall when it's a
# column anchored to the top of the screen (-> SLIDE; those are NOT jumpable). Both carry
# black jack-o-lantern face cutouts, which the safe orange bounce items (potato/burger) lack.
_HAZ_REGION = (0.30, 0.0, 0.75, 0.85)
_HAZ_TRIGGER_X = 0.62     # earlier lead time for capture/input delay; classifier still gated
_HAZ_BG_VAL = 105         # dark-zone gate (oven bricks / desert sand false-fire otherwise)
_HAZ_MIN_AREA = 0.012     # component area vs region area
_HAZ_FACE_FRAC = 0.04     # fraction of dark (V<60) pixels inside the component bbox
_HAZ_MAX_W = 720          # swept on the 143-frame corpus: 320/448/576 each DROP a borderline
                          # calibrated true positive (the fixed 5x5 morphology over-erodes at
                          # small scales); 720 reproduces the exact verified fire-set at ~15ms
                          # vs ~32ms native — detectors are ratio-based above that.
_HAZ_DOWNSCALE_MIN_FRAME_W = 1800


_K5 = np.ones((5, 5), np.uint8)


def _region_hsv(frame):
    """Shared crop+HSV for the whole hazard suite (one conversion per frame)."""
    region = _crop(frame, _HAZ_REGION)
    if region.size == 0:
        return None, 0.0, 1.0
    h, w = frame.shape[:2]
    scale = min(1.0, _HAZ_MAX_W / max(1, region.shape[1])) if w >= _HAZ_DOWNSCALE_MIN_FRAME_W else 1.0
    if scale < 1.0:
        region = cv2.resize(
            region,
            (int(region.shape[1] * scale), int(region.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    trigger_px = (_HAZ_TRIGGER_X - _HAZ_REGION[0]) * w * scale
    return cv2.cvtColor(region, cv2.COLOR_BGR2HSV), trigger_px, scale


def _comps(mask, min_area_px: float):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in stats[i]) for i in range(1, n)
            if stats[i][4] >= min_area_px]


def _orange_hazard_hsv(hsv, fw, fh, trigger_px):
    """'jump' (ground pumpkin) / 'slide' (top-anchored trunk wall) / None.
    Dark zones only (bg gate); the black jack-o-lantern face separates hazards from the
    safe orange bounce items. Live-proven (153 fires in one run, all genuine)."""
    mask = cv2.inRange(hsv, np.array([8, 140, 120]), np.array([32, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5)
    bg = hsv[:, :, 2][mask == 0]
    if bg.size == 0 or float(np.median(bg)) >= _HAZ_BG_VAL:
        return None
    rh, rw = mask.shape
    dark = (hsv[:, :, 2] < 60)
    best = None                                             # (leading_x, action)
    for x, y, bw, bh, area in _comps(mask, _HAZ_MIN_AREA * mask.size):
        if x > trigger_px:
            continue
        face = float(dark[y:y + bh, x:x + bw].mean())
        if face < _HAZ_FACE_FRAC:
            continue                                        # no black face => safe item
        if y < 0.04 * rh and bh > 0.45 * rh:
            action = "slide"                                # column hanging from the top
        elif (y + bh) > 0.62 * rh:
            action = "jump"                                 # sits at ground level
        else:
            continue                                        # mid-air decor — ignore
        if best is None or x < best[0]:
            best = (x, action)
    return best[1] if best else None


def _orange_hazard(frame):
    """Back-compat wrapper (tests + older callers)."""
    hsv, trigger_px, scale = _region_hsv(frame)
    if hsv is None:
        return None
    h, w = frame.shape[:2]
    return _orange_hazard_hsv(hsv, w * scale, h * scale, trigger_px)


# --- Taxonomy detectors, CALIBRATED against 143 real frames by a 7-agent measure/verify
# workflow (per-class pixel measurement -> gate tuning -> adversarial cross-verification;
# every constant below is measurement-backed, see the tune notes in the session log).
# Every detector keeps the jump-spam discipline: tight colour mask + geometry gate +
# confusable veto, and fires only when the blob's leading edge is inside the trigger
# column. False negatives cost one HP hit; false positives cost uncontrolled air time
# (the proven survival killer), so gates err toward NOT firing.
# Verified fire-set on the 143-frame corpus (zero unjustified false positives):
#   ice_tower  hit_01_82s + f_042_66s     scissors  f_115_179s (slide)
#   hedgehog   f_105/f_116/f_119          rocks     f_105_164s + f_117_181s
#   pins       none (deliberately inert)  ---

def _ice_tower(hsv, fw, fh, trigger_px):
    """Pale-cyan cat tower with a black cat inside: ground column -> jump.
    Calibrated: S hi 175 (towers measure S p50 104-159; saturated sky is S>=190) and
    height floor 0.10*fh — thorn vines fragment the blob so only the lower body (~0.12*fh)
    is one component; the old 0.24*fh floor was why the detector never fired. This class
    is the SOLE coverage for the hit_01_82s HP hit (bright zone: the orange detector's
    dark-zone gate blocks it there)."""
    mask = cv2.inRange(hsv, np.array([90, 50, 190]), np.array([110, 175, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5)
    rh = mask.shape[0]
    dark = (hsv[:, :, 2] < 70)
    for x, y, bw, bh, area in _comps(mask, 0.0015 * fw * fh):
        if x > trigger_px:
            continue
        if bh < 0.10 * fh or not (0.05 * fw <= bw <= 0.12 * fw):
            continue                                        # lower-body proportions
        if (y + bh) < 0.85 * rh:
            continue                                        # must stand on the ground
        if float(dark[y:y + bh, x:x + bw].mean()) < 0.03:
            continue                                        # no black cat body => bg hill
        return "jump"
    return None


def _hedgehog(hsv, fw, fh, trigger_px):
    """Spiked hedgehog at ground/platform height -> jump. Calibrated: the maroon
    colourway is WRAPAROUND red (H171-179, not H0-10) and the brown colourway is H8-16
    S160-250, so the mask is two inRange calls OR'd. Spiky hull (>=1.25 excess, real
    bodies measure 1.62-1.81) + pale-face gate (S<110 & V>180 >=6%; the old 'cream'
    S<60&V>200 was unsatisfiable — real faces measure S62-101) reject bounce items,
    brick walls, and hit-flash blobs."""
    mask = (cv2.inRange(hsv, np.array([168, 110, 60]), np.array([179, 210, 175]))
            | cv2.inRange(hsv, np.array([8, 160, 80]), np.array([16, 250, 185])))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5)
    rh = mask.shape[0]
    pale = (hsv[:, :, 1] < 110) & (hsv[:, :, 2] > 180)
    for x, y, bw, bh, area in _comps(mask, 0.0025 * fw * fh):
        if x > trigger_px or (y + bh) < 0.55 * rh:
            continue                                        # too high = jungle underbrush decor
        if not (0.06 * fw <= bw <= 0.12 * fw) or not (0.09 * fh <= bh <= 0.16 * fh):
            continue                                        # body proportions (188-246 x 162-204)
        if not (0.9 <= bw / max(bh, 1) <= 1.6):
            continue                                        # kills brick walls / hit-flash blobs
        blob = mask[y:y + bh, x:x + bw]
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        c_area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if c_area <= 0 or hull_area / c_area < 1.25:
            continue                                        # smooth silhouette => safe item
        if float(pale[y:y + bh, x:x + bw].mean()) < 0.06:
            continue                                        # no pale face => underbrush
        return "jump"
    return None


def _scissor_blades(hsv, fw, fh, trigger_px):
    """Cream scissor blades stabbing down from the ceiling -> SLIDE (never jump).
    Calibrated: the whitish blade core is hue-unstable, so the mask is V>=180 & S<=135 &
    (H<=32 | S<=25); a tall vertical CLOSE bridges the semi-transparent stage-banner strip
    that splits the blade (V drops to ~75 inside it). The navy-head gate is REMOVED (the
    head is off-screen in the actionable mid-stab pose) and replaced by a downward-taper
    gate (blades measure 0.25; text blocks/light shafts/the BONUSTIME giant-cookie turban
    are rectangular). Left-edge components (x<3) are vetoed — that's the turban."""
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((v_ >= 180) & (s_ <= 135) & ((h_ <= 32) | (s_ <= 25))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5)
    close_k = np.ones((max(3, round(91 / 1440 * fh)), max(3, round(13 / 2560 * fw))), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    rh = mask.shape[0]
    for x, y, bw, bh, area in _comps(mask, 0.002 * fw * fh):
        if x < 3 or x > trigger_px or y > 0.04 * rh:
            continue                                        # must hang from the top, not left edge
        if not (0.05 * fw <= bw <= 0.16 * fw) or bh < 0.28 * rh:
            continue                                        # blade-column proportions
        blob = mask[y:y + bh, x:x + bw]
        fifth = max(1, bh // 5)
        top_w = float((blob[:fifth] > 0).any(axis=0).sum())
        bot_w = float((blob[-fifth:] > 0).any(axis=0).sum())
        if top_w <= 0 or bot_w / top_w > 0.55:
            continue                                        # no downward taper => not a blade
        return "slide"
    return None


def _rock_spikes(hsv, fw, fh, trigger_px):
    """Grey triangular rock spikes on the ground (green grass tufts at the base) -> jump.
    Calibrated: S<=65, V 115-230 (the old V<=215 clipped the white peak tips and V>=140
    the shadowed base, which cost the blob its point) + a height floor 0.07*fh that
    rejects flat grey pedestals/platform lips (real spikes measure 0.11-0.14*fh)."""
    mask = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([179, 65, 230]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5)
    rh = mask.shape[0]
    green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([65, 255, 255]))
    for x, y, bw, bh, area in _comps(mask, 0.002 * fw * fh):
        if x > trigger_px or (y + bh) < 0.72 * rh:
            continue                                        # rocks sit at ground level
        if bh < 0.07 * fh:
            continue                                        # flat pedestal / platform lip
        blob = mask[y:y + bh, x:x + bw]
        top = blob[: max(1, bh // 5)]
        top_w = float((top > 0).any(axis=0).sum())
        base_w = float((blob[-max(1, bh // 5):] > 0).any(axis=0).sum())
        if base_w <= 0 or top_w / base_w > 0.5:
            continue                                        # not a peak (round/rectangular)
        base_band = green[min(y + bh, rh - 1):min(y + bh + max(2, bh // 6), rh), x:x + bw]
        if base_band.size == 0 or float((base_band > 0).mean()) < 0.02:
            continue                                        # no grass tufts => bg mountain
        return "jump"
    return None


def _falling_pins(hsv, fw, fh, trigger_px):
    """DELIBERATELY INERT (verified 0 fires on all 143 real frames). Calibration proved
    the taxonomy's 'falling pins' instance is a bonus-time coin shower: the thin silver
    bars are edge-on frames of SPINNING COINS (same sprites appear inside coin chains on
    26 frames). Any gate loose enough to catch them jump-spams every silver-coin chain.
    Kept as a guarded placeholder in case a real pin zone shows up in longer runs."""
    mask = cv2.inRange(hsv, np.array([0, 0, 205]), np.array([179, 30, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bars = []
    for x, y, bw, bh, area in _comps(mask, 0.0004 * fw * fh):
        if x > trigger_px:
            continue
        if not (0.007 * fw <= bw <= 0.02 * fw) or not (0.05 * fh <= bh <= 0.10 * fh):
            continue                                        # pin proportions (~32x95 native)
        if bh / max(bw, 1) < 2.2:
            continue                                        # round => coin, veto
        if area / max(bw * bh, 1) < 0.55:
            continue                                        # sparse => sparkle/text, veto
        bars.append(x)
    bars.sort()
    for i in range(len(bars) - 1):
        if bars[i + 1] - bars[i] <= 0.16 * fw:
            return "jump"                                   # >=2 pins close together = rain
    return None


# BONUSTIME suppression. The greyed banner is a PERMANENT watermark (present in every
# frame), but its 9 letter cells encode game state: live play lights letters in RANDOM
# collection order, while the scripted bonus stage starts with an all-9-lit flash and
# then DRAINS right-to-left — so lit cells form a contiguous prefix from 'B'. During the
# bonus there are zero live hazards; dodging there is pure arc risk. Verified TRUE on
# exactly the 7 bonus-stage frames of the corpus and FALSE on all 136 others (a random
# live pattern that happens to be a prefix is a ~1%/cycle transient alias — acceptable).
_BONUS_BOX = (0.0566, 0.0139, 0.3203, 0.0764)   # the letter tracker, top-left HUD
_BONUS_WH = (675, 90)                           # canonical banner size for cell geometry


def _bonus_active(frame) -> bool:
    crop = _crop(frame, _BONUS_BOX)
    if crop.size == 0:
        return False
    crop = cv2.resize(crop, _BONUS_WH, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lit = (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 120)
    states = []
    for i in range(9):                          # 9 letter cells, 75px pitch, 8px inset
        frac = float(lit[10:80, i * 75 + 8:(i + 1) * 75 - 8].mean())
        states.append("L" if frac > 0.4 else ("G" if frac < 0.15 else "M"))
    k = 0
    while k < 9 and states[k] == "L":
        k += 1
    if k < 3:
        return False                            # too short a prefix / not a drain pattern
    # cell k may be mid-fade; everything after it must be fully grey
    return all(s == "G" for s in states[k + 1:])


def _hazard(frame):
    """Priority-ordered hazard suite; the first confirmed hazard wins. The bonus gate
    runs first (scripted sections have no live hazards). Scissors decide before anything
    else (overhead, do-NOT-jump — verified conflict on f_115 where the orange classifier
    says jump), then the live-proven orange classifier, then the calibrated taxonomy
    classes."""
    if _bonus_active(frame):
        return None
    hsv, trigger_px, scale = _region_hsv(frame)
    if hsv is None:
        return None
    h, w = frame.shape[:2]
    sw, sh = w * scale, h * scale
    for det in (_scissor_blades, _orange_hazard_hsv, _ice_tower,
                _hedgehog, _rock_spikes, _falling_pins):
        action = det(hsv, sw, sh, trigger_px)
        if action:
            return action
    return None


@dataclass
class Features:
    pit_ahead: bool
    hazard: "str | None"      # 'jump' | 'slide' | None from the hazard suite
    low_obstacle: bool
    overhead_obstacle: bool
    coins_ahead: bool


@dataclass(frozen=True)
class ActionDecision:
    action: int
    reason: str
    features: Features | None = None
    confirmed: int = 0


def extract_features(frame, cfg) -> Features:
    return Features(
        pit_ahead=_pit_ahead(frame),
        hazard=_hazard(frame),
        low_obstacle=_edge_fraction(_crop(frame, _GROUND_BAND)) > _OBSTACLE_FRAC,
        overhead_obstacle=_edge_fraction(_crop(frame, _OVER_BAND)) > _OVER_FRAC,
        coins_ahead=_gold_fraction(_crop(frame, _COIN_BAND)) > _COIN_FRAC,
    )


class RuleBasedAgent:
    """Survive first (jump ground obstacles, slide overhead), then actively grab airborne
    coin arcs by jumping into them — the cookie has maxed HP so tanking the odd enemy hit
    is fine, and the coin yield (the farming goal) comes from collection + long survival."""
    def __init__(self, cfg, seek_coins: bool = False):
        self._cfg = cfg
        self._seek_coins = seek_coins   # default OFF: the magnet auto-collects and every
        self._cooldown = 0              # extra jump is an uncontrolled ballistic arc
        self._jump_cooldown = max(1, round(getattr(cfg, "decision_hz", 10) * _JUMP_COOLDOWN_S))

    def reset(self) -> None:
        self._cooldown = 0

    def act(self, frame) -> int:
        if self._cooldown > 0:
            self._cooldown -= 1
        f = extract_features(frame, self._cfg)
        # Jump ONLY with a classified reason (pit / face-gated pumpkin); every unreasoned
        # jump is an uncontrolled arc that lands on obstacles as often as it clears them.
        # The generic edge-density triggers are deliberately NOT wired to actions: measured
        # over full runs they fire on clutter, not hazards, and caused the jump spam.
        if f.pit_ahead:
            self._cooldown = self._jump_cooldown
            return ACTION_JUMP           # PIT ahead => jump (falling = instant death)
        if f.hazard == "slide":
            return ACTION_SLIDE          # hanging trunk wall => duck under (NOT jumpable)
        if f.hazard == "jump" and self._cooldown == 0:
            self._cooldown = self._jump_cooldown
            return ACTION_JUMP           # ground pumpkin => one precise jump over
        if self._seek_coins and f.coins_ahead and self._cooldown == 0:
            self._cooldown = self._jump_cooldown
            return ACTION_JUMP           # jump up to grab an airborne coin arc
        return ACTION_NOOP


class StreamingRuleBasedAgent:
    """Streaming controller over the same detectors.

    It does not treat every frame as an isolated order. Hazards must persist across
    consecutive new frames before we click, duplicate frame reads are ignored, and every
    click has a reason for logging. Pits stay immediate because falling is terminal.
    """
    def __init__(self, cfg, seek_coins: bool = False, confirm_frames: int = 2):
        self._cfg = cfg
        self._seek_coins = seek_coins
        self._confirm_frames = max(1, int(confirm_frames))
        self._jump_cooldown = max(1, round(getattr(cfg, "decision_hz", 10) * _JUMP_COOLDOWN_S))
        self.reset()

    def reset(self) -> None:
        self._cooldown = 0
        self._pending: tuple[int, str] | None = None
        self._pending_count = 0
        self._last_frame_marker: tuple[int, tuple[tuple[int, ...], int]] | None = None

    def _candidate(self, f: Features) -> tuple[int, str]:
        if f.pit_ahead:
            return ACTION_JUMP, "pit"
        if f.hazard == "slide":
            return ACTION_SLIDE, "hazard:slide"
        if f.hazard == "jump":
            return ACTION_JUMP, "hazard:jump"
        if self._seek_coins and f.coins_ahead:
            return ACTION_JUMP, "coins"
        return ACTION_NOOP, "clear"

    def decide(self, frame) -> ActionDecision:
        signature = _frame_signature(frame)
        marker = (id(frame), signature)
        if marker == self._last_frame_marker:
            return ActionDecision(ACTION_NOOP, "duplicate-frame", confirmed=self._pending_count)
        self._last_frame_marker = marker
        if self._cooldown > 0:
            self._cooldown -= 1
        f = extract_features(frame, self._cfg)
        action, reason = self._candidate(f)
        if action == ACTION_NOOP:
            self._pending = None
            self._pending_count = 0
            return ActionDecision(ACTION_NOOP, reason, f)
        if action == ACTION_JUMP and self._cooldown > 0:
            return ActionDecision(ACTION_NOOP, "jump-cooldown", f)
        key = (action, reason)
        if key == self._pending:
            self._pending_count += 1
        else:
            self._pending = key
            self._pending_count = 1
        needed = 1 if reason == "pit" else self._confirm_frames
        if self._pending_count < needed:
            return ActionDecision(ACTION_NOOP, f"confirming:{reason}", f, self._pending_count)
        if action == ACTION_JUMP:
            self._cooldown = self._jump_cooldown
        self._pending = None
        count = self._pending_count
        self._pending_count = 0
        return ActionDecision(action, reason, f, count)

    def act(self, frame) -> int:
        return self.decide(frame).action
