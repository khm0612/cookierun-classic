from __future__ import annotations
from dataclasses import dataclass, field
import yaml


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int

    def crop(self, img):
        return img[self.y:self.y + self.h, self.x:self.x + self.w]


@dataclass(frozen=True)
class Gestures:
    jump_button: tuple[int, int]
    slide_button: tuple[int, int]
    slide_hold_ms: int
    jump_hold_ms: int = 250       # held press = higher/longer jump than a tap
    # Anti-detection input humanization (0 = off, deterministic). Production configs enable
    # these so the bot never taps the same pixel / holds the same exact ms twice.
    tap_jitter_px: int = 0        # Gaussian scatter radius around the button centre (px)
    hold_jitter_frac: float = 0.0  # +/- fraction jitter on hold duration (e.g. 0.15 = +/-15%)
    # Slide/jump timing (tunable). A slide is a press-and-hold: it stays down while the model
    # predicts slide, plus `slide_grace_s` after it stops, and for at least `slide_min_hold_s`
    # total once started. Defaults are R3's PROVEN values (grace 0.20, NO min-hold, jump cd 0.25):
    # a live A/B showed a 0.45s min-hold DEGRADED R3 (97s/3k vs 318s/99k) — it forced R3's occasional
    # MISTIMED slides to last 0.45s -> sliding through pits -> early death. Raise slide_min_hold only
    # for a model that under-holds correct slides, never a model that slides at the wrong time.
    slide_grace_s: float = 0.40
    slide_min_hold_s: float = 1.5     # user: once it slides, HOLD >= 1.5s — a jump does NOT cut it
                                      # short inside this window (see SlideHold.protecting / farm loop)
    jump_cooldown_s: float = 0.25
    # Min softmax prob for the model to SLIDE. Slides are CHEAP in principle (a wrong slide doesn't
    # directly kill), BUT live A/B (2026-07-06) showed lowering this on R3 to 0.60 DEGRADED survival
    # to ~90s (vs proven 318s): R3's LOW-confidence slides are UNRELIABLE (wrong slides) and a held
    # slide BLOCKS the one-finger jump -> it slides when it should jump. So R3's proven value is the
    # strict 0.90. A retrained model with a RELIABLE slide head could safely slide at a low gate.
    slide_conf: float = 0.35


@dataclass(frozen=True)
class RewardWeights:
    w_coin: float
    w_box: float
    w_survive: float
    death_penalty: float


@dataclass(frozen=True)
class SpendingConfig:
    allow_coin_boosts: bool = False
    max_boost_cost_per_run: int = 0
    double_coins_first_cost: int = 1200
    double_coins_reroll_cost: int = 600
    max_double_coin_rolls: int = 3
    forbid_crystals: bool = True


@dataclass(frozen=True)
class Config:
    device_serial: str | None
    capture_backend: str
    max_fps: int
    decision_hz: int
    target_stage: str
    regions: dict[str, Region]
    gestures: Gestures
    reward: RewardWeights
    menu_allowlist: list[str]
    menu_denylist: list[str]
    templates_dir: str
    # Windows-native input (for emulators like BlueStacks that block `adb shell input`).
    window_title: str = "BlueStacks App Player"
    window_top_bar: int = 40
    window_right_bar: int = 40
    # On-device bridge app (capture_backend == "network").
    phone_host: str = ""
    phone_port: int = 8080
    spending: SpendingConfig = field(default_factory=SpendingConfig)
    adb_path: str = ""


_REQUIRED_REGIONS = [
    "play_area", "coin_counter", "mystery_box_counter",
    "results_coins", "results_ingredients",
]
CAPTURE_BACKENDS = {"scrcpy", "adb", "ldplayer", "bluestacks", "network"}


def _read_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def load_config(path: str = "config.yaml") -> Config:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc

    try:
        regions = {k: Region(*raw["regions"][k]) for k in _REQUIRED_REGIONS}
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"missing/invalid regions: {exc}") from exc

    try:
        dev = raw.get("device", {})
        loop = raw["loop"]
        g = raw["gestures"]
        rw = raw["reward"]
        menu = raw["menu"]
        win = raw.get("window", {})
        spending = raw.get("spending", {})
        capture = dev.get("capture", "scrcpy")
        if capture not in CAPTURE_BACKENDS:
            raise ConfigError(f"unknown capture backend: {capture}")
        phone = raw.get("phone", {})
        if capture == "network" and not phone.get("host"):
            raise ConfigError("phone.host is required for network capture")
        return Config(
            device_serial=dev.get("serial"),
            capture_backend=capture,
            max_fps=int(dev.get("max_fps", 60)),
            decision_hz=int(loop["decision_hz"]),
            target_stage=str(loop["target_stage"]),
            regions=regions,
            gestures=Gestures(tuple(g["jump_button"]), tuple(g["slide_button"]),
                              int(g["slide_hold_ms"]),
                              jump_hold_ms=int(g.get("jump_hold_ms", 250)),
                              tap_jitter_px=int(g.get("tap_jitter_px", 0)),
                              hold_jitter_frac=float(g.get("hold_jitter_frac", 0.0)),
                              slide_grace_s=float(g.get("slide_grace_s", 0.40)),
                              slide_min_hold_s=float(g.get("slide_min_hold_s", 1.5)),
                              jump_cooldown_s=float(g.get("jump_cooldown_s", 0.25)),
                              slide_conf=float(g.get("slide_conf", 0.35))),
            reward=RewardWeights(float(rw["w_coin"]), float(rw["w_box"]),
                                 float(rw["w_survive"]), float(rw["death_penalty"])),
            menu_allowlist=list(menu["allowlist"]),
            menu_denylist=list(menu["denylist"]),
            templates_dir=str(raw.get("templates_dir", "templates")),
            window_title=str(win.get("title", "BlueStacks App Player")),
            window_top_bar=int(win.get("top_bar", 40)),
            window_right_bar=int(win.get("right_bar", 40)),
            phone_host=str(phone.get("host", "")),
            phone_port=int(phone.get("port", 8080)),
            adb_path=str(dev.get("adb_path", "")),
            spending=SpendingConfig(
                allow_coin_boosts=_read_bool(spending.get("allow_coin_boosts"), False),
                max_boost_cost_per_run=int(spending.get("max_boost_cost_per_run", 0)),
                double_coins_first_cost=int(spending.get("double_coins_first_cost", 1200)),
                double_coins_reroll_cost=int(spending.get("double_coins_reroll_cost", 600)),
                max_double_coin_rolls=int(spending.get("max_double_coin_rolls", 3)),
                forbid_crystals=_read_bool(spending.get("forbid_crystals"), True),
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"missing config key: {exc}") from exc
