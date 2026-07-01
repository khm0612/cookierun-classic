from __future__ import annotations
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RewardWeights:
    w_coin: float
    w_box: float
    w_survive: float
    death_penalty: float


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


_REQUIRED_REGIONS = [
    "play_area", "coin_counter", "mystery_box_counter",
    "results_coins", "results_ingredients",
]


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
        return Config(
            device_serial=dev.get("serial"),
            capture_backend=dev.get("capture", "scrcpy"),
            max_fps=int(dev.get("max_fps", 60)),
            decision_hz=int(loop["decision_hz"]),
            target_stage=str(loop["target_stage"]),
            regions=regions,
            gestures=Gestures(tuple(g["jump_button"]), tuple(g["slide_button"]),
                              int(g["slide_hold_ms"])),
            reward=RewardWeights(float(rw["w_coin"]), float(rw["w_box"]),
                                 float(rw["w_survive"]), float(rw["death_penalty"])),
            menu_allowlist=list(menu["allowlist"]),
            menu_denylist=list(menu["denylist"]),
            templates_dir=str(raw.get("templates_dir", "templates")),
            window_title=str(win.get("title", "BlueStacks App Player")),
            window_top_bar=int(win.get("top_bar", 40)),
            window_right_bar=int(win.get("right_bar", 40)),
        )
    except KeyError as exc:
        raise ConfigError(f"missing config key: {exc}") from exc
