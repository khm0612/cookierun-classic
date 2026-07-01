import numpy as np
import pytest
from cookierun_bot.config import load_config, Region, ConfigError


def test_region_crop():
    r = Region(10, 20, 3, 4)
    img = np.arange(100 * 100).reshape(100, 100)
    out = r.crop(img)
    assert out.shape == (4, 3)          # (h, w)
    assert out[0, 0] == img[20, 10]


def test_load_config_parses_all_sections(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
device: {serial: null, capture: scrcpy, max_fps: 60}
loop: {target_stage: "Episode 1", decision_hz: 15}
regions:
  play_area: [0, 200, 1080, 1200]
  coin_counter: [800, 40, 200, 60]
  mystery_box_counter: [500, 40, 120, 60]
  results_coins: [400, 900, 300, 80]
  results_ingredients: [400, 1000, 300, 80]
gestures: {jump_button: [200, 1600], slide_button: [880, 1600], slide_hold_ms: 300}
reward: {w_coin: 1.0, w_box: 50.0, w_survive: 0.01, death_penalty: 10.0}
menu:
  allowlist: [restart, replay, collect, ok, start]
  denylist: [revive_crystals, buy, purchase, watch_ad]
templates_dir: templates
        """
    )
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg.capture_backend == "scrcpy"
    assert cfg.decision_hz == 15
    assert cfg.regions["coin_counter"].w == 200
    assert cfg.gestures.slide_hold_ms == 300
    assert cfg.reward.w_box == 50.0
    assert "purchase" in cfg.menu_denylist


def test_load_config_missing_region_raises(tmp_path):
    (tmp_path / "config.yaml").write_text("device: {capture: scrcpy}\n")
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "config.yaml"))
