import numpy as np

from cookierun_bot import farm
from cookierun_bot.agents import boost_watch


class GateMatcher:
    def __init__(self, checked=True, banner=True):
        self.checked = checked
        self.banner = banner

    def find(self, frame, name, threshold=0.8):
        if name in {"tile_hp", "tile_watch", "tile_x2", "multibuy"}:
            return (1, 1)
        return None

    def present(self, frame, name, threshold=0.8):
        if name == "tilecheck":
            return self.checked
        if name == "dblbanner":
            return self.banner
        if name in {"multibtn", "pickboosts"}:
            return True
        return False


def test_boost_gate_status_reports_ready_when_required_tiles_and_banner_present():
    status = farm.read_boost_gate_status(np.zeros((4, 4, 3), np.uint8), GateMatcher())

    assert status.required_tiles_checked is True
    assert status.double_coin_banner is True
    assert status.ready_to_play is True
    assert "ready=True" in farm.format_boost_gate_status(status)
    assert "tile_hp=checked" in farm.format_boost_gate_status(status)


def test_boost_gate_status_blocks_without_banner():
    status = farm.read_boost_gate_status(
        np.zeros((4, 4, 3), np.uint8), GateMatcher(checked=True, banner=False)
    )

    assert status.required_tiles_checked is True
    assert status.ready_to_play is False
    assert "double_banner=False" in farm.format_boost_gate_status(status)


def test_boost_watch_prints_status(monkeypatch, fake_device):
    monkeypatch.setattr(boost_watch, "TemplateMatcher", lambda templates_dir: GateMatcher())
    lines = []

    boost_watch.watch_device(
        fake_device,
        type("Cfg", (), {"templates_dir": "templates"})(),
        frames=1,
        interval_s=0,
        now=lambda: 0.0,
        sleep=lambda _: None,
        out=lines.append,
    )

    assert fake_device.taps == []
    assert fake_device.holds == []
    assert "ready=True" in lines[0]
