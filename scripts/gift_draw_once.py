from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import os
import time

from cookierun_bot.config import load_config
from cookierun_bot.detect import TemplateMatcher
from cookierun_bot.device import open_device
from cookierun_bot.gift_draw import GIFT_TEMPLATES, draw_gifts
from cookierun_bot.win_input import find_window


def focus_window(title: str) -> bool:
    if os.name != "nt" or not title:
        return False
    user32 = ctypes.windll.user32
    hwnd = find_window(title)
    if not hwnd:
        return False
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--capture", default="adb",
                        choices=("scrcpy", "adb", "ldplayer", "network", "bluestacks"))
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    cfg = replace(load_config(args.config), capture_backend=args.capture)
    if focus_window(cfg.window_title):
        print(f"[gift] focused window: {cfg.window_title}", flush=True)
    matcher = TemplateMatcher(cfg.templates_dir)
    missing = [name for name in GIFT_TEMPLATES if not matcher.has(name)]
    if missing:
        print("[gift] missing templates: " + ", ".join(missing), flush=True)
        return 2

    dev = open_device(cfg)
    try:
        dev.start()
        print(f"[gift] device ready capture={args.capture} guest={dev.resolution}", flush=True)
        result = draw_gifts(dev, matcher, log=lambda msg: print(msg, flush=True),
                            max_steps=args.max_steps)
        print(f"[gift] done draws={result.draws} depleted={result.depleted} "
              f"opened={result.opened}", flush=True)
        return 0
    finally:
        dev.stop()


if __name__ == "__main__":
    raise SystemExit(main())
