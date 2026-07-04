from __future__ import annotations
import argparse
import time

from ..config import load_config
from ..detect import TemplateMatcher
from ..device import open_device
from ..farm import format_boost_gate_status, read_boost_gate_status


def watch_device(device, cfg, frames: int | None = None,
                 interval_s: float = 0.25, now=time.monotonic,
                 sleep=time.sleep, out=print) -> None:
    matcher = TemplateMatcher(cfg.templates_dir)
    frame_no = 0
    start = now()
    while frames is None or frame_no < frames:
        frame_no += 1
        frame = device.last_frame()
        if frame is None:
            out(f"[frame {frame_no}] no frame")
        else:
            status = read_boost_gate_status(frame, matcher)
            out(f"[frame {frame_no}] t={max(0.0, now() - start):.3f}s "
                f"{format_boost_gate_status(status)}")
        if frames is None or frame_no < frames:
            sleep(interval_s)


def watch(cfg_path: str = "config.yaml", frames: int | None = None,
          interval_s: float = 0.25) -> None:
    cfg = load_config(cfg_path)
    device = open_device(cfg)
    device.start()
    try:
        watch_device(device, cfg, frames=frames, interval_s=interval_s)
    finally:
        device.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only pre-run boost gate checker. Sends no input."
    )
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args(argv)
    watch(args.config, frames=args.frames, interval_s=args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
