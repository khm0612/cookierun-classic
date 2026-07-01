from __future__ import annotations
import sys
import time
import cv2
from .config import load_config
from .device import open_device


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    cfg_path = argv[0] if argv else "config.yaml"
    cfg = load_config(cfg_path)
    dev = open_device(cfg)
    dev.start()
    time.sleep(2.0)                     # allow scrcpy frames to arrive
    frame = dev.last_frame()
    dev.stop()
    if frame is None:
        print("No frame captured. Is the phone connected and scrcpy working?")
        return 1
    out = "calibration_screenshot.png"
    cv2.imwrite(out, frame)
    print(f"resolution={dev.resolution} saved={out} shape={frame.shape}")
    print("Open the PNG in an image editor, read pixel rects for each region,")
    print("and crop button/counter images into the templates/ folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
