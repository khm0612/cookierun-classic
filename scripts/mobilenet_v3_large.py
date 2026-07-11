"""Train MobileNetV3-Large and save data/demo/mobilenet_v3_large.* outputs."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


TRAIN = Path(__file__).with_name("train2.py")

sys.argv = [
    str(TRAIN),
    *sys.argv[1:],
    "--arch",
    "mobilenet_v3_large",
    "--out-prefix",
    "mobilenet_v3_large",
]
runpy.run_path(str(TRAIN), run_name="__main__")
