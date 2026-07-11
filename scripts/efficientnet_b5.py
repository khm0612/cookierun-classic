"""Train EfficientNet-B5 and save data/demo/efficientnet_b5.* outputs."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


TRAIN = Path(__file__).with_name("train2.py")

sys.argv = [
    str(TRAIN),
    *sys.argv[1:],
    "--arch",
    "efficientnet_b5",
    "--out-prefix",
    "efficientnet_b5",
]
runpy.run_path(str(TRAIN), run_name="__main__")
