from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
CONFIG = ROOT / "config.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
