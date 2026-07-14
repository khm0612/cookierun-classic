from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
CONFIG = ROOT / "config.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def recording_is_complete(metadata) -> bool:
    """Legacy recordings are accepted; explicit failures and empty captures are not."""
    # ponytail: old human demos predate `complete`; nonempty frames are the compatibility gate.
    return metadata.get("complete") is not False and bool(metadata.get("frames"))
