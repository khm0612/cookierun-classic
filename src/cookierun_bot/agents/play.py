from __future__ import annotations
import sys

from ..farm import farm


def play(cfg_path: str = "config.yaml", max_runs: int | None = None) -> None:
    farm(cfg_path, max_runs)


if __name__ == "__main__":
    args = sys.argv[1:]
    play(args[0] if args else "config.yaml",
         int(args[1]) if len(args) > 1 else None)
