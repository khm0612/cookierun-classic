from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from dataclasses import replace

from ..config import load_config
from ..detect import read_coins, read_results
from ..device import open_device


@dataclass(frozen=True)
class CoinWatchSample:
    frame_no: int
    elapsed_s: float
    coins: int | None
    source: str
    delta: int
    coins_per_hour: float
    result_coins: int
    result_ingredients: int


def coins_per_hour(start: int | None, current: int | None, elapsed_s: float) -> float:
    if start is None or current is None or elapsed_s <= 0 or current < start:
        return 0.0
    return (current - start) / elapsed_s * 3600.0


def read_sample(frame, cfg, frame_no: int, start_coins: int | None,
                elapsed_s: float) -> CoinWatchSample:
    coins = read_coins(frame, cfg)
    results = read_results(frame, cfg)
    source = "live"
    if coins is None and results["coins"] > 0:
        coins = results["coins"]
        source = "results"
    elif coins is None:
        source = "unknown"
    delta = 0 if start_coins is None or coins is None or coins < start_coins else coins - start_coins
    return CoinWatchSample(
        frame_no=frame_no,
        elapsed_s=elapsed_s,
        coins=coins,
        source=source,
        delta=delta,
        coins_per_hour=coins_per_hour(start_coins, coins, elapsed_s),
        result_coins=results["coins"],
        result_ingredients=results["ingredients"],
    )


def format_sample(s: CoinWatchSample) -> str:
    coins = "?" if s.coins is None else str(s.coins)
    return (
        f"[frame {s.frame_no}] t={s.elapsed_s:.1f}s coins={coins} "
        f"source={s.source} delta={s.delta} coins/hr={s.coins_per_hour:.0f} "
        f"results={{coins:{s.result_coins}, ingredients:{s.result_ingredients}}}"
    )


def watch_device(device, cfg, frames: int | None = None, interval_s: float = 1.0,
                 stable_reads: int = 2, now=time.monotonic, sleep=time.sleep,
                 out=print) -> None:
    start_time = now()
    start_coins: int | None = None
    frame_no = 0
    previous_live: int | None = None
    stable_live_reads = 0

    while frames is None or frame_no < frames:
        frame_no += 1
        frame = device.last_frame()
        if frame is None:
            out(f"[frame {frame_no}] no frame")
        else:
            elapsed_s = max(0.0, now() - start_time)
            sample = read_sample(frame, cfg, frame_no, start_coins, elapsed_s)
            if sample.source == "live" and sample.coins is not None:
                tolerance = max(10, int(sample.coins * 0.05))
                if previous_live is not None and abs(sample.coins - previous_live) <= tolerance:
                    stable_live_reads += 1
                else:
                    stable_live_reads = 1
                previous_live = sample.coins
                if stable_live_reads < stable_reads:
                    sample = replace(
                        sample, coins=None, source="unstable",
                        delta=0, coins_per_hour=0.0,
                    )
            if start_coins is None and sample.coins is not None:
                start_coins = sample.coins
                sample = read_sample(frame, cfg, frame_no, start_coins, elapsed_s)
            out(format_sample(sample))

        if frames is None or frame_no < frames:
            sleep(interval_s)


def watch(cfg_path: str = "config.yaml", frames: int | None = None,
          interval_s: float = 1.0, stable_reads: int = 2) -> None:
    cfg = load_config(cfg_path)
    device = open_device(cfg)
    device.start()
    try:
        watch_device(device, cfg, frames=frames, interval_s=interval_s,
                     stable_reads=stable_reads)
    finally:
        device.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only coin watcher. Sends no input.")
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stable-reads", type=int, default=2)
    args = parser.parse_args(argv)
    watch(args.config, frames=args.frames, interval_s=args.interval,
          stable_reads=args.stable_reads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
