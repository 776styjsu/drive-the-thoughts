"""Replay each target's rollout log through the Time Headway monitor.

The monitor (two_second_monitor.py, unchanged) is a MessageBroadcaster handler:
live, AlpaSim passes it every LogEntry it also writes to rollout.asl, so feeding
the logged entries to on_message in order reproduces the live run without
re-simulating. The map is loaded from the rollout's USDZ as the runtime does
(Artifact.map).

Writes <alerts-dir>/<rollout uuid>.jsonl (the monitor's own output) and the
per-target results (the monitor's row at each target step; no row means no
alarm). Needs the AlpaSim environment and the rollouts; run after
match_targets.py, e.g.:

    cd alpasim && uv run --project src/runtime python ../baselines/time_headway/replay.py \\
        --matches ../runs/time_headway/target_matches.json \\
        --alerts-dir ../runs/time_headway/alerts --output ../runs/time_headway/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from alpasim_utils.artifact import Artifact
from alpasim_utils.logs import async_read_pb_log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_second_monitor import TwoSecondMonitor  # noqa: E402


class ErrorCounter(logging.Handler):
    """The monitor logs its exceptions instead of raising them; count them."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


async def replay(asl: str, usdz: str, alerts_path: Path) -> TwoSecondMonitor:
    alerts_path.unlink(missing_ok=True)
    monitor = TwoSecondMonitor(str(alerts_path.parent), vector_map=Artifact(source=usdz).map)
    monitor.alerts_path = str(alerts_path)
    async for entry in async_read_pb_log(asl):
        await monitor.on_message(entry)
    return monitor


async def run(args: argparse.Namespace) -> None:
    errors = ErrorCounter()
    logging.getLogger("two_second_monitor").addHandler(errors)
    args.alerts_dir.mkdir(parents=True, exist_ok=True)

    targets_by_log = {}
    for t in json.loads(args.matches.read_text()):
        targets_by_log.setdefault(t["asl"], []).append(t)

    results = []
    for i, (asl, targets) in enumerate(targets_by_log.items(), 1):
        rollout = Path(asl).parent.name
        alerts_path = args.alerts_dir / f"{rollout}.jsonl"
        before = errors.count
        monitor = await replay(asl, targets[0]["usdz"], alerts_path)
        rows = {r["time_s"]: r for r in map(json.loads, alerts_path.read_text().splitlines())}
        for t in targets:
            time_s = round((t["time_now_us"] - monitor.start_us) / 1e6, 2)
            row = rows.get(time_s)
            results.append({"set": t["set"], "clip_id": t["clip_id"], "rollout": rollout,
                            "time_s": time_s,
                            "in_warmup": t["time_now_us"] - monitor.start_us < monitor.warmup_us,
                            "alarm": bool(row and row["alerts"]), "row": row,
                            "window5_alarm": any(rows.get(round(time_s - 0.1 * k, 2), {}).get("alerts")
                                                 for k in range(5)),
                            "monitor_errors": errors.count - before,
                            "has_lanes": monitor.lanes is not None})
        print(f"[{i}/{len(targets_by_log)}] {rollout}: {sum(bool(r['alerts']) for r in rows.values())}"
              f"/{len(rows)} steps alert", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(f"monitor errors: {errors.count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=Path, default=Path("runs/time_headway/target_matches.json"))
    parser.add_argument("--alerts-dir", type=Path, default=Path("runs/time_headway/alerts"))
    parser.add_argument("--output", type=Path, default=Path("runs/time_headway/results.json"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
