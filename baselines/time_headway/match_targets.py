"""Map each benchmark and challenge-set target to the rollout log that produced it.

A target is identified by its scene, its planning time (``time_now_us`` in the
target's metadata.json) and its trajectory: the matching rollout.asl has a
driver_return at that time whose first ten poses equal the target's
``trajectory_poses`` to within 1 mm. Every target must match exactly one log.
The log's ``nre_uuid`` names the USDZ that holds the map the runtime loaded.

Needs the AlpaSim environment (it reads AlpaSim's log format) and the original
rollouts, which are not released. Run from the repository root, e.g.:

    cd alpasim && uv run --project src/runtime python ../baselines/time_headway/match_targets.py \\
        --rollout-root <rollouts> --usdz-dir <nre-artifacts>/all-usdzs \\
        --output ../runs/time_headway/target_matches.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from alpasim_utils.logs import async_read_pb_log

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETS = {
    "bench150": "data/benchmark/benchmark.json",
    "ood15": "data/benchmark/ood_hazard_2604_unsafe_traces_adjudicated.json",
}
N_POSES = 10
TOLERANCE_M = 1e-3
SCENE_RE = re.compile(r"clipgt-([0-9a-f-]{36})")


def load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["entries"] if isinstance(data, dict) else data


def load_targets(sets: dict[str, Path]) -> list[dict]:
    targets = []
    for name, path in sets.items():
        for item in load_entries(path):
            meta = json.loads((REPO_ROOT / item["source_scene_file"]).read_text())
            poses = meta["trajectory_poses"]
            if isinstance(poses, str):
                poses = ast.literal_eval(poses)
            targets.append({"set": name, "clip_id": item["clip_id"],
                            "scene": SCENE_RE.search(item["clip_id"]).group(1),
                            "time_now_us": int(meta["time_now_us"]),
                            "poses": [(p["x"], p["y"]) for p in poses[:N_POSES]]})
    return targets


async def scan(asl: str, times: set[int]) -> tuple[dict, dict[int, list]]:
    """The log's session metadata and its returned poses at the wanted times."""
    session, returned, now = {}, {}, None
    async for entry in async_read_pb_log(asl):
        kind = entry.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            session = {"nre_uuid": entry.rollout_metadata.session_metadata.nre_uuid}
        elif kind == "driver_request":
            now = entry.driver_request.time_now_us
        elif kind == "driver_return" and now in times:
            returned[now] = [(p.pose.vec.x, p.pose.vec.y)
                             for p in entry.driver_return.trajectory.poses[:N_POSES]]
    return session, returned


def list_logs(args: argparse.Namespace) -> list[str]:
    logs = []
    if args.asl_index:
        logs += args.asl_index.read_text().split()
    for root in args.rollout_root:
        logs += sorted(str(p) for p in root.rglob("rollout.asl"))
    if not logs:
        raise SystemExit("no rollout.asl found; pass --rollout-root or --asl-index")
    return logs


async def run(args: argparse.Namespace) -> None:
    sets = dict(DEFAULT_SETS)
    for spec in args.set or []:
        name, _, path = spec.partition("=")
        sets[name] = path
    targets = load_targets({k: REPO_ROOT / v for k, v in sets.items()})

    logs_by_scene = defaultdict(list)
    for asl in list_logs(args):
        if m := SCENE_RE.search(asl):
            logs_by_scene[m.group(1)].append(asl)
    times_by_scene = defaultdict(set)
    for t in targets:
        times_by_scene[t["scene"]].add(t["time_now_us"])
    scanned = {asl: await scan(asl, times)
               for scene, times in times_by_scene.items()
               for asl in logs_by_scene[scene]}

    matches = []
    for t in targets:
        exact = []
        for asl in logs_by_scene[t["scene"]]:
            returned = scanned[asl][1].get(t["time_now_us"])
            if returned and max(math.dist(a, b) for a, b in zip(returned, t["poses"])) < TOLERANCE_M:
                exact.append(asl)
        if len(exact) != 1:
            raise SystemExit(f"{t['clip_id']}: {len(exact)} matching logs")
        nre_uuid = scanned[exact[0]][0]["nre_uuid"]
        usdz = next((str(p) for d in args.usdz_dir if (p := d / f"{nre_uuid}.usdz").exists()), None)
        if usdz is None:
            raise SystemExit(f"{t['clip_id']}: {nre_uuid}.usdz not found in --usdz-dir")
        matches.append({"set": t["set"], "clip_id": t["clip_id"],
                        "time_now_us": t["time_now_us"], "asl": exact[0], "usdz": usdz})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matches, indent=1) + "\n")
    print(f"matched {len(matches)} targets to {len({m['asl'] for m in matches})} logs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", metavar="NAME=PATH",
                        help="Target set (benchmark-entry JSON, list or {'entries': [...]}). "
                             f"Repeatable; overrides or adds to the defaults {DEFAULT_SETS}.")
    parser.add_argument("--rollout-root", type=Path, action="append", default=[],
                        help="Directory searched recursively for rollout.asl. Repeatable.")
    parser.add_argument("--asl-index", type=Path, help="Text file listing rollout.asl paths.")
    parser.add_argument("--usdz-dir", type=Path, action="append", required=True,
                        help="Directory holding <nre_uuid>.usdz scene files. Repeatable.")
    parser.add_argument("--output", type=Path, default=Path("runs/time_headway/target_matches.json"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
