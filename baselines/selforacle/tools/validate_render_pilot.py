"""CPU-only gate between a two-scene pilot and the full rendering array."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def validate(campaign):
    with (campaign / "cohort.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:2]
    if len(rows) != 2:
        raise ValueError("The gate requires two pilot scenes")
    reports = []
    for row in rows:
        report = json.loads((campaign / "scenes" / row["scene_id"] / "render_complete.json").read_text())
        if (report["scene_id"] != row["scene_id"] or report["artifact_uuid"] != row["uuid"]
                or report["hf_revision"] != row["hf_revision"]):
            raise ValueError("Pilot does not match the pinned cohort")
        if report["frames"] < 140 or report["cadence_us"] != 100000 or report["driver_calls"] != 0:
            raise ValueError("Pilot must contain a complete GT-only 10 Hz stream")
        if report["dimensions_wh"][1] != 1080 or not 1880 <= report["dimensions_wh"][0] <= 1930:
            raise ValueError("Pilot camera must match the evaluation resolution")
        if report["gpu_samples"] < 1 or report["peak_gpu_memory_mib"] <= 0:
            raise ValueError("Pilot must contain valid GPU measurements")
        if report["peak_gpu_memory_mib"] > 32768:
            raise ValueError("Pilot exceeds 32 GiB; reconsider GPU pool before scaling")
        reports.append(report)
    result = {"status": "passed", "pilot_scene_count": len(reports),
              "cohort_sha256": hashlib.sha256((campaign / "cohort.csv").read_bytes()).hexdigest(),
              "max_peak_gpu_memory_mib": max(r["peak_gpu_memory_mib"] for r in reports),
              "gpu_names": sorted({r["gpu"]["name"] for r in reports}),
              "pilot_frames": sum(r["frames"] for r in reports),
              "provisioned_minimum_vram_gb": 40,
              "note": "Observed peaks are pilot measurements, not proof of an exact minimum for all 900 scenes."}
    temporary = campaign / "pilot_validated.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(campaign / "pilot_validated.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.campaign.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
