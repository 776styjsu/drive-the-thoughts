"""A failed memory or provenance check must never release the full GPU array."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pilot_gate", Path(__file__).parents[1] / "tools/validate_render_pilot.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("failure", [None, "memory", "revision"])
def test_gate_releases_only_valid_pilots(tmp_path, failure):
    rows = [{"scene_id": f"scene-{i}", "uuid": f"asset-{i}", "hf_revision": "pinned"} for i in range(2)]
    with (tmp_path / "cohort.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        report = {"scene_id": row["scene_id"], "artifact_uuid": row["uuid"], "hf_revision": row["hf_revision"],
                  "frames": 180, "cadence_us": 100000, "driver_calls": 0,
                  "gpu_samples": 10, "dimensions_wh": [1916, 1080],
                  "peak_gpu_memory_mib": 34000 if failure == "memory" else 8000,
                  "gpu": {"name": "NVIDIA A40"}}
        if failure == "revision":
            report["hf_revision"] = "another-reconstruction"
        folder = tmp_path / "scenes" / row["scene_id"]
        folder.mkdir(parents=True)
        (folder / "render_complete.json").write_text(json.dumps(report))
    if failure:
        with pytest.raises(ValueError):
            gate.validate(tmp_path)
        assert not (tmp_path / "pilot_validated.json").exists()
    else:
        assert gate.validate(tmp_path)["pilot_frames"] == 360
        assert json.loads((tmp_path / "pilot_validated.json").read_text())["status"] == "passed"
