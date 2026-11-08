import asyncio

import pytest

from selforacle_baseline.data import read_json, write_json
from selforacle_baseline.evaluate import align_to_decisions, main


def test_causal_decision_alignment_and_stale_or_warmup_exclusion():
    rows = [
        {"sequence_id": "a", "now_us": 100, "scored": True, "flag": True},
        {"sequence_id": "a", "now_us": 200, "scored": False, "flag": None},
        {"sequence_id": "a", "now_us": 400, "scored": True, "flag": False},
    ]
    result = align_to_decisions(rows, [0, 100, 150, 175, 200, 350, 400], 50)
    assert [r["scored"] for r in result] == [False, True, True, False, False, False, True]
    assert [r["flag"] for r in result] == [False, True, True, False, False, False, False]
    with pytest.raises(ValueError, match="Duplicate"):
        align_to_decisions(rows + [rows[0]], [100], 50)


def test_existing_rollout_evaluator_with_synthetic_log(tmp_path):
    pytest.importorskip("rollout_monitoring.analysis")
    import polars as pl
    from alpasim_grpc.v0.logging_pb2 import LogEntry
    from alpasim_utils.logs import LogWriter

    asl = tmp_path / "clipgt-example" / "run1" / "rollout.asl"

    async def write_log():
        async with LogWriter(str(asl)) as writer:
            for t in (100_000, 200_000, 300_000, 400_000):
                request = LogEntry()
                request.driver_request.time_now_us = t
                await writer.on_message(request)
                response = LogEntry()
                response.driver_return.SetInParent()
                await writer.on_message(response)

    asyncio.run(write_log())
    metric_rows = []
    for t in (100_000, 200_000, 300_000, 400_000):
        for name, value in (("eval_relevant", 1.0), ("collision_any", float(t == 400_000)),
                            ("collision_front", float(t == 400_000)), ("offroad", 0.0)):
            metric_rows.append({"name": name, "timestamps_us": t, "values": value})
    pl.DataFrame(metric_rows).write_parquet(asl.with_name("metrics.parquet"))
    predictions = tmp_path / "predictions.json"
    write_json(predictions, {"schema_version": 1, "monitor": "selforacle_alpasim_v1", "results": [
        {"sequence_id": "s1", "scene_id": "clipgt-example", "now_us": t,
         "scored": True, "flag": t == 200_000} for t in (100_000, 200_000, 300_000, 400_000)
    ]})
    mapping = tmp_path / "mapping.json"
    write_json(mapping, {"s1": str(asl)})
    output = tmp_path / "evaluation.json"
    main(["--predictions", str(predictions), "--rollout-map", str(mapping), "--output", str(output)])
    result = read_json(output)
    assert result["summary"]["event_recall_within_horizon"] == 1.0
    assert result["summary"]["flag_precision_within_horizon"] == 1.0
    assert result["summary"]["lead_time"]["median_s"] == 0.2
    assert result["coverage"]["s1"]["scored_decisions"] == 4
    assert "inconsistency_rate" not in result["rollouts"][0]

