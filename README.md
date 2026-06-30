# Drive the Thoughts Artifact

This directory is a staged artifact for the paper "Drive the Thoughts:
Runtime Monitoring of VLA Reasoning-Trajectory Consistency".

The current supported reproduction path is data-first: recompute tables and
sanity checks from the released benchmark and JSON monitor outputs. Full
AlpaSim/Alpamayo reruns are intended as a later tier. Apptainer is the tested
environment path; Docker/local Python can be attempted on a best-effort basis
but is not guaranteed.

## Contents

- `data/benchmark/benchmark.json`: canonical benchmark with artifact-relative
  media paths.
- `data/results/llm_matrix/`: raw LLM judge outputs and the reliable subset
  derived from the current benchmark.
- `data/results/rule/benchmark.rule_consistency.json`: deterministic rule
  monitor output.
- `data/media/videos/`: rollout videos copied from benchmark references.
- `data/media/scenes/`: per-clip scene metadata and supporting files when
  available.
- `tools/` and `src/`: scripts and monitor implementation used for the
  JSON-output reproduction tier.

## Current Staging Counts

- Benchmark entries: 150
- Reliable benchmark entries: 100
- Benchmark entries with videos: 149
- Copied videos: 149
- Copied scene directories: 150
- Copied result JSON files: 15

These counts are generated from the current files at staging time. If the paper
tables or benchmark are updated, rerun `tools/build_artifact_release.py` and
check `ARTIFACT_REPORT.json`.

## Quick Checks

From the artifact root:

```bash
python tools/check_consistency_accuracy.py \
  --consistency-file data/results/llm_matrix/gpt.f_llm_map_graph.run_003.json \
  --benchmark-file data/benchmark/benchmark.json \
  --reliable-only
```

For plotting scripts, use a Python environment with `matplotlib` installed.
The project-developed execution path uses `uv run`; local/Docker execution is
best-effort and may require installing dependencies manually.

## Validation

`ARTIFACT_REPORT.json` records warnings/errors from the staging run. Treat the
artifact as release-ready only when `ok` is `true`.
