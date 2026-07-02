# Drive the Thoughts Artifact

This directory is a staged artifact for the paper "Drive the Thoughts:
Runtime Monitoring of VLA Reasoning-Trajectory Consistency".

The artifact has two reproduction tiers:

1. A lightweight data-first tier that recomputes checks, tables, and plots from
   the released benchmark plus JSON monitor outputs.
2. A pinned AlpaSim source tier under `alpasim/` for reviewers who want to
   inspect or attempt full simulator reruns.

Apptainer is the tested environment path for AlpaSim. Docker/local Python can
be attempted on a best-effort basis, but is not guaranteed.

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
- `pyproject.toml`: lightweight `uv` environment for the data-first tier.
- `requirements-artifact.txt`: plain `venv`/`pip` fallback dependencies.
- `environment-artifact.yml`: optional Conda environment for the same
  lightweight tier.
- `alpasim/`: pinned AlpaSim workspace snapshot with its own `pyproject.toml`
  and `uv.lock`. Generated rollouts, caches, model/data artifacts, paper source,
  secrets, and obsolete benchmark variants are intentionally not included.

## Current Staging Counts

- Benchmark entries: 150
- Reliable benchmark entries: 100
- Benchmark entries with videos: 149
- Copied videos: 149
- Copied scene directories: 150
- Copied result JSON files: 15
- Copied AlpaSim source-tier files: 483

These counts are generated from the current files at staging time. If the paper
tables or benchmark are updated, rerun `tools/build_artifact_release.py` and
check `ARTIFACT_REPORT.json`.

## Lightweight Setup

The recommended lightweight path uses `uv` from the artifact root:

```bash
uv sync --frozen
```

If `uv.lock` is not present because the artifact was rebuilt locally without
locking, use `uv sync` once to resolve the small dependency set.

Plain Python fallback:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-artifact.txt
export PYTHONPATH="$PWD/src/utils:$PWD/src/tools${PYTHONPATH:+:$PYTHONPATH}"
```

Optional Conda fallback:

```bash
conda env create -f environment-artifact.yml
conda activate drive-the-thoughts-artifact
export PYTHONPATH="$PWD/src/utils:$PWD/src/tools${PYTHONPATH:+:$PYTHONPATH}"
```

## Quick Checks

From the artifact root:

```bash
uv run python tools/check_consistency_accuracy.py \
  --consistency-file data/results/llm_matrix/gpt.f_llm_map_graph.run_003.json \
  --benchmark-file data/benchmark/benchmark.json \
  --reliable-only
```

For plotting:

```bash
uv run python tools/plot_repeated_main_eval.py \
  --run-dir data/results/llm_matrix \
  --benchmark data/benchmark/benchmark.json \
  --rule-result data/results/rule/benchmark.rule_consistency.json \
  --output repeated_main_eval.pdf
```

Run the packaged unit tests:

```bash
uv run pytest
```

## AlpaSim Setup

The full simulator tier keeps AlpaSim isolated in `alpasim/` so its `uv`
workspace and lockfile do not interfere with the lightweight artifact tools:

```bash
cd alpasim
source setup_local_env.sh
uv sync --frozen
```

Use `uv run ...` from inside `alpasim/` for simulator commands. This source
snapshot is intended to preserve the implementation context and Apptainer setup,
not to bundle the full upstream data/model store. Large generated data and
rollouts are omitted; benchmark media needed for the JSON-output tier lives in
the artifact root under `data/media/`.

## Validation

`ARTIFACT_REPORT.json` records warnings/errors from the staging run. Treat the
artifact as release-ready only when `ok` is `true`.
