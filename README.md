# Drive the Thoughts Artifact

This repository is the artifact for the paper "Drive the Thoughts: Runtime
Monitoring of VLA Reasoning-Trajectory Consistency".

The artifact has two reproduction tiers:

1. A lightweight data-first tier that recomputes checks, tables, and plots from
   the released benchmark plus JSON monitor outputs.
2. A pinned AlpaSim source tier under `alpasim/` for reviewers who want to
   inspect or attempt full simulator reruns.

Apptainer is the tested environment path for AlpaSim. Docker/local Python can
be attempted on a best-effort basis, but is not guaranteed.

## Layout

- `data/benchmark/benchmark.json`: canonical benchmark (150 entries, 100 in
  the reliable subset) with artifact-relative media paths.
- `data/results/llm_matrix/`: raw LLM judge outputs over the reliable subset.
- `data/results/rule/benchmark.rule_consistency.json`: deterministic rule
  monitor output (produced with `--match-mode exact`).
- `data/media/videos/`, `data/media/scenes/`: rollout videos and per-clip
  scene directories (`metadata.json`, `trajectory_plot_geometry.json`,
  `additional_info.json`, camera frame).
- `src/`: the monitor implementation as installable packages:
  - `alpasim_utils/`: shared monitor core — rule-based CoT parsing and
    consistency matching, trajectory features, lane projection, and the
    OpenAI-compatible LLM judge client (`alpasim_utils.cot_consistency`).
  - `cot_analysis/`: the LLM-judge pipeline (`python -m cot_analysis`) and the
    deterministic rule-based checker
    (`python -m cot_analysis.consistency_check`).
  - `trajectory_safety/`: geometric safety outcomes for planned trajectories
    (`python -m trajectory_safety`, Experiment A).
  - `benchmark_analysis/`: shared analysis library (benchmark loading, entry
    schema accessors, judgment normalization, classification metrics) used by
    every script in `tools/`.
- `tools/`: thin analysis/plotting CLIs plus serving and Slurm helpers.
- `tests/`: unit tests (`uv run pytest`).
- `alpasim/`: pinned AlpaSim workspace snapshot with its own `pyproject.toml`
  and `uv.lock`, kept frozen for provenance. Generated rollouts, caches,
  model/data artifacts, paper source, secrets, and obsolete benchmark variants
  are intentionally not included.

## Lightweight Setup

The recommended lightweight path uses `uv` from the artifact root:

```bash
uv sync --frozen
```

This installs the small dependency set and the `src/` packages, so
`uv run ...` works for every command below without PYTHONPATH tweaks.

Plain Python fallback:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-artifact.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Optional Conda fallback:

```bash
conda env create -f environment-artifact.yml
conda activate drive-the-thoughts-artifact
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Optional extras: `uv sync --extra llm` adds the `openai` client needed to
rerun the LLM judge; `--extra xlsx` adds `openpyxl` for spreadsheet output.

## Quick Checks

From the artifact root:

```bash
# LLM judge vs. human labels on the reliable subset
uv run python tools/check_consistency_accuracy.py \
  --consistency-file data/results/llm_matrix/gpt.f_llm_map_graph.run_003.json \
  --benchmark-file data/benchmark/benchmark.json \
  --reliable-only

# Rule-based monitor vs. human labels
uv run python tools/check_consistency_accuracy.py \
  --consistency-file data/results/rule/benchmark.rule_consistency.json \
  --benchmark-file data/benchmark/benchmark.json \
  --consistency-type alpasim_cot_consistency_report \
  --reliable-only
```

Regenerate the main-evaluation figure and the reliability heatmap (defaults
point at the released data):

```bash
uv run python tools/plot_repeated_main_eval.py --output repeated_main_eval.pdf
uv run python tools/plot_cot_reliability_heatmap.py --output heatmap.html
```

Run the packaged unit tests:

```bash
uv run pytest
```

## Recomputing Monitor Outputs

The deterministic rule monitor is fully rerunnable offline and reproduces
`data/results/rule/benchmark.rule_consistency.json` exactly:

```bash
uv run python -m cot_analysis.consistency_check \
  --benchmark_json data/benchmark/benchmark.json \
  --match-mode exact \
  --output benchmark.rule_consistency.json
```

The LLM judge runs need API access (or a local vLLM server for Qwen; see
`tools/serve_qwen_vllm.sh`). A dry run without keys verifies the trajectory
feature pipeline:

```bash
uv run python -m cot_analysis \
  --benchmark_json data/benchmark/benchmark.json \
  --prompt center_of_lane_v5 --trajectory_frame dual \
  --lane_reference map_graph --skip-unreliable-cot \
  --output cot_dry.json
```

`tools/run_benchmark_llm_matrix.py` orchestrates the full provider/variant
matrix (repeated runs, manifests, logs) into `runs/llm_matrix/`;
`tools/run_benchmark_llm_matrix_update.slurm.sh` wraps it for Slurm clusters.

## AlpaSim Setup

The full simulator tier keeps AlpaSim isolated in `alpasim/` so its `uv`
workspace and lockfile do not interfere with the lightweight artifact tools:

```bash
cd alpasim
source setup_local_env.sh
uv sync --frozen
```

Use `uv run ...` from inside `alpasim/` for simulator commands. This source
snapshot is intended to preserve the implementation context and Apptainer
setup, not to bundle the full upstream data/model store. Large generated data
and rollouts are omitted; benchmark media needed for the JSON-output tier
lives in the artifact root under `data/media/`. Note that `alpasim/` predates
the cleanup of the artifact's own `src/` and `tools/`, so its copies of the
monitor code reflect the original workspace layout.

## Validation

`ARTIFACT_REPORT.json` records warnings/errors from the original staging run
of this artifact. The per-result-file checks there (result counts vs. the
reliable subset) still describe the shipped `data/` contents.
