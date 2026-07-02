# Drive the Thoughts Artifact

This repository accompanies the paper "Drive the Thoughts: Runtime Monitoring
of VLA Reasoning-Trajectory Consistency".

It supports two reproduction paths:

1. Use the lightweight, data-first path to recompute checks, tables, and plots
   from the released benchmark and JSON monitor outputs.
2. Use the pinned AlpaSim source under `alpasim/` if you want to inspect the
   simulator setup or attempt full simulator reruns.

Use Apptainer for AlpaSim. The README intentionally avoids documenting
alternate simulator setup paths so the supported workflow stays focused.

## Layout

- `data/benchmark/benchmark.json`: canonical benchmark with 150 entries, 100
  of them in the reliable subset, using artifact-relative media paths.
- `data/results/llm_matrix/`: raw LLM judge outputs over the reliable subset.
- `data/results/rule/benchmark.rule_consistency.json`: deterministic rule
  monitor output (produced with `--match-mode exact`).
- `data/media/videos/`, `data/media/scenes/`: rollout videos and per-clip
  scene directories (`metadata.json`, `trajectory_plot_geometry.json`,
  `additional_info.json`, camera frame).
- `src/`: installable packages for the monitor implementation:
  - `alpasim_utils/`: shared monitor core — rule-based CoT parsing and
    consistency matching, trajectory features, lane projection, and the
    OpenAI-compatible LLM judge client (`alpasim_utils.cot_consistency`).
  - `cot_analysis/`: the LLM-judge pipeline (`python -m cot_analysis`) and the
    deterministic rule-based checker
    (`python -m cot_analysis.consistency_check`).
  - `trajectory_safety/`: geometric safety outcomes for planned trajectories
    (`python -m trajectory_safety`, Experiment A).
  - `benchmark_analysis/`: shared analysis library used by every script in
    `tools/` for benchmark loading, entry schema accessors, judgment
    normalization, and classification metrics.
- `tools/`: thin analysis/plotting CLIs plus serving and Slurm helpers.
- `tests/`: unit tests (`uv run pytest`).
- `alpasim/`: pinned AlpaSim workspace snapshot with its own `pyproject.toml`
  and `uv.lock`. We keep it frozen for provenance and intentionally omit
  generated rollouts, caches, model/data artifacts, paper source, secrets, and
  obsolete benchmark variants.

## Lightweight Setup

For the lightweight path, run `uv` from the artifact root:

```bash
uv sync --frozen
```

This installs the small dependency set and the `src/` packages. After that,
`uv run ...` works for every command below without PYTHONPATH changes.

Optional extras: use `uv sync --extra llm` to add the `openai` client for LLM
judge reruns, and `uv sync --extra xlsx` to add `openpyxl` for spreadsheet
output.

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

Regenerate the main-evaluation figure and the reliability heatmap. The default
inputs point at the released data:

```bash
uv run python tools/plot_repeated_main_eval.py --output repeated_main_eval.pdf
uv run python tools/plot_cot_reliability_heatmap.py --output heatmap.html
```

Run the packaged unit tests:

```bash
uv run pytest
```

## Recomputing Monitor Outputs

You can rerun the deterministic rule monitor fully offline. It reproduces
`data/results/rule/benchmark.rule_consistency.json` exactly:

```bash
uv run python -m cot_analysis.consistency_check \
  --benchmark_json data/benchmark/benchmark.json \
  --match-mode exact \
  --output benchmark.rule_consistency.json
```

The LLM judge needs API access, or a local vLLM server for Qwen through
`tools/serve_qwen_vllm.sh`. You can run the command below without keys to
verify the trajectory feature pipeline:

```bash
uv run python -m cot_analysis \
  --benchmark_json data/benchmark/benchmark.json \
  --prompt center_of_lane_v5 --trajectory_frame dual \
  --lane_reference map_graph --skip-unreliable-cot \
  --output cot_dry.json
```

`tools/run_benchmark_llm_matrix.py` runs the full provider/variant matrix and
writes repeated runs, manifests, and logs to `runs/llm_matrix/`.
`tools/run_benchmark_llm_matrix_update.slurm.sh` wraps the same workflow for
Slurm clusters.

## AlpaSim Setup

The full simulator tier keeps AlpaSim isolated in `alpasim/`, so its `uv`
workspace and lockfile do not interfere with the lightweight artifact tools:

```bash
cd alpasim
source setup_local_env.sh
uv sync --frozen
```

Run simulator commands with `uv run ...` from inside `alpasim/`. This source
snapshot preserves the implementation context and Apptainer setup, but it does
not bundle the full upstream data/model store. We omit large generated data and
rollouts; the benchmark media needed for the JSON-output tier lives in the
artifact root under `data/media/`.

Note that `alpasim/` predates the cleanup of this artifact's own `src/` and
`tools/`, so its monitor-code copies still reflect the original workspace
layout.

## Validation

`ARTIFACT_REPORT.json` records warnings and errors from the original artifact
staging run. Its per-result-file checks, including result counts against the
reliable subset, still describe the shipped `data/` contents.
