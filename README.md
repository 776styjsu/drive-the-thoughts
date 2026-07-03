# Drive the Thoughts Artifact

This repository implements the paper "Drive the Thoughts: Runtime Monitoring
of VLA Reasoning-Trajectory Consistency".

There are two ways to use it:

1. **Data-first (lightweight).** Recompute checks, tables, and plots from the
   released benchmark and monitor outputs. Needs only `uv` and a few small
   dependencies.
2. **Simulator (full).** Inspect or rerun the pinned AlpaSim workspace under
   `alpasim/`, including the online consistency monitor.

## Layout

- `data/benchmark/benchmark.json` — the benchmark: 150 entries, 100 in the
  reliable subset, with artifact-relative media paths.
- `data/results/llm_matrix/` — raw LLM judge outputs over the reliable subset.
- `data/results/rule/benchmark.rule_consistency.json` — deterministic
  exact-label rule monitor output.
- `data/media/` — rollout videos and per-clip scene directories (metadata,
  trajectory geometry, `additional_info.json`, camera frame).
- `src/` — the monitor implementation, installed as packages:
  - `alpasim_utils` — shared monitor core: rule-based CoT parsing and
    matching, trajectory features, lane projection, and the OpenAI-compatible
    judge client.
  - `cot_analysis` — the LLM-judge pipeline (`python -m cot_analysis`) and
    the rule-based checker (`python -m cot_analysis.consistency_check`).
  - `trajectory_safety` — geometric safety outcomes (Experiment A).
  - `benchmark_analysis` — benchmark loading, schema accessors, and metrics,
    used by every script in `tools/`.
- `tools/` — analysis and plotting CLIs, the vLLM serve script, and Slurm
  helpers.
- `tests/` — unit tests (`uv run pytest`).
- `alpasim/` — pinned AlpaSim workspace with its own `pyproject.toml` and
  `uv.lock`. It ships the simulator plus exactly the monitor code it runs
  online (the `ConsistencyMonitor` and the eval scorer). Offline analysis
  lives only at the artifact root. Large generated data, models, and rollouts
  are omitted.

One rule when editing monitor code: `src/alpasim_utils` is the source of
truth, and `alpasim/` carries a byte-identical mirror so the simulator
workspace stays standalone. Edit at the root, then run
`uv run python tools/sync_monitor_core.py --fix`. `pytest` fails if the two
trees drift.

## Setup

```bash
uv sync --frozen
```

That is the whole setup for the data-first path; every command below then
works via `uv run ...` with no PYTHONPATH tweaks.

Two optional extras: `--extra llm` adds the `openai` client for LLM judge
reruns (or prefix one-off commands with `uv run --extra llm`), and
`--extra xlsx` adds `openpyxl` for spreadsheet output.

## Quick Checks

```bash
# F-LLM monitor vs. human labels on the reliable subset
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

# Main-evaluation figure and reliability heatmap (defaults use released data)
uv run python tools/plot_repeated_main_eval.py --output repeated_main_eval.pdf
uv run python tools/plot_cot_reliability_heatmap.py --output heatmap.html

# Unit tests
uv run pytest
```

## Recomputing Monitor Outputs

The rule monitor runs fully offline and reproduces the released file exactly:

```bash
uv run python -m cot_analysis.consistency_check \
  --benchmark_json data/benchmark/benchmark.json \
  --output benchmark.rule_consistency.json
```

The LLM judge needs an OpenAI-compatible backend:

- `--provider kimi` — Moonshot's public API (key: `MOONSHOT_API_KEY`)
- `--provider openai` — OpenAI (key: `OPENAI_API_KEY`)
- `--provider qwen35_4b_fp8` — your own local vLLM server (no key needed)

Pass keys with `--api_key`, the per-provider `--*-api-key` flags on the matrix
runner, the environment, or a `.env` file. Any other OpenAI-compatible host
works through `--base_url` and `--model` — no code changes. Without a key the
judge runs dry and still verifies the trajectory feature pipeline:

```bash
uv run python -m cot_analysis \
  --benchmark_json data/benchmark/benchmark.json \
  --variant center_of_lane --skip-unreliable-cot \
  --output cot_dry.json
```

### Example: the Qwen matrix, end to end

`tools/run_benchmark_llm_matrix.py` runs the provider/variant matrix and
writes repeated runs, logs, and a manifest to `runs/llm_matrix/`. For the
local Qwen judge you need a GPU, a running vLLM server, and then the runner —
in that order.

```bash
# 1. Get a GPU node (Slurm example — we used one A100 80GB; adjust the
#    account/partition/constraint to your site).
salloc -A your_hpc_account -p gpu --gres=gpu:1 -c 8 --mem=64G -t 4:00:00

# 2. Serve Qwen3.5-4B-FP8 behind an OpenAI-compatible API on localhost:8000.
tools/serve_qwen_vllm.sh qwen35 setup    # first time only: installs vLLM
tools/serve_qwen_vllm.sh qwen35 serve    # keep running (own terminal or &)
curl -sf http://localhost:8000/v1/models # ready once this responds

# 3. Run the matrix for the qwen provider (3 repeats over the reliable subset).
uv run --extra llm python tools/run_benchmark_llm_matrix.py --providers qwen
```

The runner points `qwen` at `http://localhost:8000/v1` with an `EMPTY` key by
default, so no credentials are involved. `tools/run_benchmark_llm_matrix_update.slurm.sh`
wraps the same workflow (including starting the server) as a batch job.

## AlpaSim Setup

The full simulator tier keeps AlpaSim isolated in `alpasim/`, so its `uv`
workspace and lockfile do not interfere with the lightweight artifact tools:

```bash
cd alpasim
source setup_local_env.sh
uv sync --frozen
```

Run simulator commands with `uv run ...` from inside `alpasim/`. The snapshot
preserves the implementation and Apptainer setup but not the upstream
data/model store; the benchmark media lives at the artifact root under
`data/media/`.

The snapshot has no copies of the offline analysis packages or plotting
tools — those live only at the artifact root. Simulator-side monitoring is
self-contained: the online `ConsistencyMonitor`, the eval scorer, and the
mirrored `alpasim_utils` monitor core they import.

## Validation

`ARTIFACT_REPORT.json` records warnings and errors from the original artifact
staging run. Its per-result-file checks, including result counts against the
reliable subset, still describe the shipped `data/` contents.
