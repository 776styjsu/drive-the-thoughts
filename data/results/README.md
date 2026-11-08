# Released results

Per-run outputs behind every table in the paper. To recompute all of them from
these files and the labels in `data/benchmark/`, with no simulator, GPU or API key:

```bash
uv run python tools/reproduce_paper_tables.py            # all tables
uv run python tools/reproduce_paper_tables.py safety     # one table
```

`tests/test_reproduce_paper_tables.py` pins the reported counts.

## Paper names and file names

| Paper | Files and code |
|---|---|
| Rule-based monitor | `rule`, `python -m cot_analysis.consistency_check` |
| Ego-frame monitor | variant `llm` |
| Lane-aware monitor | variant `f_llm_map_graph` in file names; `--variant center_of_lane` on the command line |
| Qwen3.5-4B | `qwen` (local vLLM, `RedHatAI/Qwen3.5-4B-FP8-dynamic`) |
| DeepSeek V4.1 Flash | `deepseek` (OpenRouter, pinned to DeepSeek) |
| Opus 5.5 | `opus` (OpenRouter Batch API, `tools/run_openrouter_batch.py`) |
| Jev 1.14 | `jev` (OpenRouter decisions API, `tools/run_jev_decisions.py`) |
| Time Headway | `time_headway`, `baselines/time_headway/` |
| SelfOracle SAE / VAE | `selforacle`, `baselines/selforacle/` |
| Challenge set | `challenge_set`, labels in `data/benchmark/ood_hazard_2604_unsafe_traces_adjudicated.json` |

Judge files are named `<judge>.<variant>.run_00<k>.json`, with one file per
repeat (k = 1, 2, 3). A pair is flagged INCONSISTENT when its 1-5 score is at
most 2. Judge-based rows use the repeat marked `table1_run` in
`llm_matrix/manifest.json`, which is the median-F1 repeat on the reliable subset.

## Tables

| Paper table | `reproduce_paper_tables.py` | Files |
|---|---|---|
| Inconsistency detection (`tab:main-results`) | `main` | `llm_matrix/`, `rule/` |
| Consistency vs. safety (`tab:cons-safety`) | `cons-safety` | labels only |
| Unsafe-trajectory detection (`tab:safety-baselines`) | `safety` | `llm_matrix/`, `rule/`, `selforacle/`, `time_headway/`, `challenge_set/` |
| Latency (`tab:latency`) | `latency` | `llm_matrix/` (`llm_timing`), `latency/rule_latency.json` |
| Score threshold (`tab:threshold-sweep`) | `threshold` | `llm_matrix/` |
| Temporal aggregation (`tab:temporal-agg`) | `temporal` | `temporal/`, `llm_matrix/qwen.f_llm_map_graph.run_00<k>.json` |

## Folders

- `llm_matrix/`: the four judges with both monitors on all 150 benchmark
  pairs, 24 files. `manifest.json` lists each file's judge, monitor, repeat,
  and its confusion counts on the reliable subset.
- `rule/`: rule-based monitor on the 150 pairs. It is deterministic, so there
  is one run.
- `challenge_set/`: every monitor on the 15 challenge-set steps. Judge files
  follow the `llm_matrix/` naming. `rule.run_001.json` holds the rule-based
  monitor, and `selforacle/<model>_seed<k>/` holds SelfOracle per target.
- `selforacle/`: SelfOracle SAE and VAE on the 150 pairs, three training
  seeds each. Targets are scored on the five-frame mean anomaly score at
  epsilon 0.05; `per_target.csv` holds the flags. `aggregate.json` and
  `comparison.csv` cover every cohort and policy the scorer reports.
- `time_headway/`: per-target results and summary of the Time Headway
  baseline. See [its README](../../baselines/time_headway/README.md).
- `temporal/`: `windows.json` lists the five-step windows (92 targets).
  `qwen.eval92.run_00<k>.json` holds each step's lane-aware Qwen3.5-4B verdict
  for repeat k. The table takes the target-step verdict from the Table 1 run
  with the same k, so its single-step row is Table 1 restricted to these
  targets.
- `latency/rule_latency.json`: timing of the rule-based monitor
  (`tools/time_rule_monitor.py`). Judge latency comes from the `llm_timing`
  field of each judge result.
- `rollout_monitoring/`: closed-loop online monitoring (Experiment B). See
  [its README](rollout_monitoring/README.md).

## Provenance notes

- Paths are relative to the repository root. Strings that start with `runs/`
  name the authors' run tree, which is not released. They are kept as
  provenance only.
- The challenge-set runs read `runs/ood_hazard_2604_20261101/ood15.json`. That
  file is the list form of the adjudicated challenge set, with
  `cot_reliability.reliable = true` added; the released JSON now carries that
  field itself. Their scene and video paths point to the copies in
  `data/media/`. A dry run from those copies rebuilds the same judge input
  (`judge_input_fingerprint`) for all 15 steps.
- Some files are not released. These are the SelfOracle checkpoints and
  frame windows, the raw outputs of the temporal runs (about 20 MB per
  repeat; the verdicts are in `temporal/`), the Opus batch job records, and
  run logs.
- A few `input_sha256` fields refer to input files as they were before their
  paths were made relative.
