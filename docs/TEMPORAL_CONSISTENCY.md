# Temporal consistency: classification at a labeled frame

This experiment compares a single-frame verdict with two consecutive flags,
three consecutive flags, 2-of-3 voting, 3-of-5 voting, and weighted decay.
Every window ends at
the labeled target; its historical frames contribute monitor verdicts only.
This evaluates target-frame classification, not onset detection or intervention
outcomes.

Run the following commands from the artifact repository root. `uv run python`
can be replaced with `.venv/bin/python` in an existing environment.

## Prepare the inputs

```bash
uv run python tools/prepare_temporal_consistency.py \
  --source-benchmark <rollout-workspace>/benchmark.json \
  --source-root <rollout-workspace>/tutorial_alpamayo \
  --output-dir runs/temporal_consistency
```

The default target benchmark is `data/benchmark/benchmark.json`; only explicitly
reliable-CoT targets are selected. The source benchmark supplies original
extracted-frame locations. Relative source paths are resolved against its parent
and any additional `--source-root` directories, in that order. Original target
timestamps, CoTs, and trajectories must match the released metadata.

The preparer reads the target and four immediately preceding `_tNNNN` indices
from the same extracted scene directory. All five must contain a driver request,
CoT, and trajectory, with consecutive timestamps spaced by `--step-us` (default
100000). It excludes incomplete windows instead of skipping missing frames or
extending history across gaps. Five frames at this cadence span 0.4 seconds.

Outputs:

- `windows.json`: target labels, scene IDs, ordered frame IDs/timestamps,
  preparation settings, and exclusions with reasons.
- `frames.json`: deduplicated monitor inputs with absolute metadata paths.
  Historical frames receive no inherited human labels or reliability annotations.

For a lane-relative F-LLM run, add `--require-lane-geometry` during preparation
to require usable lane polylines at every frame. Use a separate output directory
if keeping both preparations. These files reference the original data; they do
not copy media or modify the source directories.

## Score and evaluate

The existing rule runner uses exact-label matching and cached trajectory
meta-actions, just as in the released benchmark evaluation:

```bash
uv run python -m cot_analysis.consistency_check \
  --benchmark_json runs/temporal_consistency/frames.json \
  --output runs/temporal_consistency/rule_predictions.json

uv run python tools/evaluate_temporal_consistency.py \
  --windows runs/temporal_consistency/windows.json \
  --predictions runs/temporal_consistency/rule_predictions.json \
  --consistency-type rule_based \
  --output runs/temporal_consistency/rule_evaluation.json
```

For F-LLM, use the existing judge runner on the prepared `frames.json`, keeping
the provider/model, prompt, features, and decoding settings fixed across frames:

```bash
uv run --extra llm python -m cot_analysis \
  --benchmark_json runs/temporal_consistency/frames.json \
  --provider openai --variant center_of_lane \
  --output runs/temporal_consistency/llm_predictions.json

uv run python tools/evaluate_temporal_consistency.py \
  --windows runs/temporal_consistency/windows.json \
  --predictions runs/temporal_consistency/llm_predictions.json \
  --output runs/temporal_consistency/llm_evaluation.json
```

The judge command makes model requests when credentials are configured. The
evaluator itself makes none. It uses the existing verdict parser: explicit
verdicts take precedence; otherwise LLM scores at or below 2 mean inconsistent
(`--score-threshold` changes this). Evaluate repeated judge runs separately;
duplicate frame IDs are rejected rather than pooled as temporal observations.

Weighted decay uses all five verdicts. The current flag has weight 1 and each
older flag has weight `2^(-age/half_life_frames)`. The policy alarms when the
sum of flagged weights divided by the sum of all weights reaches the threshold.
Defaults are `--decay-half-life-frames 2` (0.2 seconds with the default cadence)
and `--decay-threshold 0.5`. These options only change aggregation; they do not
rerun the per-frame monitor.

## Read the results

Every policy, including the single-frame baseline, uses the same targets with
five usable verdicts. Missing predictions, errors, dry runs, invalid rule parses,
nonfinite scores, and mismatched prediction timestamps exclude the whole window.
The report retains both preparation and prediction exclusions, including target
labels, so class-dependent attrition can be inspected. Do not compare a reduced
cohort's temporal accuracy with a baseline measured on all 100 targets.

The output JSON records per-target flags and policy decisions, class counts,
confusion counts, accuracy, balanced accuracy, and inconsistent-class
precision/recall/F1. Its sibling CSV contains the summary table. Undefined
precision/recall and balanced accuracy with an absent class are `null`.

Differences in accuracy and balanced accuracy against the single-frame baseline
include paired 95% percentile bootstrap intervals (2000 resamples, seed 0;
configurable with `--bootstrap` and `--seed`). Whole scenes are sampled together
and the same draw is used for every policy. Balanced-accuracy intervals omit
resamples lacking either class and report the number of valid resamples.
These are exploratory, individual intervals, without multiple-comparison
adjustment. The script does not select an optimal policy.

The JSON also records input file paths and SHA-256 hashes. Preserve the manifest
and prediction file with the results. Compare monitors on an explicitly shared
cohort if their failures or input requirements produce different exclusions.

## Interactive HTML comparison

```bash
uv run python tools/build_temporal_comparison.py \
  --report 'Kimi K2.5=runs/temporal_consistency_kimi/kimi_evaluation.json' \
  --report 'Rule monitor=runs/temporal_consistency_kimi/rule_evaluation_same_cohort.json' \
  --output runs/temporal_consistency_kimi/comparison.html
```

Open the HTML directly in a browser. It embeds all data and figures, works
offline, and includes monitor/policy selectors, metric charts with paired
intervals where available, a performance table, and searchable target histories.
Charts download as SVG and the summary table as CSV. Each `--report` is labeled
with `LABEL=PATH`; multiple reports must contain identical target cohorts.
