# Closed-loop online monitoring results (Experiment B)

`benchmark_rollouts_next100.online_monitoring.{json,csv}` are the outputs of
the `rollout_monitoring` tool (see `alpasim/src/tools/rollout_monitoring/`)
over 79 closed-loop AlpaSim rollouts (`tutorial_alpamayo_next100_unique`):
per-rollout leading-indicator results plus the aggregate summary (event
recall, lead-time distribution, per-flag precision, AUROC of the
inconsistency rate, and the offline at-flag intervention upper bound).

Provenance: these files were produced in the full development workspace with
the rule monitor's *family* match mode (Alpamayo-R1 family-grouped labels) and
two predictors (`flag`, `contradiction`). The monitor core shipped in this
artifact is exact-label only (the configuration evaluated in the paper), and
the underlying rollouts (`rollout.asl` + `metrics.parquet`) are too large to
ship, so these results are provided as-is for reference and cannot be
recomputed from the artifact alone. The tool itself runs on any directory of
AlpaSim rollouts.
