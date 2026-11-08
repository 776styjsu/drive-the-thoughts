# SelfOracle baseline setup

An isolated PyTorch adaptation of SelfOracle for AlpaSim camera streams. Includes
SAE and VAE training, nominal-data Gamma calibration, causal temporal filtering,
and timestamped test predictions. This is a starting implementation, not a
reproduction of the paper's reported accuracy or a pretrained monitor.

The root artifact and AlpaSim environments do not depend on this package. Run
the commands below **from the repository root**. Outputs go under the already
ignored `runs/selforacle/` directory; this project's `.venv` is also ignored.

## Install

```bash
# CPU development and smoke testing; creates baselines/selforacle/.venv.
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --project baselines/selforacle --extra cpu --extra dev --locked

# Alternatively, on a GPU node with a driver compatible with CUDA 12.8:
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --project baselines/selforacle --extra cuda --extra dev --locked
```

The CPU and CUDA extras are mutually exclusive. The examples use CPU; replace
`--extra cpu` with `--extra cuda` and select `--device cuda` for GPU work.
Training is not automatically submitted to Slurm or started by installation.

## Prepare a dataset

Use directories containing AlpaSim's extracted `*/metadata.json` and camera
JPEGs. The extraction utility is `alpasim/tools/extract_rollout_frames.sh`;
extract consecutive frames rather than one annotated decision per scene. Do
not use the composite rollout videos or trajectory plots as camera inputs.

```bash
uv run --project baselines/selforacle --extra cpu --locked selforacle discover \
  --root <rollout-root> \
  --holdout-benchmark data/benchmark/benchmark.json \
  --output runs/selforacle/inventory.json
```

Repeat `--root` for multiple collections. Discovery inventories frame directories,
does not move images, and does not assign training data automatically. It uses
the extractor's `<scene_id>_tNNNN` folder names; preparation then checks the
metadata contents against those names. Use the
underlying rollout collection rather than a directory of scene symlinks.

Edit `inventory.json`:

- Assign each sequence `split: "train"`, `"calibration"`, `"test"`, or `"exclude"`.
- Set `nominal: true` for every training and calibration sequence after checking
  that it represents the intended nominal driving conditions. A successful
  outcome alone does not establish that a sequence is nominal.
- Keep all recordings of the same scene in one split. Benchmark scenes passed
  through `--holdout-benchmark` are reserved for testing, including alternative
  rollouts of those scenes. Add any other intended evaluation scenes to
  `reserved_test_scenes` before preparation.
- `sequence_id` identifies an extracted recording directory, not an AlpaSim
  rollout UUID. Do not combine multiple rollouts in one extraction directory.

Example sequence (paths and IDs are illustrative):

```json
{
  "scene_id": "clipgt-example",
  "sequence_id": "recording-example",
  "frames_dir": "/data/run/extracted_frames/clipgt-example",
  "split": "train",
  "nominal": true
}
```

```bash
uv run --project baselines/selforacle --extra cpu --locked selforacle prepare \
  --inventory runs/selforacle/inventory.json \
  --output runs/selforacle/manifest.json
uv run --project baselines/selforacle --extra cpu --locked selforacle validate \
  --manifest runs/selforacle/manifest.json
```

Preparation hashes every selected image and rejects missing camera frames,
duplicate images/timestamps, scene overlap across splits, unassigned sequences,
and non-nominal training/calibration assignments. It does not infer safety from
CoT consistency. Model commands verify the selected images against those hashes.
The manifest intentionally uses absolute paths; regenerate it when moving data.

### Audit a rendered GT cohort automatically

For a campaign produced by `tools/render_nominal.py`, run the CPU audit before
assigning nominal labels. It joins the existing cohort splits, excludes the full
reserved scene list, decodes and hashes every camera JPEG, checks source identity,
frame counts, dimensions and consecutive 10 Hz timestamps, and finds duplicate
JPEGs, model-input pixels, source assets and exact retained GT recordings.

```bash
baselines/selforacle/.venv/bin/python baselines/selforacle/tools/audit_training_data.py \
  --campaign runs/selforacle/gt_render_20261010 \
  --reserved-scenes runs/selforacle/cohort_recommendation_20261009/reserved_test_scenes.csv \
  --output runs/selforacle/gt_render_20261010/manifest_audit \
  --workers 8
```

Use a compute node for the complete audit. `tools/audit_training_data.slurm.sh`
requests 16 CPU cores and 32 GB RAM; export `NOMINAL_CAMPAIGN`,
`NOMINAL_RESERVED_SCENES` and `NOMINAL_AUDIT_OUTPUT` when submitting. The output
directory must be new, so reruns cannot overwrite review decisions.

Outputs include `inventory.draft.json`, `manifest.draft.json`, `validation.json`,
per-frame and per-scene audit records, `duplicate_groups.json`, `dataset_card.md`,
and `SHA256SUMS`. `review.html` provides a searchable, paginated gallery of
five-frame contact sheets; `review_decisions.csv` starts with every decision
pending. Darkness/uniformity flags are inspection aids, not automatic exclusions.
Original log identifiers are unavailable here, so partial recording overlap
between different source scene IDs cannot be fully ruled out; exact retained GT
recording fingerprints are checked.

All draft nominal labels remain null, and the training loader rejects the draft.
After visual/domain review, apply accepted decisions to a separate curated
inventory and use `selforacle prepare` to create the frozen training manifest.
The audit does not train models or change any rendered inputs.

## Train, calibrate, predict

### Full training campaign

`selforacle_baseline.full_training` freezes a 630/70/100/100 scene split from the
audited 900-scene GT replay cohort. The 70 validation scenes come only from the
original 700-scene training pool, stratified by lighting and speed with seed 0;
the original calibration and nominal-test sets are retained. Pending nominal
labels are preserved, and `--allow-pending-nominal` explicitly starts provisional
training. The standard training/scoring CLI continues to reject these manifests.
These checkpoints do not certify completion of nominal review.

```bash
baselines/selforacle/.venv/bin/python -m selforacle_baseline.full_training prepare \
  --campaign runs/selforacle/gt_render_20261010 \
  --output runs/selforacle/full_training
export FULL_TRAINING_ROOT="$PWD/runs/selforacle/full_training"
```

Submit `tools/full_training_cache.slurm.sh` first, then
`tools/full_training.slurm.sh` with an `afterok` dependency on its job ID. The
cache job checks each original JPEG and audited resized-pixel hash and creates
four row-order-bound uint8 NPY files. The array runs SAE and VAE with seeds 0/1/2,
one 24 GB MIG device and four CPU cores per run. Each run loads its training and
validation caches onto the GPU and verifies their hashes against the manifest.

The frozen protocol uses Adam 0.001, batch 32, maximum 50 epochs, patience 5,
relative improvement 0.1%, and the lowest unaugmented validation reconstruction
MSE checkpoint. Training-only augmentation has probability 0.6, applying flip,
source-resolution-scaled translation, HLS-lightness shadow and HSV-value
brightness in consistently resized RGB. This is an adaptation: augmentation is
applied after resize, whereas the release applies it before its mixed RGB/YUV
preprocessing. VAE training retains the released reconstruction-only objective.
After training, each selected checkpoint is calibrated on the 100 calibration
scenes at epsilon 0.05/0.01 and scored on the 100 nominal-test scenes. The 150-case
safety benchmark is not read or used for epoch/threshold selection. Per-model
`progress.json`, `training.json`, `model.pt`, `calibration.json` and
`nominal_test_predictions.json` record progress and results.

### GPU technical pilot before nominal review

The dedicated smoke runner can test audited inputs whose nominal review is still
pending. It selects 40 scenes from the original training pool only, stratified by
lighting and speed: 30 training, 5 development calibration and 5 development
prediction scenes. Original held-out sets and nominal decisions are unchanged.
Its string artifact schema is rejected by the research training/scoring pipeline;
these outputs must not be used as paper results.

```bash
baselines/selforacle/.venv/bin/python -m selforacle_baseline.smoke prepare \
  --campaign runs/selforacle/gt_render_20261010 \
  --output runs/selforacle/training_smoke/manifest.json

export SMOKE_MANIFEST="$PWD/runs/selforacle/training_smoke/manifest.json"
export SMOKE_OUTPUT="$PWD/runs/selforacle/training_smoke/run"
export UV_BIN="$(command -v uv)"
sbatch --output="$PWD/runs/selforacle/training_smoke/slurm-%j.out" \
  --error="$PWD/runs/selforacle/training_smoke/slurm-%j.err" \
  baselines/selforacle/tools/training_smoke.slurm.sh
```

The job requests one 24 GB MIG device, four CPU cores, 16 GB RAM and one hour.
It installs the locked CUDA backend into `runs/selforacle/envs/torch211_cu128`,
preserving the CPU development environment. It verifies raw and resized pixel
hashes and caches uint8 inputs in process, then runs SAE for 1 and 5 epochs and
VAE for 1 epoch, with batch size 32, Adam learning rate 0.001 and seed 0. It checks
finite losses/gradients/parameters, checkpoint reload equality, Gamma fits at
epsilon 0.05 and 0.01, complete predictions and five-frame temporal warmup.
`progress.json` records progress; `result.json` summarizes memory and throughput.
This pilot omits augmentation and validation checkpoint selection. Its throughput
uses cached inputs and includes additional finite-gradient checks.

```bash
uv run --project baselines/selforacle --extra cpu --locked selforacle train \
  --manifest runs/selforacle/manifest.json --model sae --epochs 20 \
  --device cpu --output runs/selforacle/sae_seed0

uv run --project baselines/selforacle --extra cpu --locked selforacle calibrate \
  --manifest runs/selforacle/manifest.json \
  --checkpoint runs/selforacle/sae_seed0/model.pt \
  --epsilon 0.05 --window 5 --max-gap-us 150000 \
  --output runs/selforacle/sae_seed0/calibration.json

uv run --project baselines/selforacle --extra cpu --locked selforacle predict \
  --manifest runs/selforacle/manifest.json \
  --checkpoint runs/selforacle/sae_seed0/model.pt \
  --calibration runs/selforacle/sae_seed0/calibration.json \
  --output runs/selforacle/sae_seed0/predictions.json
```

For the VAE, use `--model vae` and a new output directory, then calibrate that
checkpoint separately. `--epochs 1` is useful for a pipeline smoke test; 20 is
an initial training setting, not a validated optimum. Training uses fixed
epochs, with no model selection on calibration or test scenes. A seed is
recorded; numerical identity across hardware is not guaranteed.

Calibration fits `Gamma(shape, loc=0, scale)` to **raw L2 reconstruction errors**
and uses its `1-epsilon` quantile. Both raw and filtered exceedance rates are
reported. Epsilon is a nominal tail target under the fitted distribution, not
a guarantee about false alarms on unseen data. Constant/zero error distributions
are rejected instead of silently choosing a threshold.

The default filter is a trailing five-frame arithmetic mean. At a 100 ms
cadence this spans 0.4 s. It resets at sequence boundaries or gaps greater than
150 ms, and requires a full window before returning a verdict. Set the cadence
and window before evaluation; inference inherits them from calibration. For
explicitly single-frame experiments, calibrate with `--window 1`. Sparse
annotated snapshots cannot measure temporal early warning.

## Prediction contract and evaluation boundary

`predictions.json` contains `results`, one row per test camera frame:

```json
{
  "scene_id": "clipgt-example",
  "sequence_id": "recording-example",
  "clip_id": "clipgt-example_t0004",
  "now_us": 400000,
  "raw_score": 23.1,
  "score": 22.7,
  "scored": true,
  "flag": false
}
```

Higher scores mean more unusual visual inputs. These are neither probabilities
nor CoT-consistency scores. During warmup `scored=false`, `score=null`, and
`flag=null`; exclude those rows rather than counting them as safe decisions.
Checkpoint, calibration, manifest hashes, and preprocessing settings accompany
the output. `scoring_seconds` includes batched image loading and inference;
it is not a measurement of deployed per-decision latency.

Do not feed these scores into `check_consistency_accuracy.py` or reinterpret
anomaly flags as inconsistency.

## Evaluate alarms against rollout safety outcomes

For the 150 annotated benchmark decisions, use the separate
`selforacle_baseline.benchmark_safety` module. It evaluates camera anomaly flags
against `planned_trajectory_safe`, with unsafe as positive. It reconstructs each
causal five-frame camera window from the original extraction, verifies that the
target metadata and JPEG match the released benchmark, and rejects
training/calibration overlap. Historical CoT/trajectory availability is not a
requirement for this image monitor. Thresholds come unchanged from calibration.

```bash
baselines/selforacle/.venv/bin/python -m selforacle_baseline.benchmark_safety prepare \
  --source-benchmark <rollout-workspace>/benchmark.json \
  --source-root <rollout-workspace>/tutorial_alpamayo \
  --output runs/selforacle/safety150/windows.json
```

`tools/benchmark_safety.slurm.sh` evaluates the five-epoch SAE and one-epoch VAE
pilot checkpoints on one MIG device. Set `SAFETY_MANIFEST`,
`SAFETY_TRAINING_MANIFEST`, `SAFETY_PILOT_RUN`, and `SAFETY_OUTPUT` to their absolute
paths. The explicit `--allow-pilot-checkpoint` option preserves the exploratory
status; it does not bypass nominal review in the production training pipeline.
Outputs include per-target scores/alarms and summary CSV/JSON for all targets and
the predefined reliable-CoT subset, using five-frame means and a single-frame
control. Continuous-score AUROC and average precision use unsafe as positive.
This annotated safety task does not measure closed-loop failures or lead time.

For the six full-training checkpoints, use
`tools/benchmark_safety_full.slurm.sh` with `SAFETY_MANIFEST`, `SAFETY_FULL_ROOT`,
and `SAFETY_OUTPUT` set to absolute paths. It evaluates SAE/VAE seeds 0/1/2 on
one MIG device, using each selected checkpoint's unchanged calibration thresholds.
The explicit `--allow-pending-nominal` option retains pending nominal-review
status in the reports. Benchmark scenes must be disjoint from all four cohort
splits, including validation and nominal test. After the job completes, run:

```bash
baselines/selforacle/.venv/bin/python baselines/selforacle/tools/summarize_benchmark_safety.py \
  --campaign runs/selforacle/safety150_full_20261011
```

The summary tool independently checks labels, confusion counts, pairwise AUROC,
average precision, frozen artifact hashes and thresholds. It writes per-seed
`comparison.csv`, `aggregate.csv`/JSON, `verification.json` and `SHA256SUMS`.
Aggregates report the mean and sample standard deviation across three training
seeds on the same targets; they are not confidence intervals or additional test
samples. Undefined precision retains its missing status and a count of defined
seeds accompanies each mean.

The `selforacle_baseline.evaluate` adapter uses the existing
`alpasim/src/tools/rollout_monitoring/analysis.py` evaluator. Run it in the
**AlpaSim environment**, after setting up that workspace as described in the
root README; the training environment does not need simulator dependencies.

Create `runs/selforacle/rollout_map.json`, mapping every prediction sequence ID
to the exact recording from which its camera frames were extracted:

```json
{
  "recording-example": "/data/scene_example/rollouts/clipgt-example/rollout-uuid/rollout.asl"
}
```

`metrics.parquet` must be alongside each log. Mapping paths may alternatively
be relative to the mapping file. Verify the source recording UUID yourself:
scene matching alone cannot distinguish alternative rollouts of the same scene.

```bash
PYTHONPATH="$PWD/baselines/selforacle/src:$PWD/alpasim/src/tools" \
  alpasim/.venv/bin/python -m selforacle_baseline.evaluate \
  --predictions runs/selforacle/sae_seed0/predictions.json \
  --rollout-map runs/selforacle/rollout_map.json \
  --outcome offroad_or_collision_at_fault --horizon-s 4 --max-age-us 150000 \
  --output runs/selforacle/sae_seed0/safety_evaluation.json
```

The adapter reads driver request/return timestamps, selects the most recent
camera verdict **at or before** each decision, and excludes unscored or stale
observations. It rejects duplicate decision timestamps, mismatched scenes,
duplicate rollout mappings, and incomplete mappings. The evaluator retains its
existing valid-window and black-render rules; rollouts with no scored decisions
are explicitly excluded. Reports include recall, flag precision, lead time,
rollout false-alarm rate, exclusions, and eligible decision timestamps. The
inherited rate-based AUROC is labeled as an anomaly-flag-rate metric, not as
CoT inconsistency. No intervention-effectiveness claim is produced.

Use the same eligible decision timestamps and rollouts for F-LLM; the adapter
does not automatically rerun F-LLM or enforce a shared cohort with its results.
This setup does not perform online intervention or automatically curate a nominal
training corpus. Before reporting baseline results, validate the port and
preprocessing on reference data, establish scene splits and nominal coverage,
and select settings without inspecting test outcomes.

## Relation to the original implementation

Reference: [Stocco et al., ICSE 2020](https://doi.org/10.1145/3377811.3380353),
[public code at commit 83871508](https://github.com/testingautomated-usi/selforacle/tree/83871508ad691bbff1ce32838abc22f27319aee8).
The upstream MIT notice is retained in `THIRD_PARTY_NOTICES`.

The architectures follow the release: SAE has a 64-unit ReLU hidden layer;
VAE has 512-unit encoder/decoder layers and a 2-dimensional stochastic latent.
Inputs default to 80 x 160 x 3. The release trains with MSE and scores with
Euclidean reconstruction distance; its VAE implementation does not add a KL
term. This adaptation follows that released objective, including the SAE's
1e-8 activity L1 coefficient. It must not be described as a standard KL-trained
VAE or an exact paper reproduction.

Intentional adaptation choices: PyTorch, consistent RGB resize/normalization,
no augmentation in this initial setup, deterministic posterior-mean VAE
scoring, held-out nominal calibration, explicit temporal warmup/gap handling,
and complete batches including the final partial batch. The upstream generator
applies random augmentation and mixed preprocessing, and its VAE samples at
inference. Our choices are recorded/documented so performance can be assessed
as **SelfOracle adapted to AlpaSim**, rather than attributed to upstream unchanged.

## Checks

```bash
uv run --project baselines/selforacle --extra cpu --extra dev --locked pytest baselines/selforacle/tests

# Integration test using a synthetic ASL log and metrics in the simulator environment:
PYTHONPATH="$PWD/baselines/selforacle/src:$PWD/alpasim/src/tools" \
  alpasim/.venv/bin/python -m pytest baselines/selforacle/tests/test_evaluation.py
```

The tests cover scene leakage, protected benchmark scenes, temporal causality,
sequence/gap resets, invalid Gamma fits, checkpoint binding, and small synthetic
end-to-end SAE/VAE runs. The simulator integration test is skipped in the CPU
training environment and runs separately with AlpaSim's dependencies. Synthetic
test results do not establish monitor quality.

