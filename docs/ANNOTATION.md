# Viewing and annotating CoT–trajectory pairs

`tools/build_benchmark_viewer.py` turns any example set (a benchmark-entry JSON:
a list, or `{"entries": [...]}`) into an HTML page that shows, for each clip,
the front camera frame, the planned trajectory, the CoT, and the recorded
labels, with a faceted coverage panel. The pages are generated, not shipped:
build them from a clone, and open them from where they were written, since they
link to `data/media/` by relative path.

Run every command from the repository root. Generated pages named
`*.viewer.html`, `*.annotator.html` or `*.review.html` under `data/benchmark/`,
and bundles under `data/benchmark/annotation/`, are gitignored.

## Demo

The starter kit in `data/benchmark/` holds two 15-pair sets drawn from the
benchmark:

- `worked_examples_with_answers.json`: worked examples with every verdict and
  its justification filled in, for a new annotator to read first.
- `qualification_examples_unanswered.json`: the same fields left blank, for a
  new annotator to answer and compare with the recorded labels in
  `benchmark.json`.

This kit is a starting point for new annotators. It is not the protocol the
paper's labels came from (a 20-pair calibration set, then 10 independently
double-annotated pairs).

```bash
# Read-only viewer of both sets -> data/benchmark/viewer.html
uv run python tools/build_benchmark_viewer.py

# The same sets as an annotation page: recorded answers sit behind a per-clip
# reveal, each clip gets verdict buttons and justification boxes, and the
# results export as JSON in the example-set schema.
uv run python tools/build_benchmark_viewer.py --annotator \
  --output data/benchmark/demo.annotator.html
```

## Annotate or review a full set

```bash
# Annotation pass over the 150-pair benchmark.
uv run python tools/build_benchmark_viewer.py --annotator \
  --examples "Benchmark=data/benchmark/benchmark.json" \
  --title "Benchmark — annotation pass"

# Review pass: every clip starts from its recorded answer, editable in place.
# The export marks which clips changed and from what; an untouched pass writes
# the set back unchanged, in the schema it was read in.
uv run python tools/build_benchmark_viewer.py --prefill \
  --examples "Benchmark=data/benchmark/benchmark.json" \
  --title "Benchmark — review pass"

# Annotation pass over the challenge set. Sets outside benchmark.json carry no
# NuRec or decision labels, so --attributes adds facets from a per-clip CSV.
uv run python tools/build_benchmark_viewer.py --annotator \
  --examples "Challenge set=data/benchmark/ood_hazard_2604_unsafe_traces.json" \
  --attributes alpasim/data/scenes/sim_scenes_ood_hazard_2604_provenance.csv \
  --title "Challenge set — annotation pass"
```

`ood_hazard_2604_unsafe_traces.json` is the 19-pair challenge set before
adjudication. `ood_hazard_2604_unsafe_traces_adjudicated.json` is the 15-pair
set the paper reports, with each revised label carrying an `adjudication` record.

## Hand a page to someone else

Add `--bundle <dir>` to write a self-contained folder instead: `index.html` plus
a copy of every frame, plot and video under `media/`. Zip it, or serve it with
`python -m http.server --directory <dir>`. A bundle of the full benchmark is
about 430 MB, mostly video.

## Selecting new frames to annotate

To pick new decision frames from rollout videos first, see the
[frame selection tutorial](FRAME_SELECTION_TUTORIAL.md). Its export opens
directly in the annotation page above.
