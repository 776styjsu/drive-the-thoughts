# Select new benchmark frames for annotation

Use this workflow to browse new rollout videos and manually choose at most one
CoT–trajectory pair per source clip. The dashboard shows existing category
counts, your proposed additions, and the remaining gap to an editable target.
It prioritizes underrepresented CoT categories without using reliability,
consistency, or safety verdicts. Exported examples enter the existing annotation
builder with their answers unanswered.

Click **Take a tour** at the top of the selector for an introduction on the actual
interface. Each step scrolls to a control, outlines it in red, and explains what
to do. Use **Next**, **Back**, and **Finish**, or leave with **Close tour** or
**Esc**. The tour preserves your selections and any unfinished form entries.
Clips without usable extracted decisions get a preparation explanation instead
of the frame-selection steps; open a prepared clip and restart for those steps.

For a hands-on practice exercise, open `tutorial.html` in the generated workspace
or click **Practice example** in the selector. The eight-step walkthrough uses
an illustrated clip with three decisions. Practice selecting a rare category,
checking a suggestion, choosing and replacing a frame, and exporting an example.
Practice counts and selections are separate from the real workspace.

## Build the selection workspace

Run from the repository root. In the commands below, `$ROLLOUTS` is the folder of
AlpaSim scene runs to choose from (it may contain symbolic links to scene
folders), `$SELECTION` is where the selection workspace is written, and
`$EXCLUDE_LIST` is an optional text file of scenes to skip. `--benchmark`
defaults to `data/benchmark/benchmark.json`, whose category counts the dashboard
starts from.

```bash
uv run python tools/build_frame_selector.py \
  --rollouts "$ROLLOUTS" \
  --output-dir "$SELECTION" \
  --exclude-scenes "$EXCLUDE_LIST" \
  --target-n 20
```

`20` is an editable planning target, not a statistically determined sample size.
Choose targets to fit the study budget and intended comparisons before selecting
examples. The absent-decision category `none` starts with target zero.

The builder excludes every scene already represented in the supplied benchmark,
regardless of the selected timestamp. `--exclude-scenes` additionally excludes
scenes in the manual safety list. This option is repeatable and also accepts a
benchmark-style JSON list. Excluded scenes remain visible with their status and
no selectable decisions.

The output contains:

- `index.html` and `frame_selector.js`: the selection interface, with
  `frame_selector_tour.js` and `frame_selector_tour.css` for the embedded tour.
- `media/`: symlinks to the source scene directories; videos are not copied.
- `candidate_manifest.json`: decision timestamps, CoTs, category suggestions,
  media references, baseline counts and preparation status.
- `cot_label_requests.json`: unique CoT texts for an optional external labeler.
- `build_manifest.json`: benchmark hash, frame metadata hashes and exclusions.

Generated output is separate from the source rollouts. Changing source files
through a media symlink changes the originals. The interface itself writes only
to browser storage and downloads JSON; it does not modify the benchmark or media.

## Open it in a browser

```bash
uv run python -m http.server 8765 --bind 127.0.0.1 \
  --directory "$SELECTION"
```

Open `http://127.0.0.1:8765/`. If the server runs on a remote cluster host, create
an SSH tunnel from your laptop, replacing `CLUSTER_HOST` with that host:

```bash
ssh -N -L 8765:127.0.0.1:8765 CLUSTER_HOST
```

Then open the same localhost URL on your laptop. Keep the source data mounted;
the selection workspace uses symlinks and is not a portable media bundle.

## Prepare scenes that lack extracted frames

A scene without usable extracted decisions is shown as **Needs frame extraction**.
Its video can still be browsed, but arbitrary video timestamps cannot be exported
as benchmark examples. The benchmark needs the actual model input, CoT and plan
at a decision timestamp.

For example, prepare `scene_113` using the existing extractor:

```bash
cd alpasim
source setup_local_env.sh
bash tools/extract_rollout_frames.sh \
  --run-dir "$ROLLOUTS/scene_113_clipgt-a72b3cac-29e9-467c-bf38-f59f6b55a7d2" \
  --output-dir "$ROLLOUTS/scene_113_clipgt-a72b3cac-29e9-467c-bf38-f59f6b55a7d2/extracted_frames"
```

Extraction uses the simulator environment and can take substantially longer than
building the HTML. It writes through the scene symlink into the original scene.
The extractor normally processes completed rollouts. Investigate incomplete
runs before deciding whether to use its `--include-incomplete` option.

After extraction, rerun the builder and refresh the page. Finish extraction
before beginning a selection session: changing the candidate membership changes
the dataset ID and old sessions will be rejected rather than silently remapped.
Keep the old workspace and exported examples if you must rebuild mid-session.

A usable decision has metadata with a nonempty CoT and emitted plan, plus the
front-camera image. The interface reports extracted records missing these
artifacts. It does not filter on geometric safety validity or whether the CoT is
correct. Multiple rollout videos for a scene are rejected as ambiguous.

## Read the dashboard and select a frame

1. Enter your name or agent ID. Review the targets in the three category panels.
   Counts are recomputed from the supplied benchmark, not copied from the paper's
   figure. Source-only categories, such as `traffic_signs`, remain visible;
   unknown category spellings fail validation rather than being silently merged.
2. Click a rare category to focus the scene and decision lists. **Available**
   counts distinct unselected clips with matching provisional tags, not frames.
   Unclassified decisions are excluded from that availability count; zero does
   not establish that the phenomenon is absent from the videos.
3. Open a source clip. Scrub or play its video, then press **Use nearest decision
   to video time**. Alternatively, use the decision slider, previous/next buttons,
   or the candidate-decision list. All of these select actual extracted decisions.
4. Read the CoT and check its categories. Choose `none` only when the CoT states
   no decision in that channel. Check all stated causal-factor categories.
   Missing classification is distinct from an absent decision.
5. Confirm the category check and write a brief selection reason, such as
   “Adds a speed-adaptation example from an independent clip.” The preview shows
   which category counts would increase.
6. Press **Select this frame for the clip**. Selecting another timestamp from the
   same clip replaces its previous selection. Counts subtract the replaced
   frame's tags before adding the new tags. The log retains the prior choice.

Candidate priority is the sum of relative shortfalls across the proposed tags:
`max(0, target - current_count) / target`, treating a zero target as zero gain.
Counts include earlier confirmed selections. This is a browsing aid; it never
selects a frame automatically. Multi-label candidates can improve multiple
counts, and unclassified candidates receive no automated priority.

The extracted camera image identifies the exact decision input. Video seeking
uses `_tNNNN × --clip-step-s + --video-offset-s`, defaulting to a 0.1-second step
and zero offset, as in the existing viewer. Check alignment against the image;
use the builder flags when the video has another origin or cadence. The exact
frame ID and `time_now_us` are retained independently of this approximate seek.

## Supply category suggestions from models

The builder uses labels in extracted metadata when available, or an exact CoT
text match to the benchmark when all matching benchmark assignments agree.
These are provisional suggestions. It does not silently infer categories for
unknown text or call a model API.

Give `cot_label_requests.json` to a chosen model or agent. It contains unique
texts, the accepted category vocabulary and their associated frame IDs. Ask it
to classify stated decisions and causal factors only, without judging reliability
or safety. Return a JSON file shaped like this, using actual request IDs:

```json
{
  "schema_version": 1,
  "labels": [
    {
      "cot_id": "COPY_AN_ID_FROM_THE_REQUESTS_FILE",
      "source": "model name and prompt version",
      "categories": {
        "longitudinal": "speed_adaptation_road_events",
        "lateral": "lane_keeping_centering",
        "causal_factors": ["road_events"]
      }
    }
  ]
}
```

Use **Import CoT suggestions** in the page. Partial batches are supported;
unknown IDs, duplicate IDs and invalid values reject the entire import without
changing the session. Suggestions apply to matching texts across candidate
frames. Previously confirmed selections keep their categories.

Alternatively, pass `--cot-labels /path/to/proposals.json` when building. Human
confirmation is still required before selection. Machine labels are browsing
aids and are not reliability or safety ground truth.

## Save and hand off for annotation

Browser autosave belongs to this dataset and browser origin. Use **Save selection
session** for a portable JSON backup with targets, selections, imported suggestions
and event history. **Restore session** validates it against the candidate dataset
before replacing the browser state. Repeated imports cannot add a second frame
to a scene.

**Export selected examples** downloads `selected_examples.json`. Save it under
`$SELECTION/`, then build an annotation bundle:

```bash
uv run python tools/build_benchmark_viewer.py --annotator \
  --examples "$SELECTION/selected_examples.json" \
  --bundle "$SELECTION/annotation_bundle" \
  --title "New benchmark cases — annotation"
```

This is the existing unanswered annotation workflow. The bundle copies the
selected cases' media and can be served using the same HTTP-server command with
its directory. Selection reasons are not rendered in the annotation form, but
selection provenance, category labels, and source paths survive the annotation
JSON export. Downloading annotation answers does not append them to the 150-case
benchmark; review and merge remain separate actions.

The existing annotator displays trajectory evidence alongside its questions.
For a study requiring blinded CoT-reliability assessment, use a separate
reliability pass restricted to the model's current and preceding input evidence
before assessing trajectory consistency and safety. The selector is for curation
and does not itself enforce that annotation-stage blinding.

Retain the original benchmark and expansion membership, selection reasons,
category corrections and annotation provenance. Keep targets based on sample
size needs; do not use selection to steer reliability percentages.
