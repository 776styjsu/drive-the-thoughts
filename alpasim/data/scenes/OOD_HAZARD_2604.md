# OOD and hazard challenge subset (26.04)

`ood_hazard_2604` contains 123 runnable scenes from AlpaSim's `public_2604`
suite. The original priorities 1--23 are the complete exact overlap with
NVIDIA's PhysicalAI AV OOD-reasoning release. Priorities 24--59 add 36 scenes
that appear in NVIDIA CASCADE's independently reviewed, eventful/non-nominal
set. Priorities 60--123 add 64 scenes ranked from NVIDIA's older NuRec scene
taxonomy using hazard proxies.

These are hazardous or unusual situations, not ground-truth Alpamayo failures.
Use repeated closed-loop runs to determine whether Alpamayo 1.5 fails. Of the
first 23, 20 have released human-refined OOD annotations; priorities 2, 22, and
23 are challenge-test cases whose reference chains of causation are hidden.
The appended 100 are challenge candidates, not additional OOD-reasoning labels.

| Priority | Scene ID | Human annotation signal |
|---:|---|---|
| 1 | `clipgt-69dd5202-7f15-4842-82f4-82a16f5a01d5` | Pedestrian conflict: yield to a pedestrian crossing the street. |
| 2 | `clipgt-ed1734bc-6b7b-409f-b4b3-27cfbbf2c642` | OOD challenge-test pedestrian-close event. |
| 3 | `clipgt-0ca937b1-ac3f-45ec-8306-6583cb3da2aa` | Emergency/work-zone traffic: decelerate behind a lead vehicle among cones and emergency vehicles. |
| 4 | `clipgt-7c6529f6-c6cf-465d-affe-b2bb30f3716d` | Unusual vehicle: move left to pass a bus parked in the ego lane. |
| 5 | `clipgt-3764254a-8653-437b-9ad7-667778d0524a` | Road debris: steer left to avoid cones in the roadway. |
| 6 | `clipgt-44c3b4d5-de26-4ba6-b748-c3a22730a717` | Complex intersection: follow a temporary cone-defined lane. |
| 7 | `clipgt-658cc9f4-955b-458c-9531-07c2a6c0c847` | Work zone: stop at a sign under a worker's guidance. |
| 8 | `clipgt-9c628a6c-1e33-4b49-9bad-493dd8142946` | Work zone: stop at a sign under a worker's guidance. |
| 9 | `clipgt-febd36c4-abb1-498c-9e79-7f4d90d0285f` | Work-zone intersection: stop for a worker, then proceed. |
| 10 | `clipgt-95a1ee7b-9b72-4417-8d46-ec58f235be5b` | Stop behind a lead work vehicle. |
| 11 | `clipgt-000548db-e266-49e5-a832-6674ab53a615` | One-way construction control: slow and steer right on exit. |
| 12 | `clipgt-70fdeed8-93d6-41e5-93aa-36746d3e454d` | Follow right-then-left barrier and cone channelization. |
| 13 | `clipgt-52bd89ce-db30-4525-9fa1-b0c0dd72f5f4` | Pedestrians crossing near adjacent vehicles and barricades. |
| 14 | `clipgt-d35e7c57-2f5d-4de9-98c5-9e218e4e18f1` | Proceed on green through cones and construction equipment. |
| 15 | `clipgt-ac73935a-548f-402f-8a6b-16688261b219` | Proceed on green and curve left along barriers. |
| 16 | `clipgt-e828096d-5889-40bb-bc6f-2feed7db9821` | Steer left following construction barriers. |
| 17 | `clipgt-f109888b-1fff-48c7-b35c-c02307c58216` | Steer left following delineators. |
| 18 | `clipgt-63653da3-ce51-4654-9c86-7edaef8772e5` | Steer right following delineators. |
| 19 | `clipgt-37c64595-d4b1-45ab-90af-4076cada161d` | Follow a lead vehicle through cones. |
| 20 | `clipgt-e59e7852-4176-4ec6-9c09-c875384ca617` | Stop at a red light and stop line. |
| 21 | `clipgt-738c453d-7196-44ef-ab43-75c719834c15` | Follow temporary lane markings. |
| 22 | `clipgt-08b2f1e4-ffcf-4342-9447-5ad2df55dfe1` | OOD challenge-test work-zone event. |
| 23 | `clipgt-40fc64e5-b2ce-4f35-ac58-ecd3f9964818` | Two OOD challenge-test work-zone events. |

Sources:

- <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
- <https://huggingface.co/spaces/nvidia/PhysicalAI-AV-OOD-Reasoning-Challenge-2026>
- <https://huggingface.co/datasets/nvidia/cascade>
- <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>

Exact per-row provenance and selection signals are recorded in
`sim_scenes_ood_hazard_2604_provenance.csv`. CASCADE membership is used only as
an eventful-scene selection signal because its detailed annotations are gated.
The final 64 were ranked deterministically using VRUs, construction zones,
intersections, turns, lane changes, stops, wet/night conditions, traffic
density, and ego speed. Reconstruction quality breaks ties.

Run the full suite from the AlpaSim directory:

```bash
uv run alpasim_wizard \
  deploy=local_apptainer \
  topology=1gpu \
  driver=alpamayo1_5 \
  wizard.log_dir="$PWD/tutorial_alpamayo_ood_hazard_2604" \
  scenes.test_suite_id=ood_hazard_2604 \
  "scenes.suites_csv=[$PWD/data/scenes/sim_suites_ood_hazard_2604.csv]"
```

For a quicker first pass, append `scenes.limit_to_first_n=10`. To continue an
existing sequential run and process the newly appended rows, use:

```bash
tools/run_top_scenes_alpamayo.sh \
  --csv data/scenes/sim_scenes_ood_hazard_2604.csv \
  --top-n 123 \
  --resume \
  --continue-on-error
```

## Publicly reported Alpamayo failure

`reported_alpamayo1_failure` contains
`clipgt-fa0379c6-d60e-48b7-b157-11bf5b63039d`. NVlabs/alpasim issue 20
reports that Alpamayo 1 reasons about nudging left but plans right, producing a
lateral collision/off-road event. A follow-up reports reproducing it on four
H100s. This is direct failure evidence for Alpamayo 1, not proof that the
RL-post-trained Alpamayo 1.5 still fails.

- <https://github.com/NVlabs/alpasim/issues/20>
- <https://github.com/NVlabs/alpamayo/issues/38>

Run it with the same command, replacing the suite and log arguments with:

```bash
wizard.log_dir="$PWD/tutorial_alpamayo_reported_failure" \
scenes.test_suite_id=reported_alpamayo1_failure \
"scenes.suites_csv=[$PWD/data/scenes/sim_suites_reported_alpamayo1_failure.csv]"
```
