# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from eval.schema import EvalConfig
from eval.scorers.base import Scorer, ScorerGroup
from eval.scorers.collision import CollisionScorer
from eval.scorers.cot_consistency import CotConsistencyScorer
from eval.scorers.ground_truth import GroundTruthScorer
from eval.scorers.image import ImageScorer
from eval.scorers.minADE import MinADEScorer
from eval.scorers.offroad import OffRoadScorer
from eval.scorers.plan_deviation import PlanDeviationScorer
from eval.scorers.safety import SafetyScorer

SCORERS = [
    CollisionScorer,
    OffRoadScorer,
    GroundTruthScorer,
    MinADEScorer,
    PlanDeviationScorer,
    ImageScorer,
    SafetyScorer,
]

#: Opt-in scorers, not run by default (so they do not change existing eval
#: output). Pass ``extra_scorers=[CotConsistencyScorer]`` to
#: :func:`create_scorer_group` to enable the online CoT consistency monitor.
OPTIONAL_SCORERS = [CotConsistencyScorer]


def create_scorer_group(
    cfg: EvalConfig, extra_scorers: list[type[Scorer]] | None = None
) -> ScorerGroup:
    """Initialize the default scorers plus any opt-in ``extra_scorers``."""
    scorers = [scorer(cfg) for scorer in SCORERS]
    for scorer in extra_scorers or []:
        scorers.append(scorer(cfg))
    return ScorerGroup(scorers)
