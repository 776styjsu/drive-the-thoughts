# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import numpy as np

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers.base import Scorer


class GroundTruthScorer(Scorer):
    """Scorer for metrics comparing to the ground truth trajectory.

    Adds the following metrics:
    * progress: The progress along the _full_ ground truth trajectory.
    * progress_rel: The progress along the current ground truth trajectory up to
        the current timestamp. Gives a better sense of progression during the
        simulation.
    * dist_to_gt_trajectory: Projected distance to the ground truth trajectory.
    * dist_to_gt_location: The distance to the ground truth ego location at the
        current timestamp.
    """

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        # Heuristically set the first two timestamps. For
        # `progress_along_full_gt` we set them at the end by interpolation.
        progress_along_current_gt = [1.0, 1.0]
        progress_along_full_gt = []
        distance_to_gt_trajectory = [0.0, 0.0]
        distance_to_current_gt_point = [0.0, 0.0]
        distance_traveled = [0.0, 0.0]

        # Skip first two timestamps to avoid errors in shapely's project function
        for idx in range(2, len(simulation_result.timestamps_us)):
            ts = simulation_result.timestamps_us[idx]

            ego_polygon = (
                simulation_result.actor_polygons.get_polygon_for_agent_at_time(
                    "EGO", ts
                )
            )
            ego_trajectory = simulation_result.actor_trajectories["EGO"].to_linestring()
            full_gt_trajectory = simulation_result.ego_recorded_ground_truth_trajectory
            full_gt_linestring = full_gt_trajectory.to_linestring()
            current_gt_linestring = full_gt_trajectory.interpolate_to_timestamps(
                simulation_result.timestamps_us[: idx + 1]
            ).to_linestring()

            progress_along_full_gt.append(
                full_gt_linestring.project(ego_polygon.centroid, normalized=True)
            )
            progress_along_current_gt.append(
                current_gt_linestring.project(ego_polygon.centroid, normalized=True)
            )
            distance_traveled.append(ego_trajectory.project(ego_polygon.centroid))

            current_gt_point = full_gt_trajectory.interpolate_to_timestamps(
                np.array([ts])
            ).to_point()

            distance_to_current_gt_point.append(
                current_gt_point.distance(ego_polygon.centroid)
            )
            distance_to_gt_trajectory.append(
                full_gt_linestring.distance(ego_polygon.centroid)
            )

        # Heuristically interpolate the first two timestamps
        if len(progress_along_full_gt) > 0:
            progress_along_full_gt = (
                list(np.linspace(0, progress_along_full_gt[0], 3)[:2])
                + progress_along_full_gt
            )

        return [
            MetricReturn(
                name="progress",
                values=progress_along_full_gt,
                valid=[True] * len(progress_along_full_gt),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.LAST,
            ),
            MetricReturn(
                name="progress_rel",
                values=progress_along_current_gt,
                valid=[True] * len(progress_along_current_gt),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MIN,
            ),
            MetricReturn(
                name="dist_to_gt_trajectory",
                values=distance_to_gt_trajectory,
                valid=[True] * len(distance_to_gt_trajectory),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="dist_to_gt_location",
                values=distance_to_current_gt_point,
                valid=[True] * len(distance_to_current_gt_point),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="dist_traveled_m",
                values=distance_traveled,
                valid=[True] * len(distance_traveled),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.LAST,
            ),
        ]
