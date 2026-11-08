"""Monitor 1: the two-second rule, checked live during the simulation.

Each time Alpamayo returns a trajectory, check whether any actor is inside the
area EGO could reach in the next 2 seconds. The area starts at EGO's front
bumper, is (speed x 2 s) long, and widens by 1/2 x 3 m/s^2 x (2 s)^2 = 6 m on
each side. Parked cars outside the driving lanes and actors behind EGO are
ignored.

The monitor is added to AlpaSim's list of log receivers in event_loop.py. It
only writes alerts; it never changes what the car does.

Output, next to rollout.asl, one line per trajectory:
    two_second_alerts.jsonl
    {"time_s": 15.1, "ego_speed_mps": 11.8,
     "alerts": [{"id": "81", "label": "car", "gap_m": 14.2, "gap_s": 1.2}]}

    ego_speed_mps  EGO's speed, in m/s
    label          what the actor is (car, truck, pedestrian, ...)
    gap_m          distance between EGO and the actor, in meters
    gap_s          gap_m / EGO's speed: how many seconds behind the actor EGO is
"""

import json
import logging
import math
import os

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

SECONDS = 2.0  # how far ahead to look
SIDE_ACCEL = 3.0  # how fast EGO can move sideways, m/s^2
STOPPED_SPEED = 0.1  # m/s; below this EGO is treated as stopped


class TwoSecondMonitor:
    def __init__(self, rollout_dir, vector_map):
        self.alerts_path = os.path.join(rollout_dir, "two_second_alerts.jsonl")
        self.lanes = lanes_from_map(vector_map)
        self.sizes = {}  # actor id -> (length, width)
        self.labels = {}  # actor id -> what the actor is (car, truck, ...)
        self.parked = set()  # ids of actors that never move
        self.warmup_us = 0
        self.start_us = None
        self.poses = {}  # actor id -> {time_us: (x, y, heading)}
        self.now_us = None  # time of the trajectory Alpamayo is working on

    async def on_message(self, entry):
        """Called by AlpaSim for every log entry. Errors are logged, never raised."""
        try:
            self.handle(entry)
        except Exception:
            logger.exception("TwoSecondMonitor failed")

    def handle(self, entry):
        kind = entry.WhichOneof("log_entry")

        if kind == "rollout_metadata":  # at the start: sizes, labels and warm-up length
            for actor in entry.rollout_metadata.actor_definitions.actor_aabb:
                self.sizes[actor.actor_id] = (actor.aabb.size_x, actor.aabb.size_y)
                self.labels[actor.actor_id] = actor.actor_label
            self.warmup_us = entry.rollout_metadata.force_gt_duration

        elif kind == "traffic_session_request":  # at the start: which actors are parked
            for actor in entry.traffic_session_request.logged_object_trajectories:
                if actor.is_static:
                    self.parked.add(actor.object_id)

        elif kind == "actor_poses":  # every 0.1 s: where everyone is
            t = entry.actor_poses.timestamp_us
            if self.start_us is None:
                self.start_us = t
            for actor in entry.actor_poses.actor_poses:
                pose = to_xy_heading(actor.actor_pose)
                self.poses.setdefault(actor.actor_id, {})[t] = pose

        elif kind == "driver_request":  # Alpamayo is asked for a trajectory
            self.now_us = entry.driver_request.time_now_us

        elif kind == "driver_return":  # Alpamayo returned it: check now
            self.check(self.now_us)

    def check(self, now):
        if now - self.start_us < self.warmup_us:
            return  # warm-up: EGO is still following the recording

        ego = self.poses["EGO"][now]
        before = self.poses["EGO"][now - 100_000]
        speed = math.dist(ego[:2], before[:2]) / 0.1

        alerts = []
        if speed > STOPPED_SPEED:
            area = area_ahead(*ego, speed, *self.sizes["EGO"], SECONDS)
            ego_outline = actor_box(*ego, *self.sizes["EGO"])
            for actor_id, poses in sorted(self.poses.items()):
                if actor_id == "EGO" or now not in poses:
                    continue
                pose = poses[now]
                if self.is_ignored(actor_id, pose, ego):
                    continue
                outline = actor_box(*pose, *self.sizes[actor_id])
                if area.intersects(outline):
                    gap = ego_outline.distance(outline)
                    alerts.append(
                        {
                            "id": actor_id,
                            "label": self.labels[actor_id],
                            "gap_m": round(gap, 1),
                            "gap_s": round(gap / speed, 2),
                        }
                    )

        time_s = round((now - self.start_us) / 1e6, 2)
        row = {"time_s": time_s, "ego_speed_mps": round(speed, 1), "alerts": alerts}
        with open(self.alerts_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        for alert in alerts:
            logger.info("TwoSecondMonitor: %.1f s ALERT %s", time_s, alert)

    def is_ignored(self, actor_id, pose, ego):
        """A parked car outside the driving lanes, or an actor behind EGO's rear bumper."""
        if actor_id in self.parked:
            center = actor_box(*pose, *self.sizes[actor_id]).centroid
            if self.lanes is None or not self.lanes.contains(center):
                return True
        dx, dy = pose[0] - ego[0], pose[1] - ego[1]
        distance_ahead = dx * math.cos(ego[2]) + dy * math.sin(ego[2])
        return distance_ahead < -self.sizes["EGO"][0] / 2


def actor_box(x, y, heading, length, width):
    """The actor's outline: a length x width rectangle at (x, y), facing `heading`."""
    shape = box(-length / 2, -width / 2, length / 2, width / 2)
    return translate(rotate(shape, heading, origin=(0, 0), use_radians=True), x, y)


def area_ahead(x, y, heading, speed, length, width, seconds):
    """Area ahead of a vehicle's front bumper: (speed x seconds) long, widening
    by 1/2 x SIDE_ACCEL x seconds^2 on each side."""
    reach = speed * seconds
    widen = min(0.5 * SIDE_ACCEL * seconds**2, reach)
    front, side = length / 2, width / 2
    corners = [
        (front, side),
        (front + reach, side + widen),
        (front + reach, -side - widen),
        (front, -side),
    ]
    return translate(
        rotate(Polygon(corners), heading, origin=(0, 0), use_radians=True), x, y
    )


def lanes_from_map(vector_map):
    """All driving lanes from AlpaSim's scene map, combined into one shape."""
    try:
        from trajdata.maps.vec_map_elements import MapElementType

        lanes = []
        for lane in vector_map.elements[MapElementType.ROAD_LANE].values():
            if lane.left_edge is not None and lane.right_edge is not None:
                left = [tuple(p[:2]) for p in lane.left_edge.xy]
                right = [tuple(p[:2]) for p in lane.right_edge.xy]
                # buffer(0) repairs outlines that cross themselves
                lanes.append(Polygon(left + right[::-1]).buffer(0))
        return unary_union(lanes)
    except Exception:
        logger.warning(
            "TwoSecondMonitor: no lanes in the scene map; all parked cars are ignored"
        )
        return None


def to_xy_heading(pose):
    """(x, y, heading) from a log pose. The heading is the car's direction on the map."""
    q = pose.quat
    heading = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return pose.vec.x, pose.vec.y, heading
