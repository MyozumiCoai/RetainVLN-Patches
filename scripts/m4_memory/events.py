from __future__ import annotations

import math

TURN_THRESHOLD_DEG = 20.0
LOOP_MIN_GAP = 30


def norm_angle_deg(a_deg: float) -> float:
    return (a_deg + 180.0) % 360.0 - 180.0


def detect_turn(
    prev_heading_rad: float,
    cur_heading_rad: float,
    threshold_deg: float = TURN_THRESHOLD_DEG,
) -> float:
    d = norm_angle_deg(math.degrees(cur_heading_rad - prev_heading_rad))
    return d if abs(d) >= threshold_deg else 0.0


def loop_detected(
    region_grid,
    x: float,
    z: float,
    step: int,
    min_gap: int = LOOP_MIN_GAP,
) -> bool:
    if not region_grid.is_visited_before(x, z):
        return False
    last = region_grid.last_visit_step(x, z)
    return last >= 0 and (step - last) > min_gap
