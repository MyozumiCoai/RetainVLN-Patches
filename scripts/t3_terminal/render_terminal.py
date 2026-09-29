from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

SUB_STEP_M = 0.25
RENDER_GEO = 4.5


def _dist_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def extend_path_with_goal(path: Sequence[Sequence[float]],
                          goal: Sequence[float]) -> List[List[float]]:
    ext = [[float(v) for v in p] for p in path]
    if not ext:
        return [[float(v) for v in goal]]
    if _dist_xz(ext[-1], goal) > 0.15:
        ext.append([float(goal[0]), float(ext[-1][1]), float(goal[2])])
    return ext


def densify_path(path_ext: Sequence[Sequence[float]],
                 sub_step: float = SUB_STEP_M) -> List[dict]:
    subs: List[dict] = []
    for j in range(len(path_ext) - 1):
        a, b = path_ext[j], path_ext[j + 1]
        dx = float(b[0]) - float(a[0])
        dz = float(b[2]) - float(a[2])
        seg_len = math.hypot(dx, dz)
        if seg_len < 1e-9:
            continue
        heading = math.atan2(dz, dx)
        n_sub = max(1, int(round(seg_len / sub_step)))
        step = seg_len / n_sub
        y = float(a[1]) if len(a) > 1 else 0.0
        for k in range(1, n_sub + 1):
            d = step * k
            subs.append({
                "position": [float(a[0]) + dx / seg_len * d, y,
                             float(a[2]) + dz / seg_len * d],
                "heading_rad": heading,
            })
    return subs


def _geodesic(sim, pos, goal) -> float:
    import habitat_sim
    import numpy as np

    pf = getattr(sim, "pathfinder", None)
    if pf is None or not pf.is_loaded:
        return 0.0
    try:
        sp = habitat_sim.ShortestPath()
        sp.requested_start = pf.snap_point(np.asarray(pos, dtype=np.float32))
        sp.requested_end = pf.snap_point(np.asarray(goal, dtype=np.float32))
        if not pf.find_path(sp):
            return 0.0
        pts = np.asarray(sp.points, dtype=np.float64)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    except Exception:
        return 0.0


def render_episode_terminal(
    scene_glb: str,
    path: Sequence[Sequence[float]],
    goal_position: Sequence[float],
    out_dir: str,
    *,
    render_geo: float = RENDER_GEO,
    sub_step: float = SUB_STEP_M,
    resolution: int = 512,
    eye_height: float = 1.25,
) -> List[dict]:
    import habitat_sim
    import imageio
    import numpy as np
    from habitat_sim.utils.common import quat_from_angle_axis

    path_ext = extend_path_with_goal(path, goal_position)
    subs = densify_path(path_ext, sub_step)
    if not subs:
        return []

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensor_spec = habitat_sim.SensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.resolution = [resolution, resolution]
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_spec.position = [0.0, eye_height, 0.0]
    agent_cfg.sensor_specifications = [sensor_spec]
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    goal_pt = np.array([float(v) for v in goal_position], dtype=np.float32)
    os.makedirs(out_dir, exist_ok=True)
    agent = sim.get_agent(0) if hasattr(sim, "get_agent") else sim.agents[0]

    records: List[dict] = []
    try:
        for t, sub in enumerate(subs, start=1):
            px, pz = float(sub["position"][0]), float(sub["position"][2])
            heading = float(sub["heading_rad"])
            geo = _geodesic(sim, np.array([px, 0.0, pz]), goal_pt)
            frame_path: Optional[str] = None
            if geo < render_geo:
                rot = quat_from_angle_axis(
                    -heading - math.pi / 2, np.array([0.0, 1.0, 0.0])
                )
                state = habitat_sim.AgentState(
                    np.array([px, 0.0, pz], dtype=np.float32), rot
                )
                agent.set_state(state)
                obs = sim.get_sensor_observations()["rgb"]
                frame_path = os.path.join(out_dir, f"step{t:04d}.png")
                imageio.imwrite(frame_path, obs)
            records.append({
                "timestep": t,
                "position": [px, 0.0, pz],
                "heading_rad": heading,
                "geodesic_to_goal": geo,
                "rendered": frame_path is not None,
                "frame": frame_path,
                "yawed": False,
                "drifting": False,
            })
    finally:
        if hasattr(sim, "close"):
            sim.close()
    return records
