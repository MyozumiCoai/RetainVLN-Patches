import time
from typing import Any

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator

_GEO_TIMEOUT_S = 10.0


@registry.register_measure(name="M4Pose")
@registry.register_measure(name="M4_POSE")
@registry.register_measure(name="m4_pose")
class M4Pose(Measure):
    cls_uuid: str = "m4_pose"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        self._episode = None
        self._geo_disabled = False
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode=None, **kwargs: Any):
        if episode is not None:
            self._episode = episode
        self._metric = self._build()

    def update_metric(self, *args: Any, episode=None, **kwargs: Any):
        if episode is not None:
            self._episode = episode
        self._metric = self._build()

    def _build(self) -> dict:
        try:
            state = self._sim.get_agent_state()
            pos = [float(v) for v in state.position]
            rot = state.rotation
            if hasattr(rot, "w"):
                rot = [float(rot.x), float(rot.y), float(rot.z), float(rot.w)]
            else:
                rot = [float(v) for v in rot]
            geo = float("nan")
            ep = self._episode
            if ep is not None and getattr(ep, "goals", None):
                geo = self._geodesic_to_goal(state.position, ep.goals[0].position)
            return {"position": pos, "rotation": rot, "geodesic_to_goal": geo}
        except Exception:
            return {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "geodesic_to_goal": float("nan")}

    def _geodesic_to_goal(self, pos, goal) -> float:
        import habitat_sim

        if self._geo_disabled:
            return float("nan")
        pf = getattr(self._sim, "pathfinder", None)
        if pf is None or not pf.is_loaded:
            return float("nan")
        try:
            sp = habitat_sim.ShortestPath()
            sp.requested_start = pf.snap_point(np.asarray(pos, dtype=np.float32))
            sp.requested_end = pf.snap_point(np.asarray(goal, dtype=np.float32))
            t0 = time.time()
            ok = pf.find_path(sp)
            if time.time() - t0 > _GEO_TIMEOUT_S:
                self._geo_disabled = True
                print("[M4Pose] geodesic circuit breaker tripped, disabling geo", flush=True)
                return float("nan")
            if not ok:
                return float("nan")
            pts = np.asarray(sp.points, dtype=np.float64)
            if len(pts) < 2:
                return 0.0
            return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        except Exception:
            return float("nan")
