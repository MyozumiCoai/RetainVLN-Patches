from __future__ import annotations

import json
import os
import random


class PerturbInjector:
    def __init__(self, p: float, deg: int, log_path: str, seed: int = 20260901):
        self.p = float(p)
        self.deg = int(deg)
        assert self.p >= 0.0, "NAVI_PERTURB_P 必须 >= 0"
        assert self.deg % 15 == 0 and self.deg > 0, "NAVI_PERTURB_DEG 必须为 15 的正倍数"
        self.turn_steps = self.deg // 15
        self.log_path = log_path
        self.rng = random.Random(seed)
        self.active_eps: set[int] = set()
        self._n_perturbs = 0
        self._n_forced = 0
        self._fh = None

    def _ensure_open(self):
        if self._fh is None:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._fh = open(self.log_path, "a", encoding="utf-8")
        return self._fh

    def _write(self, row: dict):
        self._ensure_open().write(json.dumps(row, ensure_ascii=False) + "\n")

    def maybe_force(self) -> tuple[int, int] | None:
        if self.p <= 0.0:
            return None
        if self.rng.random() >= self.p:
            return None
        side = 2 if self.rng.random() < 0.5 else 3
        self._n_perturbs += 1
        return side, self.turn_steps

    def log_perturb(self, episode_id, step: int, side: int, geo: float):
        self._n_forced += 1
        self.active_eps.add(episode_id)
        self._write({
            "type": "perturb",
            "episode_id": episode_id,
            "step": int(step),
            "side": "left" if side == 2 else "right",
            "deg": self.deg,
            "geo": float(geo),
        })

    def log_step(self, episode_id, step: int, geo: float):
        if episode_id not in self.active_eps:
            return
        self._write({
            "type": "step",
            "episode_id": episode_id,
            "step": int(step),
            "geo": float(geo),
        })

    def end_episode(self, episode_id):
        self.active_eps.discard(episode_id)

    def stats(self) -> dict:
        return {"perturb_events": self._n_perturbs, "forced_steps": self._n_forced}

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def load_injector() -> PerturbInjector | None:
    p = float(os.environ.get("NAVI_PERTURB_P", "0"))
    if p <= 0.0:
        return None
    deg = int(os.environ.get("NAVI_PERTURB_DEG", "90"))
    log_path = os.environ.get(
        "NAVI_PERTURB_LOG",
        os.path.join(os.getcwd(), "perturb_events.jsonl"),
    )
    seed = int(os.environ.get("NAVI_PERTURB_SEED", "20260901"))
    return PerturbInjector(p=p, deg=deg, log_path=log_path, seed=seed)



RECOVERY_WINDOW = 10
RECOVERY_DROP = 0.5


def compute_recovery(rows: list[dict]) -> dict:
    perturbs = [r for r in rows if r["type"] == "perturb"]
    steps: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        if r["type"] == "step":
            steps.setdefault(r["episode_id"], []).append((r["step"], r["geo"]))
    for v in steps.values():
        v.sort()

    per_episode, events = {}, []
    for pt in perturbs:
        ep, t0, geo0 = pt["episode_id"], pt["step"], pt["geo"]
        recovered = False
        for t, g in steps.get(ep, []):
            if t > t0 and t <= t0 + RECOVERY_WINDOW and g <= geo0 - RECOVERY_DROP:
                recovered = True
                break
        events.append({**pt, "recovered": recovered})
        per_episode.setdefault(ep, []).append(recovered)

    n_events = len(events)
    n_recovered = sum(1 for e in events if e["recovered"])
    return {
        "n_perturb_events": n_events,
        "n_recovered": n_recovered,
        "recovery_rate": (n_recovered / n_events) if n_events else float("nan"),
        "n_episodes_perturbed": len(per_episode),
        "per_episode": {str(k): v for k, v in per_episode.items()},
        "events": events,
    }
