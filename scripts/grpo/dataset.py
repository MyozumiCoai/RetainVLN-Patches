from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

from scripts.grpo.memory_inject import build_step_memory, memory_for_step

EpKey = Tuple[str, object]


def load_points(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_by_episode(points: List[dict]) -> Dict[EpKey, List[dict]]:
    grouped: Dict[EpKey, List[dict]] = defaultdict(list)
    for p in points:
        grouped[(str(p["scene_id"]), p["episode_id"])].append(p)
    for k in grouped:
        grouped[k].sort(key=lambda r: r["timestep"])
    return grouped


def _assert_contiguous(rows: List[dict], key: EpKey) -> None:
    ts = [r["timestep"] for r in rows]
    if ts != list(range(len(ts))):
        raise ValueError(
            f"episode {key} timestep 不连续（{ts}）——记忆重建需 0..n-1 完整前缀，"
            f"请用 build_grpo_data 的整条 episode 输出"
        )


def attach_memory(
    points: List[dict], *, memory_level: str = "level1", budget: int = 400
) -> List[dict]:
    if memory_level not in ("off", "level1", "level2"):
        raise ValueError(f"memory_level 必须是 off/level1/level2，收到 {memory_level!r}")
    if any(r.get("prefix_poses") is not None for r in points):
        if memory_level == "off":
            out = [dict(r, memory_text="", memory_level="off") for r in points]
        else:
            out = []
            for r in points:
                pp = [(p[:3], p[3]) for p in r["prefix_poses"]]
                mt = memory_for_step(
                    pp, len(pp) - 1, budget_tokens=budget, level=memory_level
                )
                rr = dict(r)
                rr["memory_text"] = mt
                rr["memory_level"] = memory_level
                out.append(rr)
        out.sort(key=lambda r: (str(r["scene_id"]), str(r["episode_id"]), r["timestep"]))
        return out
    grouped = group_by_episode(points)
    out: List[dict] = []
    for key, rows in grouped.items():
        _assert_contiguous(rows, key)
        if memory_level == "off":
            texts = [""] * len(rows)
        else:
            poses = [(r["position"], r["heading_rad"]) for r in rows]
            texts = build_step_memory(poses, budget_tokens=budget, level=memory_level)
        for r, mt in zip(rows, texts):
            rr = dict(r)
            rr["memory_text"] = mt
            rr["memory_level"] = memory_level
            out.append(rr)
    out.sort(key=lambda r: (str(r["scene_id"]), str(r["episode_id"]), r["timestep"]))
    return out


def prepare_samples(path: str, *, memory_level: str = "level1", budget: int = 400) -> List[dict]:
    return attach_memory(load_points(path), memory_level=memory_level, budget=budget)
