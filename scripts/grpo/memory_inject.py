from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from scripts.m4_memory.manager import MemoryManager

Pose = Tuple[Sequence[float], float]


def build_step_memory(
    poses: Sequence[Pose],
    *,
    budget_tokens: int = 400,
    level: str = "level1",
) -> List[str]:
    if level not in ("level1", "level2"):
        raise ValueError(f"level 必须是 level1/level2，收到 {level!r}")
    include_notebook = level == "level2"
    mm = MemoryManager(budget_tokens=budget_tokens)
    texts: List[str] = []
    for step, (position, heading_rad) in enumerate(poses):
        mm.update(position, heading_rad, step)
        texts.append(mm.context_text(include_notebook=include_notebook))
    return texts


def memory_for_step(
    poses: Sequence[Pose],
    step: int,
    *,
    budget_tokens: int = 400,
    level: str = "level1",
) -> str:
    if step < 0 or step >= len(poses):
        raise IndexError(f"step={step} 超出 poses 范围 [0,{len(poses)})")
    return build_step_memory(poses[: step + 1], budget_tokens=budget_tokens, level=level)[step]


def inject(question: str, memory_text: Optional[str]) -> str:
    if not memory_text:
        return question
    return f"{memory_text}\n{question}"


def poses_from_rows(rows: Sequence[dict]) -> List[Pose]:
    poses: List[Pose] = []
    for r in rows:
        pos = r.get("position", r.get("pos", r.get("current_position")))
        head = r.get("heading_rad", r.get("heading"))
        if pos is None or head is None:
            raise KeyError(f"行缺 position/heading_rad，无法重建记忆: keys={sorted(r.keys())}")
        poses.append((pos, float(head)))
    return poses
