from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .events import detect_turn, loop_detected
from .region import RegionGrid


class EpisodeMemory:

    def __init__(self, max_salient: int = 3, update_every: int = 5):
        self.max_salient = max_salient
        self.update_every = update_every
        self.progress: str = ""
        self.salient: List[str] = []
        self.outcome: str = "进行中"
        self.key_frames: List[str] = []
        self._step = 0
        self._cum_disp = 0.0
        self._last_pos = None
        self._turn_count = 0

    def add_salient(self, text: str):
        if text not in self.salient:
            self.salient.append(text)
            self.salient = self.salient[-self.max_salient:]

    def update(
        self,
        position: Sequence[float],
        heading_rad: float,
        step: int,
        *,
        prev_heading_rad: Optional[float] = None,
    ):
        self._step = step
        if self._last_pos is not None:
            dx = position[0] - self._last_pos[0]
            dz = position[2] - self._last_pos[2]
            self._cum_disp += (dx * dx + dz * dz) ** 0.5
        self._last_pos = list(position)

        turn = detect_turn(prev_heading_rad, heading_rad) if prev_heading_rad is not None else 0.0
        if abs(turn) >= 1e-9:
            self._turn_count += 1
            side = "左" if turn > 0 else "右"
            self.add_salient(f"第 {self._turn_count} 个转角（{side}转 {abs(turn):.0f}°）")

        if step % self.update_every == 0:
            self.progress = f"已移动 {self._cum_disp:.1f}m，当前第 {step} 步"

    def finalize(self, done_info: Optional[dict] = None):
        if done_info:
            success = bool(done_info.get("success", done_info.get("os", 0)) >= 0.5)
            self.outcome = "成功到达" if success else "未成功"
        else:
            self.outcome = "进行中"


class EvidenceNotebook:

    def __init__(self, max_each: int = 5):
        self.max_each = max_each
        self.searched_regions: List[str] = []
        self.candidate_locations: List[str] = []
        self.rejected_hypotheses: List[str] = []
        self.landmark_cues: List[str] = []

    def _add(self, lst: List[str], item: str):
        if item not in lst:
            lst.append(item)
            del lst[: max(0, len(lst) - self.max_each)]

    def add_searched(self, region_label: str):
        self._add(self.searched_regions, region_label)

    def add_candidate(self, text: str):
        self._add(self.candidate_locations, text)

    def add_rejected(self, text: str):
        self._add(self.rejected_hypotheses, text)

    def add_landmark(self, text: str):
        self._add(self.landmark_cues, text)

    def absorb(self, episode: EpisodeMemory, region_grid: RegionGrid):
        for c in region_grid.visited_cells():
            self.add_searched(region_grid.label(c))

    def to_dict(self) -> dict:
        return {
            "searched_regions": self.searched_regions,
            "candidate_locations": self.candidate_locations,
            "rejected_hypotheses": self.rejected_hypotheses,
            "landmark_cues": self.landmark_cues,
        }


class MemoryManager:

    def __init__(self, budget_tokens: int = 400):
        self.episode = EpisodeMemory()
        self.notebook = EvidenceNotebook()
        self.region_grid = RegionGrid(cell_size=2.0)
        self.budget_tokens = budget_tokens
        self._step = 0
        self._prev_heading: Optional[float] = None

    def update(
        self,
        position: Sequence[float],
        heading_rad: float,
        step: int,
        *,
        done_info: Optional[dict] = None,
    ):
        x, z = position[0], position[2]
        if loop_detected(self.region_grid, x, z, step):
            self.episode.add_salient("回到已访问区域")
        is_new = self.region_grid.mark(x, z, step)
        if is_new:
            label = self.region_grid.label(self.region_grid.cell(x, z))
            self.episode.add_salient(f"进入新区域 {label}")
            self.notebook.add_searched(label)

        self.episode.update(
            position,
            heading_rad,
            step,
            prev_heading_rad=self._prev_heading,
        )
        self._prev_heading = heading_rad
        self._step = step

        if done_info:
            self.episode.finalize(done_info)
            self.notebook.absorb(self.episode, self.region_grid)

    def context_text(self, include_notebook: bool = True) -> str:
        from .text import build_memory_context

        notebook = self.notebook.to_dict() if include_notebook else {}
        return build_memory_context(
            {
                "progress": self.episode.progress,
                "salient": self.episode.salient,
                "outcome": self.episode.outcome,
            },
            notebook,
            budget_tokens=self.budget_tokens,
        )
