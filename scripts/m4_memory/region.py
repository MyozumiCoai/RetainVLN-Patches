from __future__ import annotations

from typing import Dict, Sequence, Tuple


class RegionGrid:
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.visited: Dict[Tuple[int, int], int] = {}

    def cell(self, x: float, z: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(z // self.cell_size))

    def mark(self, x: float, z: float, step: int) -> bool:
        c = self.cell(x, z)
        if c in self.visited:
            self.visited[c] = step
            return False
        self.visited[c] = step
        return True

    def is_visited_before(self, x: float, z: float) -> bool:
        return self.cell(x, z) in self.visited

    def last_visit_step(self, x: float, z: float) -> int:
        return self.visited.get(self.cell(x, z), -1)

    def visited_cells(self) -> Sequence[Tuple[int, int]]:
        return sorted(self.visited.keys())

    def label(self, c: Tuple[int, int]) -> str:
        return f"区域({c[0]},{c[1]})"
