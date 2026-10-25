from dataclasses import dataclass, field
from enum import Enum, auto


class ClusterType(Enum):
    NORMAL = auto()
    EXIT = auto()


@dataclass
class Cluster:
    cells: list[tuple[int, int]] = field(default_factory=list)
    type: ClusterType = ClusterType.NORMAL
    centroid: tuple[float, float] | None = None

    def calculate_centroid(self):
        if not self.cells:
            raise ValueError("Cannot calculate a centroid for an empty cluster")

        cell_count = len(self.cells)
        center_x = sum(x for x, _ in self.cells) / cell_count
        center_y = sum(y for _, y in self.cells) / cell_count
        self.centroid = (center_x, center_y)

    def get_centroid(self):
        if self.centroid is None:
            raise RuntimeError("Cluster centroid has not been calculated")
        return self.centroid