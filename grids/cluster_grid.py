from map.cell import CellType
from grids.grid import Grid
from utils.config import cluster_size
from collections import deque


class ClusterGrid:
    def __init__(self, grid):
        self.grid = Grid(grid.width, grid.height)
        self.clusters = {}
        self.exit_clusters = set()
        self.adjacent_cluster_pairs = set()
        self.build(grid)

    def build(self, grid):
        cluster_id = -1
        cluster_width = 2 * cluster_size + 1

        for block_y in range(0, grid.height, cluster_width):
            for block_x in range(0, grid.width, cluster_width):
                origin_x = block_x + cluster_size
                origin_y = block_y + cluster_size

                for y in range(block_y, min(block_y + cluster_width, grid.height)):
                    for x in range(block_x, min(block_x + cluster_width, grid.width)):
                        if grid.get(x, y) == CellType.WALL:
                            self.grid.set(x, y, -1)
                        elif grid.get(x, y) != CellType.EXIT and self.grid.get(x, y) is None:
                            cluster_id += 1
                            self.clusters[cluster_id] = []
                            self.fill(grid, x, y, origin_x, origin_y, cluster_id)

        cluster_id = self.add_exit_sectors(grid, cluster_id)
        self.build_adjacent_cluster_pairs()

    def add_exit_sectors(self, grid, cluster_id):
        seen = set()

        for y in range(grid.height):
            for x in range(grid.width):
                if grid.get(x, y) != CellType.EXIT or (x, y) in seen:
                    continue

                cluster_id += 1
                self.clusters[cluster_id] = []
                self.exit_clusters.add(cluster_id)
                pending = deque([(x, y)])
                seen.add((x, y))

                while pending:
                    current_x, current_y = pending.popleft()
                    self.grid.set(current_x, current_y, cluster_id)
                    self.clusters[cluster_id].append((current_x, current_y))

                    for neighbor_x, neighbor_y in (
                        (current_x + 1, current_y),
                        (current_x - 1, current_y),
                        (current_x, current_y + 1),
                        (current_x, current_y - 1),
                    ):
                        neighbor = (neighbor_x, neighbor_y)
                        if (
                            grid.in_bounds(*neighbor)
                            and grid.get(*neighbor) == CellType.EXIT
                            and neighbor not in seen
                        ):
                            seen.add(neighbor)
                            pending.append(neighbor)

        return cluster_id

    def fill(self, grid, x, y, origin_x, origin_y, cluster_id):
        pending = [(x, y)]
        neighbors = (
            (1, 0), (-1, 0), (0, 1), (0, -1) #diagonal neighbors only if orthogonal path found
        )

        while pending:
            current_x, current_y = pending.pop()
            if not grid.in_bounds(current_x, current_y):
                continue
            if self.grid.get(current_x, current_y) is not None:
                continue

            distance = max(
                abs(current_x - origin_x),
                abs(current_y - origin_y),
            )
            if distance > cluster_size:
                continue
            if grid.get(current_x, current_y) == CellType.WALL:
                self.grid.set(current_x, current_y, -1)
                continue
            if grid.get(current_x, current_y) == CellType.EXIT:
                continue

            self.grid.set(current_x, current_y, cluster_id)
            self.clusters[cluster_id].append((current_x, current_y))

            for dx, dy in neighbors:
                pending.append((current_x + dx, current_y + dy))

    def get(self, x, y):
        return self.grid.get(x, y)

    def build_adjacent_cluster_pairs(self):
        self.adjacent_cluster_pairs = set()

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cluster_id = self.grid.get(x, y)
                if cluster_id not in self.clusters:
                    continue

                for dx, dy in ((1, 0), (0, 1)):
                    neighbor_x, neighbor_y = x + dx, y + dy
                    if not self.grid.in_bounds(neighbor_x, neighbor_y):
                        continue

                    neighbor_id = self.grid.get(neighbor_x, neighbor_y)
                    if neighbor_id in self.clusters and neighbor_id != cluster_id:
                        self.adjacent_cluster_pairs.add(
                            (min(cluster_id, neighbor_id), max(cluster_id, neighbor_id))
                        )

    def get_centroid(self, cluster_id):
        cells = self.clusters[cluster_id]
        sum_x = sum(cell[0] for cell in cells)
        sum_y = sum(cell[1] for cell in cells)
        count = len(cells)

        return (sum_x / count, sum_y / count)