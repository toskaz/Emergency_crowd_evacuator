from map.cell import CellType
from config import cluster_size


class ClusterGrid:
    def __init__(self, grid):
        self.grid = [[None] * grid.width for _ in range(grid.height)]
        self.clusters = {}
        self.build(grid)

    def build(self, grid):
        cluster_id = 0
        cluster_width = 2 * cluster_size + 1

        for block_y in range(0, grid.height, cluster_width):
            for block_x in range(0, grid.width, cluster_width):
                origin_x = block_x + cluster_size
                origin_y = block_y + cluster_size

                for y in range(block_y, min(block_y + cluster_width, grid.height)):
                    for x in range(block_x, min(block_x + cluster_width, grid.width)):
                        if grid.get(x, y) == CellType.WALL:
                            self.grid[y][x] = -1
                        elif self.grid[y][x] is None:
                            cluster_id += 1
                            self.clusters[cluster_id] = []
                            self.fill(grid, x, y, origin_x, origin_y, cluster_id)

    def fill(self, grid, x, y, origin_x, origin_y, cluster_id):
        pending = [(x, y)]
        neighbors = (
            (1, 0), (-1, 0), (0, 1), (0, -1) #diagonal neighbors only if orthogonal path found
        )

        while pending:
            current_x, current_y = pending.pop()
            if not grid.in_bounds(current_x, current_y):
                continue
            if self.grid[current_y][current_x] is not None:
                continue

            distance = max(
                abs(current_x - origin_x),
                abs(current_y - origin_y),
            )
            if distance > cluster_size:
                continue
            if grid.get(current_x, current_y) == CellType.WALL:
                self.grid[current_y][current_x] = -1
                continue

            self.grid[current_y][current_x] = cluster_id
            self.clusters[cluster_id].append((current_x, current_y))

            for dx, dy in neighbors:
                pending.append((current_x + dx, current_y + dy))

    def get(self, x, y):
        return self.grid[y][x]