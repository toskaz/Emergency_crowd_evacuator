import math
from collections import deque

import numpy as np

from grid import Grid
from map.cell import CellType
from graph_approach.floyd_warshall import floyd_warshall, reconstruct_path

NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# Running Floyd-Warshall on every single cell of the map would take minutes, so the drones plan
# on a much smaller graph instead. The map is cut into block_size x block_size blocks and every
# connected piece of free space inside a block becomes one node (a region). Regions that touch
# get an edge, weighted by the number of steps between their middle cells.
class NavigationGraph:

    def __init__(self, map_grid, block_size):
        self.width = map_grid.width
        self.height = map_grid.height
        self.block_size = block_size

        self.node_of = Grid(self.width, self.height, None)
        self.node_cells = []
        self.representatives = []
        self.edges = []

        self.build_nodes(map_grid)
        self.build_edges()

        self.dist, self.next_node = floyd_warshall(self.weight_matrix())

    def build_nodes(self, map_grid):
        for y in range(self.height):
            for x in range(self.width):
                if map_grid.get(x, y) != CellType.WALL and self.node_of.get(x, y) is None:
                    self.add_node(map_grid, x, y)

    def add_node(self, map_grid, start_x, start_y):
        node = len(self.node_cells)
        block = self.block_of(start_x, start_y)

        cells = []
        queue = deque([(start_x, start_y)])
        self.node_of.set(start_x, start_y, node)

        while queue:
            x, y = queue.popleft()
            cells.append((x, y))

            for neighbor_x, neighbor_y in self.neighbors(x, y):
                if self.block_of(neighbor_x, neighbor_y) != block:
                    continue

                if map_grid.get(neighbor_x, neighbor_y) == CellType.WALL:
                    continue

                if self.node_of.get(neighbor_x, neighbor_y) is not None:
                    continue

                self.node_of.set(neighbor_x, neighbor_y, node)
                queue.append((neighbor_x, neighbor_y))

        self.node_cells.append(cells)
        self.representatives.append(central_cell(cells))

    def build_edges(self):
        pairs = set()

        for node, cells in enumerate(self.node_cells):
            for x, y in cells:
                for neighbor in self.neighbors(x, y):
                    other = self.node_of.get(*neighbor)

                    if other is not None and other != node:
                        pairs.add((min(node, other), max(node, other)))

        for u, v in sorted(pairs):
            goal = self.representatives[v]
            path = self.local_path(self.representatives[u], {u, v}, lambda cell: cell == goal)
            self.edges.append((u, v, len(path) - 1))

    def weight_matrix(self):
        node_count = len(self.node_cells)

        weights = np.full((node_count, node_count), np.inf)
        np.fill_diagonal(weights, 0.0)

        for u, v, weight in self.edges:
            weights[u, v] = weight
            weights[v, u] = weight

        return weights

    def block_of(self, x, y):
        return x // self.block_size, y // self.block_size

    def neighbors(self, x, y):
        for dx, dy in NEIGHBOR_OFFSETS:
            if self.node_of.in_bounds(x + dx, y + dy):
                yield x + dx, y + dy

    def node_at(self, cell):
        if not self.node_of.in_bounds(*cell):
            return None

        return self.node_of.get(*cell)

    def distance(self, start, goal):
        start_node = self.node_at(start)
        goal_node = self.node_at(goal)

        if start_node is None or goal_node is None:
            return math.inf

        return float(self.dist[start_node, goal_node])

    def route(self, start, goal):
        start_node = self.node_at(start)
        goal_node = self.node_at(goal)

        if start_node is None or goal_node is None:
            return []

        return reconstruct_path(self.next_node, start_node, goal_node)

    # The cell to step on next when going from start to goal. If there's no way to get
    # there at all, we just give back start, so whoever asked simply stays where they are.
    def next_cell(self, start, goal):
        start_node = self.node_at(start)
        goal_node = self.node_at(goal)

        if start_node is None or goal_node is None:
            return start

        if start_node == goal_node:
            path = self.local_path(start, {start_node}, lambda cell: cell == goal)
        else:
            next_node = int(self.next_node[start_node, goal_node])

            if next_node < 0:
                return start

            # No need to aim for the middle of the next region, we only have to get into it,
            # so we head for whichever of its cells is closest.
            path = self.local_path(
                start,
                {start_node, next_node},
                lambda cell: self.node_of.get(*cell) == next_node
            )

        if path is None or len(path) < 2:
            return start

        return path[1]

    # Ordinary BFS, except it only walks on cells of the regions in allowed_nodes. Gives back the
    # whole path from start to the first cell that passes is_goal, or None if it can't get there.
    def local_path(self, start, allowed_nodes, is_goal):
        parent = {start: None}
        queue = deque([start])

        while queue:
            cell = queue.popleft()

            if is_goal(cell):
                path = []

                while cell is not None:
                    path.append(cell)
                    cell = parent[cell]

                return path[::-1]

            for neighbor in self.neighbors(*cell):
                if neighbor not in parent and self.node_of.get(*neighbor) in allowed_nodes:
                    parent[neighbor] = cell
                    queue.append(neighbor)

        return None


def central_cell(cells):
    center_x = sum(x for x, _ in cells) / len(cells)
    center_y = sum(y for _, y in cells) / len(cells)

    return min(cells, key=lambda cell: (cell[0] - center_x) ** 2 + (cell[1] - center_y) ** 2)
