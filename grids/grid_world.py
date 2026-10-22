from collections import deque

from controls.action import Action
from graph_approach.navigation_gaphx import NavigationGraphx
from grids.cluster_grid import ClusterGrid
from grids.grid import Grid
from map.cell import CellType
from map.exit import find_exits
from utils.config import image_path


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


class Grid_World:

    def __init__(self, map_grid=None):
        self.map_grid = map_grid if map_grid is not None else Grid.from_image(image_path)
        self.occupancy_grid = Grid(self.map_grid.width, self.map_grid.height, None)
        self.drone_occupancy_grid = Grid(self.map_grid.width, self.map_grid.height, None)
        self.cluster_grid = ClusterGrid(self.map_grid)
        self.navigation_graph = NavigationGraphx(self.cluster_grid)
        self.distance_map = self.build_distance_map()

    @property
    def exits(self):
        return find_exits(self.map_grid)

    def set_cell(self, x, y, cell_type):
        if self.map_grid.get(x, y) == cell_type:
            return

        self.map_grid.set(x, y, cell_type)
        self.cluster_grid = ClusterGrid(self.map_grid)
        self.navigation_graph = NavigationGraphx(self.cluster_grid)
        self.distance_map = self.build_distance_map()

    def build_distance_map(self):
        width = self.map_grid.width
        height = self.map_grid.height
        unreachable = width * height + 1
        distances = [[unreachable for _ in range(width)] for _ in range(height)]
        pending = deque()

        for exit_ in self.exits:
            for x, y in exit_.cells:
                distances[y][x] = 0
                pending.append((x, y))

        while pending:
            x, y = pending.popleft()
            for dx, dy in ACTION_DIRECTIONS.values():
                if dx == 0 and dy == 0:
                    continue

                next_x = x + dx
                next_y = y + dy
                if not self.can_move_to(next_x, next_y):
                    continue
                if distances[next_y][next_x] <= distances[y][x] + 1:
                    continue

                distances[next_y][next_x] = distances[y][x] + 1
                pending.append((next_x, next_y))

        return distances

    def is_exit(self, x, y):
        return self.map_grid.get(x, y) == CellType.EXIT

    def can_move_to(self, x, y):
        return self.map_grid.in_bounds(x, y) and self.map_grid.get(x, y) != CellType.WALL

    def add_evacuee(self, evacuee):
        self.occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, evacuee)

    def remove_evacuee(self, evacuee):
        self.occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, None)

    def add_drone(self, drone):
        self.drone_occupancy_grid.set(drone.grid_x, drone.grid_y, drone)

    def remove_drone(self, drone):
        self.drone_occupancy_grid.set(drone.grid_x, drone.grid_y, None)

    def get_desired_moves(self, entities, actions):
        desired_moves = {}
        destinations = set()

        for entity, action in zip(entities, actions):
            dx, dy = ACTION_DIRECTIONS[action]
            x = entity.grid_x + dx
            y = entity.grid_y + dy

            if not self.can_move_to(x, y):
                continue

            destination = (x, y)
            if destination in destinations:
                continue

            destinations.add(destination)
            desired_moves[entity] = destination

        return desired_moves

    def can_complete_move(self, entity, root, desired_moves, visited, occupancy_grid):
        destination = desired_moves.get(entity)
        if destination is None:
            return False

        occupant = occupancy_grid.get(*destination)
        if occupant is None or occupant is root:
            return True

        if occupant in visited:
            return False

        visited.add(entity)
        return self.can_complete_move(occupant, root, desired_moves, visited, occupancy_grid)

    def resolve_moves(self, entities, desired_moves, occupancy_grid):
        moves = {}

        for entity in entities:
            if entity in desired_moves and self.can_complete_move(
                entity, entity, desired_moves, set(), occupancy_grid
            ):
                moves[entity] = desired_moves[entity]

        return moves

    def apply_moves(self, moves, occupancy_grid):
        for entity in moves:
            occupancy_grid.set(entity.grid_x, entity.grid_y, None)

        for entity, (x, y) in moves.items():
            entity.grid_x = x
            entity.grid_y = y
            occupancy_grid.set(x, y, entity)

    def step_evacuees(self, evacuees, actions):
        return self._step_entities(evacuees, actions, self.occupancy_grid)

    def step_drones(self, drones, actions):
        return self._step_entities(drones, actions, self.drone_occupancy_grid)

    def _step_entities(self, entities, actions, occupancy_grid):
        desired_moves = self.get_desired_moves(entities, actions)
        moves = self.resolve_moves(entities, desired_moves, occupancy_grid)
        self.apply_moves(moves, occupancy_grid)
        return moves