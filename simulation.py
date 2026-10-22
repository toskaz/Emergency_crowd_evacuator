from grids.cluster_grid  import ClusterGrid
from utils.config import image_path
from grids.grid import Grid
from entities.evacuee import Evacuee
from entities.drone import Drone
from map.cell import CellType
from map.exit import find_exits
from controls.action import Action
from graph_approach.navigation_gaphx import NavigationGraphx


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


class Simulation:

    def __init__(self, map_grid=None):
        self.map_grid = map_grid if map_grid is not None else Grid.from_image(image_path)
        self.occupancy_grid = Grid(self.map_grid.width, self.map_grid.height, None)
        self.drone_occupancy_grid = Grid(self.map_grid.width, self.map_grid.height, None)
        self.cluster_grid = ClusterGrid(self.map_grid)
        self.navigation_graph = NavigationGraphx(self.cluster_grid)
        self.evacuees = []
        self.drones = []

        self.map_version = 0
        self.step_count = 0
        self.evacuated_count = 0
        self._exits = []
        self._exits_version = None

    def set_cell(self, x, y, cell_type):
        if self.map_grid.get(x, y) == cell_type:
            return

        self.map_grid.set(x, y, cell_type)

        self.map_version += 1
        self.cluster_grid = ClusterGrid(self.map_grid) # currently rebuilt every single time, it will need to be changes since we want editable clusters
        self.navigation_graph = NavigationGraphx(self.cluster_grid) 

    @property
    def exits(self):
        if self._exits_version != self.map_version:
            self._exits = find_exits(self.map_grid)
            self._exits_version = self.map_version

        return self._exits

    def add_evacuee(self, grid_x, grid_y):
        evacuee = Evacuee(grid_x, grid_y)

        self.evacuees.append(evacuee)
        self.occupancy_grid.set(grid_x, grid_y, evacuee)

    def remove_evacuee(self, grid_x, grid_y):
        evacuee = self.occupancy_grid.get(grid_x, grid_y)

        if evacuee is None:
            return

        self.evacuees.remove(evacuee)
        self.occupancy_grid.set(grid_x, grid_y, None)

    def add_drone(self, grid_x, grid_y):
        drone = Drone(grid_x, grid_y)
        self.drones.append(drone)
        self.drone_occupancy_grid.set(grid_x, grid_y, drone)

    def remove_drone(self, grid_x, grid_y):
        drone = self.drone_occupancy_grid.get(grid_x, grid_y)

        if drone is None:
            return

        self.drones.remove(drone)
        self.drone_occupancy_grid.set(grid_x, grid_y, None)

    def touching_exit(self, evacuee):
        return self.map_grid.get(
            evacuee.grid_x,
            evacuee.grid_y
        ) == CellType.EXIT

    def remove_evacuated(self):
        evacuated = [evacuee for evacuee in self.evacuees if self.touching_exit(evacuee)]

        for evacuee in evacuated:
            self.occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, None)
        self.evacuees[:] = [evacuee for evacuee in self.evacuees if evacuee not in evacuated]
        self.evacuated_count += len(evacuated)

    def can_move_to(self, x, y):
        if not self.map_grid.in_bounds(x, y):
            return False

        if self.map_grid.get(x, y) == CellType.WALL:
            return False

        return True

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

        if occupant is None:
            return True

        if occupant is root:
            return True

        if occupant in visited:
            return False

        visited.add(entity)

        return self.can_complete_move(occupant, root, desired_moves, visited, occupancy_grid)

    def resolve_moves(self, entities, desired_moves, occupancy_grid):
        moves = {}

        for entity in entities:
            if entity not in desired_moves:
                continue

            if self.can_complete_move(entity, entity, desired_moves, set(), occupancy_grid):
                moves[entity] = desired_moves[entity]

        return moves

    def apply_moves(self, moves, occupancy_grid):
        for entity in moves:
            occupancy_grid.set(entity.grid_x, entity.grid_y, None)

        for entity, (x, y) in moves.items():
            entity.grid_x = x
            entity.grid_y = y

            occupancy_grid.set(x, y, entity)

    def step_entities(self, entities, actions, occupancy_grid):
        desired_moves = self.get_desired_moves(entities, actions)
        moves = self.resolve_moves(entities, desired_moves, occupancy_grid)
        self.apply_moves(moves, occupancy_grid)
        return moves

    def step(self, evacuee_actions, drone_actions=None):
        self.step_entities(self.evacuees, evacuee_actions, self.occupancy_grid)
        self.remove_evacuated()

        if drone_actions is not None:
            self.step_entities(self.drones, drone_actions, self.drone_occupancy_grid)

        self.step_count += 1