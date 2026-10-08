from config import image_path, default_exit_capacity
from grid import Grid
from evacuee import Evacuee
from drone import Drone
from map.cell import CellType
from map.exit import find_exits
from action import Action


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
        self.exit_capacity_grid = Grid(self.map_grid.width, self.map_grid.height, None)
        self.evacuees = []
        self.drones = []

        self.map_version = 0
        self.step_count = 0
        self.evacuated_count = 0
        self._exits = []
        self._exits_version = None

        for y in range(self.map_grid.height):
            for x in range(self.map_grid.width):
                if self.map_grid.get(x, y) == CellType.EXIT:
                    self.exit_capacity_grid.set(x, y, default_exit_capacity)

    def set_cell(self, x, y, cell_type):
        if self.map_grid.get(x, y) == cell_type:
            return

        self.map_grid.set(x, y, cell_type)

        if cell_type == CellType.EXIT:
            self.exit_capacity_grid.set(x, y, self.new_exit_cell_capacity(x, y))
        else:
            self.exit_capacity_grid.set(x, y, None)

        self.map_version += 1

    def new_exit_cell_capacity(self, x, y):
        # Painting an exit cell right next to an existing exit just makes that exit wider,
        # so the new cell takes over its capacity instead of getting the default one.
        capacities = [
            self.exit_capacity_grid.get(neighbor_x, neighbor_y)
            for neighbor_x, neighbor_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if self.map_grid.in_bounds(neighbor_x, neighbor_y)
            and self.map_grid.get(neighbor_x, neighbor_y) == CellType.EXIT
        ]

        return max(capacities, default=default_exit_capacity)

    @property
    def exits(self):
        if self._exits_version != self.map_version:
            self._exits = find_exits(self.map_grid, self.exit_capacity_grid)
            self._exits_version = self.map_version

        return self._exits

    def exit_at(self, x, y):
        for exit_ in self.exits:
            if (x, y) in exit_.cells:
                return exit_

        return None

    def set_exit_capacity(self, exit_, capacity):
        if capacity <= 0:
            return

        exit_.capacity = capacity

        for x, y in exit_.cells:
            self.exit_capacity_grid.set(x, y, capacity)

    def evacuees_in_exit(self, exit_):
        return [
            self.occupancy_grid.get(x, y)
            for x, y in exit_.cells
            if self.occupancy_grid.get(x, y) is not None
        ]

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
        evacuated = set()

        for exit_ in self.exits:
            # First come, first served: whoever has been waiting in the exit the longest goes out first.
            waiting = sorted(
                self.evacuees_in_exit(exit_),
                key=lambda evacuee: evacuee.waiting_steps,
                reverse=True
            )
            allowed = exit_.departures_allowed(self.step_count)

            for evacuee in waiting[:allowed]:
                self.occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, None)
                evacuated.add(evacuee)

            for evacuee in waiting[allowed:]:
                evacuee.waiting_steps += 1

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
        # Someone already standing in an exit is just waiting for their turn to get out,
        # so they shouldn't wander back into the building.
        evacuee_actions = [
            Action.STAY if self.touching_exit(evacuee) else action
            for evacuee, action in zip(self.evacuees, evacuee_actions)
        ]

        self.step_entities(self.evacuees, evacuee_actions, self.occupancy_grid)
        self.remove_evacuated()

        if drone_actions is not None:
            self.step_entities(self.drones, drone_actions, self.drone_occupancy_grid)

        self.step_count += 1