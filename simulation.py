from config import image_path
from grid import Grid
from evacuee import Evacuee
from map.cell import CellType
from action import Action


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


class Simulation:

    def __init__(self):
        self.map_grid = Grid.from_image(image_path)
        self.occupancy_grid = Grid(self.map_grid.width, self.map_grid.height, None)
        self.evacuees = []

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

    def touching_exit(self, evacuee):
        return self.map_grid.get(
            evacuee.grid_x,
            evacuee.grid_y
        ) == CellType.EXIT

    def remove_evacuated(self):
        remaining = []

        for evacuee in self.evacuees:
            if self.touching_exit(evacuee):
                self.occupancy_grid.set(
                    evacuee.grid_x,
                    evacuee.grid_y,
                    None
                )
            else:
                remaining.append(evacuee)

        self.evacuees[:] = remaining

    def can_move_to(self, x, y):
        if not self.map_grid.in_bounds(x, y):
            return False

        if self.map_grid.get(x, y) == CellType.WALL:
            return False

        return True

    def get_desired_moves(self, actions):
        desired_moves = {}
        destinations = set()

        for evacuee, action in zip(self.evacuees, actions):
            dx, dy = ACTION_DIRECTIONS[action]

            x = evacuee.grid_x + dx
            y = evacuee.grid_y + dy

            if not self.can_move_to(x, y):
                continue

            destination = (x, y)

            if destination in destinations:
                continue

            destinations.add(destination)
            desired_moves[evacuee] = destination

        return desired_moves

    def can_complete_move(self, evacuee, root, desired_moves, visited):
        destination = desired_moves.get(evacuee)

        if destination is None:
            return False

        occupant = self.occupancy_grid.get(*destination)

        if occupant is None:
            return True

        if occupant is root:
            return True

        if occupant in visited:
            return False

        visited.add(evacuee)

        return self.can_complete_move(occupant, root, desired_moves, visited)

    def resolve_moves(self, desired_moves):
        moves = {}

        for evacuee in self.evacuees:
            if evacuee not in desired_moves:
                continue

            if self.can_complete_move(evacuee, evacuee, desired_moves, set()):
                moves[evacuee] = desired_moves[evacuee]

        return moves

    def apply_moves(self, moves):
        for evacuee in moves:
            self.occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, None)

        for evacuee, (x, y) in moves.items():
            evacuee.grid_x = x
            evacuee.grid_y = y

            self.occupancy_grid.set(x, y,evacuee)

    def step(self, actions):
        desired_moves = self.get_desired_moves(actions)
        moves = self.resolve_moves(desired_moves)

        self.apply_moves(moves)
        self.remove_evacuated()