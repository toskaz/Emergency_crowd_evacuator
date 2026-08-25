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

    def step(self, action):
        dx, dy = ACTION_DIRECTIONS[action]

        for evacuee in self.evacuees:
            old_x = evacuee.grid_x
            old_y = evacuee.grid_y

            new_x = old_x + dx
            new_y = old_y + dy

            if not self.map_grid.in_bounds(new_x, new_y):
                continue

            if self.map_grid.get(new_x, new_y) == CellType.WALL:
                continue

            if self.occupancy_grid.get(new_x, new_y) is not None:
                continue

            self.occupancy_grid.set(old_x, old_y, None)
            self.occupancy_grid.set(new_x, new_y, evacuee)

            evacuee.grid_x = new_x
            evacuee.grid_y = new_y
        self.remove_evacuated()