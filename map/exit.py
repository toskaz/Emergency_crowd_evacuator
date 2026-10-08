import math
from collections import deque

from map.cell import CellType


# An exit is a group of EXIT cells next to each other, so a door painted with a few cells is still
# one exit. capacity says how many people can get out through it in one simulation step.
class Exit:

    def __init__(self, cells, capacity):
        self.cells = cells
        self.capacity = capacity

    def departures_allowed(self, step):
        # How many people can get out in this step. Over t steps it adds up to floor(capacity * t),
        # so with capacity 2 two people leave every step and with 0.5 one person every other step.
        # The tiny 1e-9 is only there so float rounding (like 2.9999999) doesn't eat a person.
        return math.floor(self.capacity * (step + 1) + 1e-9) - math.floor(self.capacity * step + 1e-9)


def find_exits(map_grid, capacity_grid):
    exits = []
    seen = set()

    for y in range(map_grid.height):
        for x in range(map_grid.width):
            if map_grid.get(x, y) != CellType.EXIT or (x, y) in seen:
                continue

            cells = []
            queue = deque([(x, y)])
            seen.add((x, y))

            while queue:
                cell_x, cell_y = queue.popleft()
                cells.append((cell_x, cell_y))

                for neighbor in ((cell_x + 1, cell_y), (cell_x - 1, cell_y), (cell_x, cell_y + 1), (cell_x, cell_y - 1)):
                    if neighbor in seen or not map_grid.in_bounds(*neighbor):
                        continue

                    if map_grid.get(*neighbor) == CellType.EXIT:
                        seen.add(neighbor)
                        queue.append(neighbor)

            capacity = max(capacity_grid.get(cell_x, cell_y) for cell_x, cell_y in cells)
            exits.append(Exit(cells, capacity))

    return exits
