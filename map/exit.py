from collections import deque

from map.cell import CellType


# An exit is a group of EXIT cells next to each other, so a door painted with a few cells is still
# one exit.
class Exit:

    def __init__(self, cells):
        self.cells = cells


def find_exits(map_grid):
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

            exits.append(Exit(cells))

    return exits
