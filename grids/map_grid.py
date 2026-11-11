from collections import deque

from grids.grid import Grid
from map.cell import CellType
from map.image_extractor import extract_map_image


class Map_Grid(Grid):
    @classmethod
    def from_image(cls, image_path, block_size, exit_kernel_size=5, acceptance_threshold=0.40):
        extracted_grid = extract_map_image(image_path, block_size, exit_kernel_size, acceptance_threshold)
        grid = cls(extracted_grid.width, extracted_grid.height, CellType.EMPTY)
        grid.cells = extracted_grid.cells
        return grid

    def add_wall(self, x, y):
        self.set(x, y, CellType.WALL)

    def add_exit(self, x, y):
        self.set(x, y, CellType.EXIT)

    def add_evacuee(self, x, y):
        self.set(x, y, CellType.EVACUEE)

    def add_drone(self, x, y):
        self.set(x, y, CellType.DRONE)

    def clear_cell(self, x, y):
        self.set(x, y, CellType.EMPTY)

    def clear(self):
        self.cells = [
            [CellType.EMPTY for _ in range(self.width)]
            for _ in range(self.height)
        ]

    @property
    def exits(self):
        exits = []
        seen = set()

        for y in range(self.height):
            for x in range(self.width):
                if self.get(x, y) != CellType.EXIT or (x, y) in seen:
                    continue

                cells = []
                queue = deque([(x, y)])
                seen.add((x, y))

                while queue:
                    cell_x, cell_y = queue.popleft()
                    cells.append((cell_x, cell_y))

                    for neighbor in (
                        (cell_x + 1, cell_y),
                        (cell_x - 1, cell_y),
                        (cell_x, cell_y + 1),
                        (cell_x, cell_y - 1),
                    ):
                        if neighbor in seen or not self.in_bounds(*neighbor):
                            continue
                        if self.get(*neighbor) == CellType.EXIT:
                            seen.add(neighbor)
                            queue.append(neighbor)

                exits.append(cells)

        return exits