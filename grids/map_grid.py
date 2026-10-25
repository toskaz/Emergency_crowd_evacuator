from grids.grid import Grid
from map.cell import CellType
from map.exit import find_exits
from map.image_extractor import extract_map_image


class Map_Grid(Grid):
    @classmethod
    def from_image(cls, image_path):
        extracted_grid = extract_map_image(image_path)
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
        return find_exits(self)