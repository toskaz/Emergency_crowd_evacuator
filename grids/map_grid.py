import cv2

from grids.grid import Grid
from map.cell import CellType
from map.exit import find_exits
from map.exit_detector import ExitDetector
from utils.config import acceptance_threshold, block_size


class Map_Grid(Grid):
    @classmethod
    def from_image(cls, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        _, thresholded_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
        image_height, image_width = thresholded_image.shape
        grid_width = (image_width + block_size - 1) // block_size
        grid_height = (image_height + block_size - 1) // block_size

        grid = cls(grid_width, grid_height, CellType.EMPTY)
        grid._recognize_walls(thresholded_image)
        grid.recognize_exits()
        return grid

    def _recognize_walls(self, thresholded_image):
        for y in range(self.height):
            for x in range(self.width):
                block = thresholded_image[
                    y * block_size : (y + 1) * block_size,
                    x * block_size : (x + 1) * block_size,
                ]
                area = block.shape[0] * block.shape[1]
                white_pixels = cv2.countNonZero(block)
                black_pixels = area - white_pixels

                if black_pixels > area * acceptance_threshold:
                    self.set(x, y, CellType.WALL)

    def recognize_exits(self):
        return ExitDetector().mark_exits_on_grid(self)

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