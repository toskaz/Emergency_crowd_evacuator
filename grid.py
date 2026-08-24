import cv2
from map.cell import CellType
from config import block_size, acceptance_threshold

class Grid:

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.cells = [
            [CellType.EMPTY for _ in range(width)]
            for _ in range(height)
        ]

    @classmethod
    def from_image(cls, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

        height, width = thresh.shape

        grid_width = (width + block_size - 1) // block_size
        grid_height = (height + block_size - 1) // block_size

        grid = cls(grid_width,grid_height)

        for y in range(grid_height):
            for x in range(grid_width):

                block = thresh[y * block_size : (y + 1) * block_size, x * block_size : (x + 1) * block_size]

                area = block.shape[0] * block.shape[1]

                number_of_white_pixels = cv2.countNonZero(block)
                number_of_black_pixels = area - number_of_white_pixels

                if number_of_black_pixels > (area * acceptance_threshold):
                    grid.set(x, y, CellType.WALL)

        return grid

    def get(self, x, y):
        return self.cells[y][x]

    def set(self, x, y, cell_type):
        self.cells[y][x] = cell_type

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height