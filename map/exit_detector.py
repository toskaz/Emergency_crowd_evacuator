import cv2
import numpy as np

from map.cell import CellType


class ExitDetector:

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def detect_exits(self, grid):
        wall_mask = self._get_wall_mask(grid)
        closed_walls = self._close_doors(wall_mask)
        outside_area = self._find_outside_area(closed_walls)
        return self._find_external_openings(wall_mask, closed_walls, outside_area)

    def mark_exits_on_grid(self, grid):
        exits = self.detect_exits(grid)
        for x, y in exits:
            grid.set(x, y, CellType.EXIT)
        return exits

    def _get_wall_mask(self, grid):
        mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.get(x, y) == CellType.WALL:
                    mask[y, x] = 255
        return mask

    def _close_doors(self, wall_mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size)
        )
        return cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)

    def _find_outside_area(self, closed_walls):
        padded = cv2.copyMakeBorder(
            closed_walls, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        cv2.floodFill(padded, None, (0, 0), 128)
        return padded[1:-1, 1:-1] == 128

    def _find_external_openings(self, original_walls, closed_walls, outside_area):
        height, width = original_walls.shape
        exit_cells = []

        for y in range(height):
            for x in range(width):
                was_empty = (original_walls[y, x] == 0)
                got_closed = (closed_walls[y, x] == 255)

                if was_empty and got_closed:
                    if self._touches_outside(x, y, outside_area, width, height):
                        exit_cells.append((x, y))

        return exit_cells

    def _touches_outside(self, x, y, outside_area, width, height):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and outside_area[ny, nx]:
                return True
        return False
