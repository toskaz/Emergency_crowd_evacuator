import cv2
import numpy as np

from grids.grid import Grid
from map.cell import CellType


def extract_map_image(image_path, block_size, exit_kernel_size=5, acceptance_threshold=0.40):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    _, thresholded_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    image_height, image_width = thresholded_image.shape
    grid_width = (image_width + block_size - 1) // block_size
    grid_height = (image_height + block_size - 1) // block_size
    grid = Grid(grid_width, grid_height, CellType.EMPTY)

    _recognize_walls(grid, thresholded_image, block_size, acceptance_threshold)
    _mark_exits_on_grid(grid, exit_kernel_size)
    return grid


def _recognize_walls(grid, thresholded_image, block_size, acceptance_threshold):
    for y in range(grid.height):
        for x in range(grid.width):
            block = thresholded_image[
                y * block_size : (y + 1) * block_size,
                x * block_size : (x + 1) * block_size,
            ]
            area = block.shape[0] * block.shape[1]
            white_pixels = cv2.countNonZero(block)
            black_pixels = area - white_pixels

            if black_pixels > area * acceptance_threshold:
                grid.set(x, y, CellType.WALL)


def _mark_exits_on_grid(grid, kernel_size):
    wall_mask = _get_wall_mask(grid)
    closed_walls = _close_doors(wall_mask, kernel_size)
    outside_area = _find_outside_area(closed_walls)
    exits = _find_external_openings(wall_mask, closed_walls, outside_area)

    for x, y in exits:
        grid.set(x, y, CellType.EXIT)


def _get_wall_mask(grid):
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.get(x, y) == CellType.WALL:
                mask[y, x] = 255
    return mask


def _close_doors(wall_mask, kernel_size):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)


def _find_outside_area(closed_walls):
    padded = cv2.copyMakeBorder(
        closed_walls, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    cv2.floodFill(padded, None, (0, 0), 128)
    return padded[1:-1, 1:-1] == 128


def _find_external_openings(original_walls, closed_walls, outside_area):
    height, width = original_walls.shape
    exits = []

    for y in range(height):
        for x in range(width):
            was_empty = original_walls[y, x] == 0
            got_closed = closed_walls[y, x] == 255
            if was_empty and got_closed and _touches_outside(
                x, y, outside_area, width, height
            ):
                exits.append((x, y))

    return exits


def _touches_outside(x, y, outside_area, width, height):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor_x, neighbor_y = x + dx, y + dy
        if (
            0 <= neighbor_x < width
            and 0 <= neighbor_y < height
            and outside_area[neighbor_y, neighbor_x]
        ):
            return True
    return False