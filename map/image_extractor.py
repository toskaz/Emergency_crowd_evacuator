import cv2
import numpy as np

from grids.grid import Grid
from map.cell import CellType


def preprocess_floorplan_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    _, thresholded = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    black_mask = (thresholded == 0).astype(np.uint8)
    integral = cv2.integral(black_mask)
    h, w = thresholded.shape

    return {
        "image": image,
        "thresholded": thresholded,
        "black_mask": black_mask,
        "integral": integral,
        "width": w,
        "height": h,
    }


def detect_grid_cells(
    preprocessed,
    block_size,
    acceptance_threshold=0.40,
    exit_kernel_size=5,
    detect_exits=True,
):
    integral = preprocessed["integral"]
    w = preprocessed["width"]
    h = preprocessed["height"]

    grid_width = (w + block_size - 1) // block_size
    grid_height = (h + block_size - 1) // block_size

    y0 = np.arange(0, h, block_size)
    y1 = np.minimum(y0 + block_size, h)
    x0 = np.arange(0, w, block_size)
    x1 = np.minimum(x0 + block_size, w)

    X0, Y0 = np.meshgrid(x0, y0)
    X1, Y1 = np.meshgrid(x1, y1)

    counts = integral[Y1, X1] - integral[Y0, X1] - integral[Y1, X0] + integral[Y0, X0]
    areas = (Y1 - Y0) * (X1 - X0)

    wall_bool = counts > (areas * acceptance_threshold)
    wall_mask = (wall_bool * 255).astype(np.uint8)
    wall_cells = {
        (x, y)
        for y in range(grid_height)
        for x in range(grid_width)
        if wall_bool[y, x]
    }

    exit_cells = set()
    if detect_exits and len(wall_cells) > 0:
        kernel_sz = exit_kernel_size if exit_kernel_size is not None else 5
        closed_walls = _close_doors(wall_mask, kernel_sz)
        outside_area = _find_outside_area(closed_walls)
        exits = _find_external_openings(wall_mask, closed_walls, outside_area)
        exit_cells = set(exits)

    return wall_cells, exit_cells, grid_width, grid_height


def extract_map_image(image_path, block_size, exit_kernel_size=5, acceptance_threshold=0.40):
    preprocessed = preprocess_floorplan_image(image_path)
    wall_cells, exit_cells, grid_width, grid_height = detect_grid_cells(
        preprocessed, block_size, acceptance_threshold, exit_kernel_size, detect_exits=True
    )
    grid = Grid(grid_width, grid_height, CellType.EMPTY)
    for x, y in wall_cells:
        grid.set(x, y, CellType.WALL)
    for x, y in exit_cells:
        grid.set(x, y, CellType.EXIT)
    return grid


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