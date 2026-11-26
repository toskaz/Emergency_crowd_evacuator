from dataclasses import dataclass
import cv2
import numpy as np

from grids.grid import Grid
from map.cell import CellType


@dataclass
class ImageFilterSettings:
    mode: str = "auto"
    ignore_colors: bool = True
    gray_blocks: bool = False
    remove_thin_px: int = 0
    picked_rgb: tuple[int, int, int] | None = None
    picked_lab: tuple[float, float, float] | None = None
    sensitivity: float = 0.40


def sample_wall_color_cielab(bgr_image: np.ndarray, px: int, py: int):
    if bgr_image is None or bgr_image.size == 0:
        return None, None
    if bgr_image.ndim == 2:
        bgr = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = bgr_image

    h, w = bgr.shape[:2]
    cx = max(0, min(w - 1, int(px)))
    cy = max(0, min(h - 1, int(py)))
    x0, x1 = max(0, cx - 1), min(w, cx + 2)
    y0, y1 = max(0, cy - 1), min(h, cy + 2)

    patch_bgr = bgr[y0:y1, x0:x1]
    patch_f = patch_bgr.astype(np.float32) / 255.0
    patch_lab = cv2.cvtColor(patch_f, cv2.COLOR_BGR2LAB)
    med_lab = np.median(patch_lab.reshape(-1, 3), axis=0)
    med_bgr = np.median(patch_bgr.reshape(-1, 3), axis=0)

    picked_lab = (float(med_lab[0]), float(med_lab[1]), float(med_lab[2]))
    picked_rgb = (int(med_bgr[2]), int(med_bgr[1]), int(med_bgr[0]))
    return picked_lab, picked_rgb


def _remove_thin_and_noise(wall_mask: np.ndarray, remove_thin_px: int) -> np.ndarray:
    if remove_thin_px <= 0:
        return wall_mask
    k = int(remove_thin_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx((wall_mask > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    binary = (opened > 0).astype(np.uint8)

    min_area = max(9, k * k * 2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary

    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


def preprocess_floorplan_array(
    image: np.ndarray,
    filter_settings: ImageFilterSettings | None = None,
    sensitivity: float | None = None,
    is_vector_raster: bool = False,
):
    if image is None or image.size == 0:
        raise ValueError("Empty floorplan image array")

    if image.ndim == 2:
        gray = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    if is_vector_raster:
        _, thresholded = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        black_mask = (thresholded == 0).astype(np.uint8)
        integral = cv2.integral(black_mask)
        return {
            "image": gray,
            "thresholded": thresholded,
            "black_mask": black_mask,
            "integral": integral,
            "width": w,
            "height": h,
            "is_vector_raster": True,
            "is_binary_mask": True,
            "sensitivity": 0.40,
        }

    if filter_settings is None:
        _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        black_mask = (thresholded == 0).astype(np.uint8)
        integral = cv2.integral(black_mask)
        return {
            "image": gray,
            "thresholded": thresholded,
            "black_mask": black_mask,
            "integral": integral,
            "width": w,
            "height": h,
            "is_vector_raster": False,
            "is_binary_mask": False,
            "sensitivity": 0.40,
        }

    sens = float(sensitivity if sensitivity is not None else filter_settings.sensitivity)
    sens = max(0.10, min(0.80, sens))

    if filter_settings.mode == "pick_color" and filter_settings.picked_lab is not None:
        bgr_f = bgr.astype(np.float32) / 255.0
        lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2LAB)
        target = np.array(filter_settings.picked_lab, dtype=np.float32)
        delta_e = np.linalg.norm(lab - target, axis=2)
        tolerance = 8.0 + ((sens - 0.10) / 0.70) * (55.0 - 8.0)
        wall_mask = (delta_e <= tolerance).astype(np.uint8)
    else:
        if sens <= 0.40:
            gray_thresh = 80.0 + (sens / 0.40) * 40.0
        else:
            gray_thresh = 120.0 + ((sens - 0.40) / 0.40) * 78.0
        gray_thresh = int(round(max(60, min(215, gray_thresh))))

        wall_mask = (gray <= gray_thresh).astype(np.uint8)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        if filter_settings.ignore_colors:
            colored_non_dark = (sat > 50) & (val > 115)
            wall_mask[colored_non_dark] = 0

        if filter_settings.gray_blocks:
            neutral_gray_block = (sat < 40) & (val >= 120) & (val <= 215)
            wall_mask[neutral_gray_block] = 1

    if filter_settings.remove_thin_px > 0:
        wall_mask = _remove_thin_and_noise(wall_mask, filter_settings.remove_thin_px)

    thresholded = np.where(wall_mask > 0, 0, 255).astype(np.uint8)
    integral = cv2.integral(wall_mask)

    return {
        "image": gray,
        "thresholded": thresholded,
        "black_mask": wall_mask,
        "integral": integral,
        "width": w,
        "height": h,
        "is_vector_raster": False,
        "is_binary_mask": True,
        "sensitivity": sens,
    }


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


def _is_3x3_4connected(win_3x3: np.ndarray) -> bool:
    coords = [
        (r, c)
        for r in range(3)
        for c in range(3)
        if not (r == 1 and c == 1) and win_3x3[r, c]
    ]
    if len(coords) <= 1:
        return True
    coord_set = set(coords)
    visited = {coords[0]}
    stack = [coords[0]]
    while stack:
        cr, cc = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (cr + dr, cc + dc)
            if nb in coord_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(coord_set)


def _prune_wall_grid(wall_grid: np.ndarray, cov: np.ndarray) -> np.ndarray:
    gh, gw = wall_grid.shape
    if gh == 0 or gw == 0:
        return wall_grid

    grid = wall_grid.copy()
    candidates = [
        (float(cov[r, c]), r, c)
        for r in range(gh)
        for c in range(gw)
        if grid[r, c] and cov[r, c] < 0.48
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    def _window_3x3(r, c):
        win = np.zeros((3, 3), dtype=bool)
        r0, r1 = max(0, r - 1), min(gh, r + 2)
        c0, c1 = max(0, c - 1), min(gw, c + 2)
        win[(r0 - (r - 1)) : (r1 - (r - 1)), (c0 - (c - 1)) : (c1 - (c - 1))] = grid[r0:r1, c0:c1]
        win[1, 1] = False
        return win

    def _neighbors_4(r, c):
        nbs = []
        if r > 0 and grid[r - 1, c]:
            nbs.append((r - 1, c))
        if r + 1 < gh and grid[r + 1, c]:
            nbs.append((r + 1, c))
        if c > 0 and grid[r, c - 1]:
            nbs.append((r, c - 1))
        if c + 1 < gw and grid[r, c + 1]:
            nbs.append((r, c + 1))
        return nbs

    def _neighbors_8_count(win):
        return int(np.sum(win))

    for c_val, r, c in candidates:
        if not grid[r, c]:
            continue
        win = _window_3x3(r, c)
        n8 = _neighbors_8_count(win)
        n4 = _neighbors_4(r, c)

        if n8 == 1 and c_val < 0.16 and len(n4) == 1:
            nr, nc = n4[0]
            if cov[nr, nc] >= max(0.20, 2.0 * c_val):
                grid[r, c] = False
                continue

        if not n4:
            continue
        has_stronger_or_eq_4nb = any(cov[nr, nc] >= c_val for nr, nc in n4)
        if not has_stronger_or_eq_4nb:
            continue
        has_strictly_stronger_4nb = any(cov[nr, nc] > c_val for nr, nc in n4)
        if len(n4) == 1 and not has_strictly_stronger_4nb:
            continue
        if _is_3x3_4connected(win):
            grid[r, c] = False

    for c_val, r, c in candidates:
        if not grid[r, c]:
            continue
        win = _window_3x3(r, c)
        if _neighbors_8_count(win) < 2:
            continue
        n4 = _neighbors_4(r, c)
        if len(n4) != 1:
            continue
        if any(cov[nr, nc] >= c_val for nr, nc in n4) and _is_3x3_4connected(win):
            grid[r, c] = False

    return grid


def detect_grid_cells(
    preprocessed,
    block_size,
    acceptance_threshold=0.40,
    exit_kernel_size=None,
    detect_exits=True,
    meters_per_pixel=None,
):
    w = preprocessed["width"]
    h = preprocessed["height"]
    block_size = max(1, int(block_size))

    grid_width = (w + block_size - 1) // block_size
    grid_height = (h + block_size - 1) // block_size

    black_mask = preprocessed["black_mask"]
    pillar_grid = np.zeros((grid_height, grid_width), dtype=bool)
    integral = preprocessed["integral"]

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        black_mask, connectivity=8
    )
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        bbox_max = np.maximum(widths, heights).astype(np.float64)
        bbox_min = np.maximum(1, np.minimum(widths, heights)).astype(np.float64)
        aspect = bbox_max / bbox_min

        if meters_per_pixel is not None and float(meters_per_pixel) > 0:
            is_small = ((bbox_max * float(meters_per_pixel)) <= 0.75) & (aspect <= 2.2)
        else:
            is_small = (bbox_max <= (1.5 * block_size)) & (aspect <= 2.2)

        if np.any(is_small):
            is_pillar = is_small & (areas >= 4)
            pillar_indices = np.where(is_pillar)[0] + 1
            for comp_id in pillar_indices:
                cx, cy = centroids[comp_id]
                px = max(0, min(grid_width - 1, int(cx // block_size)))
                py = max(0, min(grid_height - 1, int(cy // block_size)))
                pillar_grid[py, px] = True

            keep_lut = np.ones(n_labels, dtype=np.uint8)
            keep_lut[0] = 0
            keep_lut[1:][is_small] = 0
            main_mask = keep_lut[labels]
            integral = cv2.integral(main_mask)

    y0 = np.arange(0, h, block_size)
    y1 = np.minimum(y0 + block_size, h)
    x0 = np.arange(0, w, block_size)
    x1 = np.minimum(x0 + block_size, w)

    X0, Y0 = np.meshgrid(x0, y0)
    X1, Y1 = np.meshgrid(x1, y1)

    counts = integral[Y1, X1] - integral[Y0, X1] - integral[Y1, X0] + integral[Y0, X0]
    cell_area = float(block_size * block_size)
    cov = counts.astype(np.float32) / max(1.0, cell_area)

    if preprocessed.get("is_vector_raster"):
        min_cov = 0.04
    elif preprocessed.get("is_binary_mask"):
        sens = float(preprocessed.get("sensitivity", acceptance_threshold))
        if abs(sens - 0.40) < 1e-6:
            min_cov = 0.045
        elif sens > 0.40:
            min_cov = max(0.025, 0.045 - ((sens - 0.40) / 0.40) * 0.018)
        else:
            min_cov = min(0.10, 0.045 + ((0.40 - sens) / 0.30) * 0.045)
    else:
        min_cov = max(0.03, min(0.12, (acceptance_threshold / 0.40) * 0.045))

    wall_grid = cov >= min_cov

    wall_grid = _prune_wall_grid(wall_grid, cov)
    wall_bool = wall_grid | pillar_grid

    wall_mask = (wall_bool * 255).astype(np.uint8)
    wall_cells = {
        (x, y)
        for y in range(grid_height)
        for x in range(grid_width)
        if wall_bool[y, x]
    }

    exit_cells = set()
    if detect_exits and len(wall_cells) > 0:
        if exit_kernel_size is not None:
            kernel_sz = exit_kernel_size
        else:
            kernel_sz = max(5, int(np.ceil(130.0 / block_size)) + 1)
            if kernel_sz % 2 == 0:
                kernel_sz += 1

        closed_walls = _close_doors(wall_mask, kernel_sz)
        outside_area = _find_outside_area(closed_walls)
        exits = _find_external_openings(wall_mask, closed_walls, outside_area)
        exit_cells = set(exits)

    return wall_cells, exit_cells, grid_width, grid_height


def extract_map_image(image_path, block_size, exit_kernel_size=None, acceptance_threshold=0.40):
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
    inside_mask = ((~outside_area) & (original_walls == 0)).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inside_mask)
    if n_labels > 1:
        largest_comp_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        real_interior = (labels == largest_comp_idx)
    else:
        real_interior = inside_mask.astype(bool)

    exits = []
    for y in range(height):
        for x in range(width):
            was_empty = original_walls[y, x] == 0
            got_closed = closed_walls[y, x] == 255
            if was_empty and got_closed:
                if _touches_region(x, y, outside_area, width, height) and _touches_region(x, y, real_interior, width, height):
                    exits.append((x, y))

    return exits


def _touches_region(x, y, region, width, height):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor_x, neighbor_y = x + dx, y + dy
        if (
            0 <= neighbor_x < width
            and 0 <= neighbor_y < height
            and region[neighbor_y, neighbor_x]
        ):
            return True
    return False
