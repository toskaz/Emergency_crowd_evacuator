from map.cell import CellType
from utils.config import evacuee_vision_radius


def can_see(world, floor_id, from_x, from_y, to_x, to_y):
    if not is_within_vision(from_x, from_y, to_x, to_y):
        return False

    walls = world.floors[floor_id].wall_and_exit_grid
    for x, y in line_cells(from_x, from_y, to_x, to_y):
        if walls.get(x, y) == CellType.WALL:
            return False
    return True


def is_within_vision(from_x, from_y, to_x, to_y):
    dx = to_x - from_x
    dy = to_y - from_y
    return dx * dx + dy * dy <= evacuee_vision_radius * evacuee_vision_radius


def line_cells(from_x, from_y, to_x, to_y):
    dx = abs(to_x - from_x)
    dy = abs(to_y - from_y)
    step_x = 1 if to_x > from_x else -1
    step_y = 1 if to_y > from_y else -1
    error = dx - dy
    x, y = from_x, from_y
    cells = []

    while (x, y) != (to_x, to_y):
        cells.append((x, y))
        doubled_error = 2 * error
        if doubled_error > -dy:
            error -= dy
            x += step_x
        if doubled_error < dx:
            error += dx
            y += step_y

    cells.append((to_x, to_y))
    return cells
