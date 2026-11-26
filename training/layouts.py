
from grids.map_grid import Map_Grid
from map.cell import CellType


LAYOUT_KINDS = ("rooms", "switchbacks", "mixed_building")
NAVIGATION_STAGES = ("basic", "obstacles", "big")
BUILDING_SIZES = ((49, 35), (63, 49), (77, 56))
CARDINAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def generate_navigation_building(simulation, rng, stage):
    """Single-floor navigation lessons; no stair cells or floor links."""
    if stage not in NAVIGATION_STAGES:
        raise ValueError(f"Unknown navigation stage: {stage}")
    sizes = BUILDING_SIZES if stage == "big" else ((35, 25), (39, 29), (43, 31))
    width, height = rng.choice(sizes)
    grid = Map_Grid(width, height, CellType.EMPTY)
    for x in range(width):
        grid.set(x, 0, CellType.WALL)
        grid.set(x, height - 1, CellType.WALL)
    for y in range(height):
        grid.set(0, y, CellType.WALL)
        grid.set(width - 1, y, CellType.WALL)
    if stage == "obstacles":
        _add_switchbacks(grid, rng, barriers=rng.randint(1, 2))
    elif stage == "big":
        if rng.random() < .5:
            _add_rooms(grid, rng)
        else:
            _add_switchbacks(grid, rng)
    _add_exits(grid, rng, count=1, width=1)
    simulation.grid_world.add_floor_from_map(grid)


def generate_training_building(simulation, rng, kind="mixed_building", floor_count=None):
    if kind not in LAYOUT_KINDS:
        raise ValueError(f"Unknown training layout: {kind}")
    floor_count = rng.randint(2, 4) if floor_count is None else floor_count
    if not 2 <= floor_count <= 4:
        raise ValueError("Generated training buildings require 2–4 floors")
    grids = []
    for _ in range(floor_count):
        width, height = rng.choice(BUILDING_SIZES)
        grid = Map_Grid(width, height, CellType.EMPTY)
        for x in range(width):
            grid.set(x, 0, CellType.WALL)
            grid.set(x, height - 1, CellType.WALL)
        for y in range(height):
            grid.set(0, y, CellType.WALL)
            grid.set(width - 1, y, CellType.WALL)
        floor_kind = rng.choice(("rooms", "switchbacks")) if kind == "mixed_building" else kind
        if floor_kind == "rooms":
            _add_rooms(grid, rng)
        else:
            _add_switchbacks(grid, rng)
        _add_exits(grid, rng)
        grids.append(grid)

    links = []
    for first_id in range(floor_count - 1):
        for _ in range(rng.randint(1, 2)):
            first = _add_stair_patch(grids[first_id], rng)
            second = _add_stair_patch(grids[first_id + 1], rng)
            links.append(((first_id, *first), (first_id + 1, *second)))
    for grid in grids:
        simulation.grid_world.add_floor_from_map(grid)
    world = simulation.grid_world
    for first, second in links:
        world.connect_stair_clusters(world.stair_cluster_at(*first), world.stair_cluster_at(*second))


def _add_rooms(grid, rng):
    columns = rng.randint(4, max(4, grid.width // 10))
    rows = rng.randint(3, max(3, grid.height // 10))
    xs = [round(i * (grid.width - 1) / columns) for i in range(columns + 1)]
    ys = [round(i * (grid.height - 1) / rows) for i in range(rows + 1)]
    for x in xs[1:-1]:
        for y in range(1, grid.height - 1):
            grid.set(x, y, CellType.WALL)
    for y in ys[1:-1]:
        for x in range(1, grid.width - 1):
            grid.set(x, y, CellType.WALL)

    start = (rng.randrange(columns), rng.randrange(rows))
    visited = {start}
    stack = [start]
    doors = set()
    while stack:
        x, y = stack[-1]
        neighbors = [
            (x + dx, y + dy) for dx, dy in CARDINAL_STEPS
            if 0 <= x + dx < columns and 0 <= y + dy < rows
            and (x + dx, y + dy) not in visited
        ]
        if not neighbors:
            stack.pop()
            continue
        neighbor = rng.choice(neighbors)
        doors.add(tuple(sorted(((x, y), neighbor))))
        visited.add(neighbor)
        stack.append(neighbor)
    for y in range(rows):
        for x in range(columns):
            for neighbor in ((x + 1, y), (x, y + 1)):
                if neighbor[0] < columns and neighbor[1] < rows and rng.random() < 0.25:
                    doors.add(tuple(sorted(((x, y), neighbor))))
    for (x1, y1), (x2, y2) in sorted(doors):
        width = rng.randint(1, 3)
        if x1 != x2:
            door_y = rng.randint(ys[y1] + 2, ys[y1 + 1] - width - 1)
            for y in range(door_y, door_y + width):
                grid.set(xs[max(x1, x2)], y, CellType.EMPTY)
        else:
            door_x = rng.randint(xs[x1] + 2, xs[x1 + 1] - width - 1)
            for x in range(door_x, door_x + width):
                grid.set(x, ys[max(y1, y2)], CellType.EMPTY)
    for y in range(rows):
        for x in range(columns):
            if rng.random() < 0.45:
                px = rng.randint(xs[x] + 3, xs[x + 1] - 4)
                py = rng.randint(ys[y] + 3, ys[y + 1] - 4)
                for dx in (0, 1):
                    for dy in (0, 1):
                        grid.set(px + dx, py + dy, CellType.WALL)


def _add_switchbacks(grid, rng, barriers=None):
    vertical = rng.choice((False, True))
    major = grid.width if vertical else grid.height
    minor = grid.height if vertical else grid.width
    barriers = rng.randint(3, 5) if barriers is None else barriers
    start_at_top = rng.choice((False, True))
    for index in range(1, barriers + 1):
        position = round(index * (major - 1) / (barriers + 1))
        gap_width = rng.randint(2, 4)
        gap_start = 1 if (index % 2 == 0) == start_at_top else minor - 1 - gap_width
        for cross in range(1, minor - 1):
            if gap_start <= cross < gap_start + gap_width:
                continue
            x, y = (position, cross) if vertical else (cross, position)
            grid.set(x, y, CellType.WALL)


def _add_exits(grid, rng, count=None, width=1):
    count = rng.randint(1, 2) if count is None else count
    for side in rng.sample(("left", "right", "top", "bottom"), count):
        if side in ("left", "right"):
            x = 0 if side == "left" else grid.width - 1
            inner_x = 1 if side == "left" else grid.width - 2
            candidates = [y for y in range(2, grid.height - 1 - width)
                          if all(grid.get(inner_x, y + offset) == CellType.EMPTY for offset in range(width))]
            start = rng.choice(candidates)
            for offset in range(width):
                grid.set(x, start + offset, CellType.EXIT)
        else:
            y = 0 if side == "top" else grid.height - 1
            inner_y = 1 if side == "top" else grid.height - 2
            candidates = [x for x in range(2, grid.width - 1 - width)
                          if all(grid.get(x + offset, inner_y) == CellType.EMPTY for offset in range(width))]
            start = rng.choice(candidates)
            for offset in range(width):
                grid.set(start + offset, y, CellType.EXIT)


def _add_stair_patch(grid, rng):
    existing = [
        (x, y) for y in range(grid.height) for x in range(grid.width)
        if grid.get(x, y) == CellType.STAIRS
    ]
    candidates = []
    for y in range(2, grid.height - 3):
        for x in range(2, grid.width - 3):
            if all(grid.get(x + dx, y + dy) == CellType.EMPTY for dx in (0, 1) for dy in (0, 1)) and all(
                abs(x - sx) + abs(y - sy) >= 6 for sx, sy in existing
            ):
                candidates.append((x, y))
    if existing:
        candidates.sort(key=lambda cell: min(abs(cell[0] - x) + abs(cell[1] - y) for x, y in existing), reverse=True)
        candidates = candidates[:max(1, len(candidates) // 4)]
    x, y = rng.choice(candidates)
    for dx in (0, 1):
        for dy in (0, 1):
            grid.set(x + dx, y + dy, CellType.STAIRS)
    return x, y
