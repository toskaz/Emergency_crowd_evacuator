from collections import deque

from controls.drone_assignment import assign_drone_groups
from controls.evacuee_rl import EXIT_REACH
from controls.line_of_sight import can_see
from entities.drone import Drone
from entities.evacuee import Evacuee
from grids.grid import Grid
from grids.map_grid import Map_Grid
from map.cell import CellType
from utils.config import acceptance_threshold, block_size, image_path, drone_vision_radius


MIN_DRONE_DISTANCE_TO_EXIT = 25
MAX_EVACUEE_DISTANCE_TO_DRONE = 6
DEFAULT_FLOOR_COUNT = 2


class CurriculumSpawnError(ValueError):
    pass


def load_default_map(simulation, floor_count=DEFAULT_FLOOR_COUNT):
    world = simulation.grid_world
    if world.floors:
        return
    for _ in range(floor_count):
        map_grid = Map_Grid.from_image(
            image_path, block_size, acceptance_threshold=acceptance_threshold
        )
        world.add_floor_from_map(map_grid)


def add_training_stairs(world):
    floor_ids = sorted(world.floors)
    linked_pairs = {
        frozenset((source[0], target[0]))
        for source, target in world._snapshot_stair_links()
    }
    for first_id, second_id in zip(floor_ids, floor_ids[1:]):
        if frozenset((first_id, second_id)) in linked_pairs:
            continue
        endpoints = []
        for floor_id in (first_id, second_id):
            cells = _training_stair_cells(world.floors[floor_id])
            for x, y in cells:
                world.set_map_cell(floor_id, x, y, CellType.STAIRS)
            endpoints.append((floor_id, *cells[0]))
        world.rebuild_changed_map_data()
        world.connect_stair_clusters(
            world.stair_cluster_at(*endpoints[0]),
            world.stair_cluster_at(*endpoints[1]),
        )
    return world._snapshot_stair_links()


def _training_stair_cells(floor):
    grid = floor.map_grid

    def available(cells):
        return all(
            grid.in_bounds(x, y) and grid.get(x, y) == CellType.EMPTY
            and not any(
                grid.in_bounds(x + dx, y + dy)
                and grid.get(x + dx, y + dy) == CellType.STAIRS
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
            )
            for x, y in cells
        )

    candidates = []
    for size in (2, 1):
        for y in range(grid.height - size + 1):
            for x in range(grid.width - size + 1):
                cells = [(x + dx, y + dy) for dy in range(size) for dx in range(size)]
                if available(cells):
                    candidates.append(cells)
        if candidates:
            break
    if not candidates:
        raise ValueError(f"No free space for training stairs on floor {floor.id}")
    return min(candidates, key=lambda cells: (
        not all(floor.distance_map[y][x] < float("inf") for x, y in cells),
        sum((x - grid.width / 2) ** 2 + (y - grid.height / 2) ** 2 for x, y in cells) / len(cells),
    ))


def configure_training_exits(world, original_exits, exit_floor_id=None):
    if exit_floor_id is not None and not original_exits.get(exit_floor_id):
        raise ValueError(f"No training exits on floor {exit_floor_id}")
    for floor_id, cells in original_exits.items():
        cell_type = CellType.EXIT if exit_floor_id in (None, floor_id) else CellType.EMPTY
        for x, y in cells:
            world.set_map_cell(floor_id, x, y, cell_type)
    world.rebuild_changed_map_data()


def place_random_scenario(
    simulation, rng, evacuee_count, floor_id=0, spread_out=False, drone_count=1
):
    world = simulation.grid_world
    reset_simulation(simulation)

    drone_cell = rng.choice(drone_start_cells(world, floor_id))
    drone = Drone(drone_cell[0], drone_cell[1], floor_id)
    simulation.drones.append(drone)
    world.add_drone(drone)
    if drone_count > 1:
        extra_drone_cells = [
            cell for cell in drone_start_cells(world, floor_id)
            if 8 <= abs(cell[0] - drone_cell[0]) + abs(cell[1] - drone_cell[1]) <= 12
        ]
        for x, y in rng.sample(extra_drone_cells, min(drone_count - 1, len(extra_drone_cells))):
            other_drone = Drone(x, y, floor_id)
            simulation.drones.append(other_drone)
            world.add_drone(other_drone)

    drone_cells = {(drone.grid_x, drone.grid_y) for drone in simulation.drones}
    candidates = [
        cell for cell in evacuee_start_cells(world, floor_id, drone_cell)
        if cell not in drone_cells
    ]
    if spread_out:
        closest = [
            cell for cell in candidates
            if abs(cell[0] - drone_cell[0]) + abs(cell[1] - drone_cell[1]) == 1
        ]
        chosen = [rng.choice(closest)] if evacuee_count and closest else []
        near = [cell for cell in candidates if cell not in chosen]
        chosen += rng.sample(near, min(3, evacuee_count - len(chosen), len(near)))
        extra = [
            cell for cell in spread_evacuee_cells(world, floor_id, drone_cell)
            if cell not in chosen and cell not in drone_cells
        ]
        chosen += rng.sample(extra, min(evacuee_count - len(chosen), len(extra)))
    else:
        chosen = rng.sample(candidates, min(evacuee_count, len(candidates)))
    for x, y in chosen:
        evacuee = Evacuee(x, y, floor_id)
        simulation.evacuees.append(evacuee)
        world.add_evacuee(evacuee)

    assign_drone_groups(world, simulation.drones, simulation.evacuees)
    world.rebuild_simulation_grid(simulation.evacuees, simulation.drones)
    simulation.started = True


def reset_simulation(simulation):
    simulation.evacuees.clear()
    simulation.drones.clear()
    simulation.step_count = 0
    simulation.evacuated_count = 0
    simulation.prepared = False
    simulation.started = False
    for floor in simulation.grid_world.floors.values():
        floor.evacuee_occupancy_grid = Grid(floor.map_grid.width, floor.map_grid.height, None)
        floor.drone_occupancy_grid = Grid(floor.map_grid.width, floor.map_grid.height, None)


def place_curriculum_scenario(simulation, rng, evacuee_count, floor_id,
                              min_distance, max_distance, require_detour=False):
    world = simulation.grid_world
    reset_simulation(simulation)
    floor = world.floors[floor_id]
    candidates = [
        (x, y) for x, y in walkable_cells(world, floor_id)
        if floor.map_grid.get(x, y) == CellType.EMPTY
        and min_distance <= floor.distance_map[y][x] <= max_distance
        and (not require_detour or not any(
            can_see(world, floor_id, x, y, ex, ey,
                    radius=floor.map_grid.width + floor.map_grid.height)
            for cells in floor.map_grid.exits for ex, ey in cells
        ))
    ]
    rng.shuffle(candidates)
    for x, y in candidates:
        nearby = [
            (px, py)
            for py in range(max(1, y - 4), min(floor.map_grid.height - 1, y + 5))
            for px in range(max(1, x - 4), min(floor.map_grid.width - 1, x + 5))
            if (px, py) != (x, y)
            and floor.map_grid.get(px, py) == CellType.EMPTY
            and floor.distance_map[py][px] > EXIT_REACH
            and can_see(world, floor_id, x, y, px, py, radius=drone_vision_radius)
        ]
        if len(nearby) >= evacuee_count:
            break
    else:
        raise CurriculumSpawnError(f"No connected curriculum spawn {min_distance}–{max_distance} steps from the exit")
    drone = Drone(x, y, floor_id)
    simulation.drones.append(drone)
    world.add_drone(drone)
    rng.shuffle(nearby)
    for px, py in nearby[:evacuee_count]:
        person = Evacuee(px, py, floor_id)
        simulation.evacuees.append(person)
        world.add_evacuee(person)
    assign_drone_groups(world, simulation.drones, simulation.evacuees)
    world.rebuild_simulation_grid(simulation.evacuees, simulation.drones)
    simulation.started = True


def place_building_scenario(
    simulation, rng, evacuee_count, floor_id, drone_count=1, distribution="scattered"
):
    if distribution not in ("clustered", "scattered", "line"):
        raise ValueError(f"Unknown crowd distribution: {distribution}")
    world = simulation.grid_world
    reset_simulation(simulation)
    start_cells = drone_start_cells(world, floor_id)
    if not start_cells:
        raise ValueError(f"No sufficiently distant drone start on floor {floor_id}")
    x, y = rng.choice(start_cells)
    primary = Drone(x, y, floor_id)
    simulation.drones.append(primary)
    world.add_drone(primary)

    candidates_by_floor = {
        candidate: drone_start_cells(world, candidate) for candidate in world.floors
    }
    for _ in range(drone_count - 1):
        candidates = [
            (candidate_floor, x, y)
            for candidate_floor, cells in candidates_by_floor.items()
            for x, y in cells
            if all(
                drone.floor_id != candidate_floor
                or abs(x - drone.grid_x) + abs(y - drone.grid_y) >= 8
                for drone in simulation.drones
            )
        ]
        same_floor = [cell for cell in candidates if cell[0] == floor_id]
        if same_floor and rng.random() < 0.5:
            candidates = same_floor
        if not candidates:
            break
        candidate_floor, x, y = rng.choice(candidates)
        drone = Drone(x, y, candidate_floor)
        simulation.drones.append(drone)
        world.add_drone(drone)

    occupied = {(drone.floor_id, drone.grid_x, drone.grid_y) for drone in simulation.drones}
    drones = simulation.drones
    counts = [evacuee_count // len(drones)] * len(drones)
    for index in range(evacuee_count % len(drones)):
        counts[index] += 1
    for drone, count in zip(drones, counts):
        anchor = drone.grid_x, drone.grid_y
        radius = rng.randint(18, 32) if distribution != "clustered" else 8
        candidates = spread_evacuee_cells(world, drone.floor_id, anchor, max_distance=radius)
        available = sum((drone.floor_id, x, y) not in occupied for x, y in candidates)
        if available < count:
            candidates = spread_evacuee_cells(world, drone.floor_id, anchor, max_distance=64)
        if distribution == "line":
            path = _crowd_line(world, drone, rng, radius)
            remainder = [cell for cell in candidates if cell not in path]
            rng.shuffle(remainder)
            candidates = path + remainder
        else:
            rng.shuffle(candidates)
        candidates.sort(key=lambda cell: abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]) != 1)
        placed = 0
        for x, y in candidates:
            if placed >= count:
                break
            position = drone.floor_id, x, y
            if position in occupied:
                continue
            evacuee = Evacuee(x, y, drone.floor_id)
            simulation.evacuees.append(evacuee)
            world.add_evacuee(evacuee)
            occupied.add(position)
            placed += 1

    assign_drone_groups(world, drones, simulation.evacuees)
    world.rebuild_simulation_grid(simulation.evacuees, drones)
    simulation.started = True


def _crowd_line(world, drone, rng, length):
    start = drone.grid_x, drone.grid_y
    parents = {start: None}
    pending = deque([(start, 0)])
    furthest = start
    while pending:
        position, distance = pending.popleft()
        furthest = position
        if distance >= length:
            continue
        neighbors = [
            (position[0] + dx, position[1] + dy)
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
            if world.can_move_to(position[0] + dx, position[1] + dy, drone.floor_id)
            and world.floors[drone.floor_id].map_grid.get(position[0] + dx, position[1] + dy) == CellType.EMPTY
            and (position[0] + dx, position[1] + dy) not in parents
        ]
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            parents[neighbor] = position
            pending.append((neighbor, distance + 1))
    path = []
    while furthest != start:
        path.append(furthest)
        furthest = parents[furthest]
    return list(reversed(path))


def spread_evacuee_cells(world, floor_id, drone_cell, max_distance=20):
    pending = deque([(drone_cell, 0)])
    seen = {drone_cell}
    cells = []
    floor = world.floors[floor_id]
    while pending:
        (x, y), distance = pending.popleft()
        if distance and floor.map_grid.get(x, y) == CellType.EMPTY:
            cells.append((x, y))
        if distance >= max_distance:
            continue
        for delta_x, delta_y in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cell = x + delta_x, y + delta_y
            if cell in seen or not world.can_move_to(*cell, floor_id):
                continue
            seen.add(cell)
            pending.append((cell, distance + 1))
    return cells


def walkable_cells(world, floor_id):
    floor = world.floors[floor_id]
    return [
        (x, y)
        for y in range(floor.map_grid.height)
        for x in range(floor.map_grid.width)
        if floor.wall_and_exit_grid.get(x, y) == CellType.EMPTY
        and floor.distance_map[y][x] != float("inf")
    ]


def drone_start_cells(world, floor_id):
    distances = world.floors[floor_id].distance_map
    return [
        (x, y)
        for x, y in walkable_cells(world, floor_id)
        if distances[y][x] >= MIN_DRONE_DISTANCE_TO_EXIT
    ]


def evacuee_start_cells(world, floor_id, drone_cell):
    drone_x, drone_y = drone_cell
    return [
        (x, y)
        for x, y in walkable_cells(world, floor_id)
        if (x, y) != drone_cell
        and abs(x - drone_x) + abs(y - drone_y) <= MAX_EVACUEE_DISTANCE_TO_DRONE
        and can_see(world, floor_id, x, y, drone_x, drone_y)
    ]
