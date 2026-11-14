from controls.drone_assignment import assign_drone_groups
from controls.line_of_sight import can_see
from entities.drone import Drone
from entities.evacuee import Evacuee
from grids.grid import Grid
from grids.map_grid import Map_Grid
from map.cell import CellType
from utils.config import acceptance_threshold, block_size, image_path


MIN_DRONE_DISTANCE_TO_EXIT = 25
MAX_EVACUEE_DISTANCE_TO_DRONE = 6
DEFAULT_FLOOR_COUNT = 2


def load_default_map(simulation, floor_count=DEFAULT_FLOOR_COUNT):
    world = simulation.grid_world
    if world.floors:
        return
    for _ in range(floor_count):
        map_grid = Map_Grid.from_image(
            image_path, block_size, acceptance_threshold=acceptance_threshold
        )
        world.add_floor_from_map(map_grid)


def place_random_scenario(simulation, rng, evacuee_count, floor_id=0):
    world = simulation.grid_world
    reset_simulation(simulation)

    drone_cell = rng.choice(drone_start_cells(world, floor_id))
    drone = Drone(drone_cell[0], drone_cell[1], floor_id)
    simulation.drones.append(drone)
    world.add_drone(drone)

    candidates = evacuee_start_cells(world, floor_id, drone_cell)
    for x, y in rng.sample(candidates, min(evacuee_count, len(candidates))):
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
    for floor in simulation.grid_world.floors.values():
        floor.evacuee_occupancy_grid = Grid(floor.map_grid.width, floor.map_grid.height, None)
        floor.drone_occupancy_grid = Grid(floor.map_grid.width, floor.map_grid.height, None)


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

