from collections import deque

from grids.grid import Grid


def calculate_drone_assignments(world, drone_positions):
    ordered_positions = sorted(
        set(drone_positions),
        key=lambda position: (position[0], position[2], position[1]),
    )
    drone_ids = {position: drone_id for drone_id, position in enumerate(ordered_positions)}
    assignment_grids = {
        floor_id: Grid(floor.map_grid.width, floor.map_grid.height, None)
        for floor_id, floor in world.floors.items()
    }
    pending = deque()
    for (floor_id, x, y), drone_id in drone_ids.items():
        assignment_grids[floor_id].set(x, y, drone_id)
        pending.append((floor_id, x, y))

    while pending:
        floor_id, x, y = pending.popleft()
        grid = assignment_grids[floor_id]
        drone_id = grid.get(x, y)
        for delta_x, delta_y in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            next_x, next_y = x + delta_x, y + delta_y
            if not world.can_move_to(next_x, next_y, floor_id):
                continue
            if grid.get(next_x, next_y) is not None:
                continue
            grid.set(next_x, next_y, drone_id)
            pending.append((floor_id, next_x, next_y))

    return drone_ids, assignment_grids


def assign_drone_groups(world, drones, evacuees):
    drone_positions = {(drone.floor_id, drone.grid_x, drone.grid_y) for drone in drones}
    if drone_positions == set(world.drone_ids):
        drone_ids = world.drone_ids
        assignment_grids = {
            floor_id: floor.drone_assignment_grid
            for floor_id, floor in world.floors.items()
        }
    else:
        drone_ids, assignment_grids = calculate_drone_assignments(world, drone_positions)

    for drone in drones:
        drone.drone_id = drone_ids[(drone.floor_id, drone.grid_x, drone.grid_y)]
    for evacuee in evacuees:
        evacuee.assigned_drone_id = assignment_grids[evacuee.floor_id].get(
            evacuee.grid_x, evacuee.grid_y
        )
