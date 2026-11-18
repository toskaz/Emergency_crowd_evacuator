from editor.tool import Tool
from grids.coordinates import pixel_to_grid
from map.cell import CellType

def handle_map_editor(simulation, mouse_pos, button, selected_tool):
    pos_x, pos_y = mouse_pos
    grid_x, grid_y = pixel_to_grid(pos_x, pos_y)
    return handle_map_editor_cell(simulation, grid_x, grid_y, button, selected_tool)


def handle_map_editor_cell(
    simulation, grid_x, grid_y, button, selected_tool, *, rebuild_clusters=True
):
    if simulation.started:
        return False

    world = simulation.grid_world
    if not world.map_grid.in_bounds(grid_x, grid_y):
        return False

    previous_cell = world.map_grid.get(grid_x, grid_y)
    if button == 1:
        changed = place(simulation, grid_x, grid_y, selected_tool)

    elif button == 3:
        changed = erase(simulation, grid_x, grid_y)
    else:
        return False

    if changed:
        simulation.prepared = False
        updated_cell = world.map_grid.get(grid_x, grid_y)
        clusters_changed = (
            CellType.EXIT in (previous_cell, updated_cell)
            or CellType.STAIRS in (previous_cell, updated_cell)
        )
        if clusters_changed and rebuild_clusters:
            world.rebuild_changed_map_data()
        return clusters_changed
    return False


def place(simulation, grid_x, grid_y, selected_tool):
    world = simulation.grid_world

    if selected_tool == Tool.WALL:
        cell_type = CellType.WALL

    elif selected_tool == Tool.EXIT:
        cell_type = CellType.EXIT

    elif selected_tool == Tool.EVACUEE:
        cell_type = CellType.EVACUEE

    elif selected_tool == Tool.DRONE:
        cell_type = CellType.DRONE

    elif selected_tool == Tool.STAIRS:
        cell_type = CellType.STAIRS
    else:
        return

    return world.set_map_cell(world.active_floor_id, grid_x, grid_y, cell_type)


def erase(simulation, grid_x, grid_y):
    world = simulation.grid_world
    return world.set_map_cell(world.active_floor_id, grid_x, grid_y, CellType.EMPTY)


def grid_line(start, end):
    x, y = start
    end_x, end_y = end
    delta_x = abs(end_x - x)
    delta_y = abs(end_y - y)
    step_x = 1 if x < end_x else -1
    step_y = 1 if y < end_y else -1
    error = delta_x - delta_y

    while True:
        yield x, y
        if (x, y) == (end_x, end_y):
            break
        doubled_error = 2 * error
        if doubled_error > -delta_y:
            error -= delta_y
            x += step_x
        if doubled_error < delta_x:
            error += delta_x
            y += step_y
