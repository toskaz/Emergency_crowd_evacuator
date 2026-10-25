from editor.tool import Tool
from grids.coordinates import pixel_to_grid

def handle_map_editor(simulation, mouse_pos, button, selected_tool):
    if simulation.started:
        return

    pos_x, pos_y = mouse_pos

    grid_x, grid_y = pixel_to_grid(pos_x, pos_y)

    if not simulation.grid_world.map_grid.in_bounds(grid_x, grid_y):
        return

    if button == 1:
        place(simulation, grid_x, grid_y, selected_tool)

    elif button == 3:
        erase(simulation, grid_x, grid_y)


def place(simulation, grid_x, grid_y, selected_tool):

    map_grid = simulation.grid_world.map_grid

    if selected_tool == Tool.WALL:
        map_grid.add_wall(grid_x, grid_y)

    elif selected_tool == Tool.EXIT:
        map_grid.add_exit(grid_x, grid_y)

    elif selected_tool == Tool.EVACUEE:
        map_grid.add_evacuee(grid_x, grid_y)

    elif selected_tool == Tool.DRONE:
        map_grid.add_drone(grid_x, grid_y)


def erase(simulation, grid_x, grid_y):
    simulation.grid_world.map_grid.clear_cell(grid_x, grid_y)
