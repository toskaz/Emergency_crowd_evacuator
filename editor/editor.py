from map.cell import CellType
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

    if selected_tool == Tool.WALL:
        simulation.grid_world.set_cell(grid_x, grid_y, CellType.WALL)

    elif selected_tool == Tool.EXIT:
        simulation.grid_world.set_cell(grid_x, grid_y, CellType.EXIT)

    elif selected_tool == Tool.EVACUEE:
        simulation.grid_world.set_cell(grid_x, grid_y, CellType.EVACUEE)

    elif selected_tool == Tool.DRONE:
        simulation.grid_world.set_cell(grid_x, grid_y, CellType.DRONE)


def erase(simulation, grid_x, grid_y):
    simulation.grid_world.set_cell(grid_x, grid_y, CellType.EMPTY)
