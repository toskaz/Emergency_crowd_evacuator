from map.cell import CellType
from editor.tool import Tool
from coordinates import pixel_to_grid
from config import exit_capacity_step

def handle_map_editor(simulation, mouse_pos, button, selected_tool):
    pos_x, pos_y = mouse_pos

    grid_x, grid_y = pixel_to_grid(pos_x, pos_y)

    if not simulation.map_grid.in_bounds(grid_x, grid_y):
        return

    if button == 1:
        place(simulation, grid_x, grid_y, selected_tool)

    elif button == 3:
        erase(simulation, grid_x, grid_y)


def place(simulation, grid_x, grid_y, selected_tool):

    if selected_tool == Tool.WALL:
        simulation.remove_drone(grid_x, grid_y)
        simulation.remove_evacuee(grid_x, grid_y)
        simulation.set_cell(grid_x, grid_y, CellType.WALL)

    elif selected_tool == Tool.EXIT:
        simulation.remove_drone(grid_x, grid_y)
        simulation.remove_evacuee(grid_x, grid_y)
        simulation.set_cell(grid_x, grid_y, CellType.EXIT)

    elif selected_tool == Tool.EVACUEE:
        simulation.remove_drone(grid_x, grid_y)
        simulation.remove_evacuee(grid_x, grid_y)
        simulation.set_cell(grid_x, grid_y, CellType.EMPTY)
        simulation.add_evacuee(grid_x, grid_y)

    elif selected_tool == Tool.DRONE:
        simulation.remove_drone(grid_x, grid_y)
        simulation.remove_evacuee(grid_x, grid_y)
        simulation.set_cell(grid_x, grid_y, CellType.EMPTY)
        simulation.add_drone(grid_x, grid_y)


def erase(simulation, grid_x, grid_y):
    simulation.remove_drone(grid_x, grid_y)
    simulation.remove_evacuee(grid_x, grid_y)
    simulation.set_cell(grid_x, grid_y, CellType.EMPTY)


def change_exit_capacity(simulation, mouse_pos, steps):
    grid_x, grid_y = pixel_to_grid(*mouse_pos)

    if not simulation.map_grid.in_bounds(grid_x, grid_y):
        return

    exit_ = simulation.exit_at(grid_x, grid_y)

    if exit_ is None:
        return

    simulation.set_exit_capacity(exit_, exit_.capacity + steps * exit_capacity_step)
