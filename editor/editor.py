import pygame
from editor.tool import Tool
from grids.coordinates import pixel_to_grid
from map.cell import CellType

def handle_map_editor(simulation, mouse_pos, button, selected_tool):
    if simulation.started:
        return

    pos_x, pos_y = mouse_pos

    grid_x, grid_y = pixel_to_grid(pos_x, pos_y)

    world = simulation.grid_world
    if not world.map_grid.in_bounds(grid_x, grid_y):
        return

    if button == 1:
        place(simulation, grid_x, grid_y, selected_tool)

    elif button == 3:
        erase(simulation, grid_x, grid_y)


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
    elif selected_tool == Tool.ERASER:
        erase(simulation, grid_x, grid_y)
        return
    else:
        return

    current = world.map_grid.get(grid_x, grid_y)
    if current is not None and current != CellType.EMPTY and current != cell_type:
        is_wall_on_exit = (cell_type == CellType.WALL and current == CellType.EXIT)
        is_exit_on_wall = (cell_type == CellType.EXIT and current == CellType.WALL)
        if not (is_wall_on_exit or is_exit_on_wall):
            simulation.last_editor_error = "Cannot place here: cell is already occupied!"
            simulation.last_editor_error_timer = pygame.time.get_ticks() + 2500
            return

    world.set_map_cell(world.active_floor_id, grid_x, grid_y, cell_type)
    if cell_type in (CellType.EXIT, CellType.STAIRS) or current in (CellType.EXIT, CellType.STAIRS):
        world.rebuild_changed_map_data()


def erase(simulation, grid_x, grid_y):
    world = simulation.grid_world
    current = world.map_grid.get(grid_x, grid_y)
    world.set_map_cell(world.active_floor_id, grid_x, grid_y, CellType.EMPTY)
    if current in (CellType.EXIT, CellType.STAIRS):
        world.rebuild_changed_map_data()
