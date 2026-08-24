from map.cell import CellType
from editor.tool import Tool
from evacuee import Evacuee
from coordinates import pixel_to_grid

def handle_map_editor(map_grid, evacuees, occupancy_grid, mouse_pos, button, drawing_size, selected_tool):
    pos_x, pos_y = mouse_pos

    grid_x, grid_y = pixel_to_grid(pos_x, pos_y, drawing_size)

    if not map_grid.in_bounds(grid_x, grid_y):
        return

    # LMB = place selected cell type
    if button == 1:
        if selected_tool == Tool.WALL:
            map_grid.set(grid_x, grid_y, CellType.WALL)

        elif selected_tool == Tool.EXIT:
            map_grid.set(grid_x, grid_y, CellType.EXIT)

        elif selected_tool == Tool.EVACUEE:

            if occupancy_grid.get(grid_x, grid_y) is None:
                evacuee = Evacuee(grid_x, grid_y)

                evacuees.append(evacuee)

                occupancy_grid.set(
                    grid_x,
                    grid_y,
                    evacuee
                )
                map_grid.set(grid_x, grid_y, CellType.EMPTY)

    # RMB = erase
    elif button == 3:
        map_grid.set(grid_x,grid_y, CellType.EMPTY)
