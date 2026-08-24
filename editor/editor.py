from map.cell import CellType
from editor.tool import Tool
from evacuee import Evacuee
from coordinates import pixel_to_grid

def handle_map_editor(grid, evacuees, mouse_pos, button, drawing_size, selected_tool):
    pos_x, pos_y = mouse_pos

    grid_x, grid_y = pixel_to_grid(pos_x, pos_y, drawing_size)


    # LMB = place selected cell type
    if button == 1:
        if selected_tool == Tool.WALL:
            grid.set(grid_x, grid_y, CellType.WALL)

        elif selected_tool == Tool.EXIT:
            grid.set(grid_x, grid_y, CellType.EXIT)

        elif selected_tool == Tool.EVACUEE:
            evacuees.append(Evacuee(grid_x,grid_y))

    # RMB = erase
    elif button == 3:
        grid.set(grid_x,grid_y, CellType.EMPTY)
