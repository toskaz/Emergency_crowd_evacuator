from map.cell import CellType
from editor.tool import Tool
from evacuee import Evacuee

def handle_map_editor(grid, evacuees, mouse_pos, button, drawing_size, selected_tool):
    pos_x, pos_y = mouse_pos

    grid_x = pos_x // drawing_size
    grid_y = pos_y // drawing_size

    #if 0 <= grid_y < max_y and 0 <= grid_x < max_x:

    # LMB = place selected cell type
    if button == 1:
        if selected_tool == Tool.WALL:
            grid.set(grid_x, grid_y, CellType.WALL)

        elif selected_tool == Tool.EXIT:
            grid.set(grid_x, grid_y, CellType.EXIT)

        elif selected_tool == Tool.EVACUEE:
            evacuees.append(Evacuee(pos_x,pos_y))

    # RMB = erase
    elif button == 3:
        grid.set(grid_x,grid_y, CellType.EMPTY)
