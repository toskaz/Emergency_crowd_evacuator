from map.cell import CellType

def handle_map_editor(map_matrix, mouse_pos, button, drawing_size, selected_cell_type):
    pos_x, pos_y = mouse_pos

    grid_x = pos_x // drawing_size
    grid_y = pos_y // drawing_size

    max_y = len(map_matrix)
    max_x = len(map_matrix[0])

    if 0 <= grid_y < max_y and 0 <= grid_x < max_x:

        # LMB = place selected cell type
        if button == 1:
            map_matrix[grid_y][grid_x] = selected_cell_type

        # RMB = erase
        elif button == 3:
            map_matrix[grid_y][grid_x] = CellType.EMPTY
