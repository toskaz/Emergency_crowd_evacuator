import pygame
from map.cell import CellType
from editor.editor import handle_map_editor

def handle_input(map_matrix, drawing_size, selected_cell_type, show_grid):
    for event in pygame.event.get():

        # Close window
        if event.type == pygame.QUIT:
            return False, selected_cell_type, show_grid

        # Keyboard input
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_1:
                selected_cell_type = CellType.WALL

            elif event.key == pygame.K_2:
                selected_cell_type = CellType.EVACUEE

            elif event.key == pygame.K_3:
                selected_cell_type = CellType.EXIT

            elif event.key == pygame.K_g:
                show_grid = not show_grid

        # Mouse input
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_map_editor(
                map_matrix,
                event.pos,
                event.button,
                drawing_size,
                selected_cell_type
            )

    return True, selected_cell_type, show_grid