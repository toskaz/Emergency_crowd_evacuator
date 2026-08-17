import pygame
from map.cell import CellType
from editor.editor import handle_map_editor
from editor.tool import Tool

def handle_input(window, evacuees, map_matrix, drawing_size, selected_tool, show_grid, simulation_running):
    for event in pygame.event.get():

        # Close window
        if event.type == pygame.QUIT:
            window.close()
            return selected_tool, show_grid

        # Keyboard input
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_1:
                selected_tool = Tool.WALL

            elif event.key == pygame.K_2:
                selected_tool = Tool.EVACUEE

            elif event.key == pygame.K_3:
                selected_tool = Tool.EXIT

            elif event.key == pygame.K_g:
                show_grid = not show_grid

            elif event.key == pygame.K_p:
                simulation_running = not simulation_running

        # Mouse input
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_map_editor(
                map_matrix,
                evacuees,
                event.pos,
                event.button,
                drawing_size,
                selected_tool
            )

    return selected_tool, show_grid, simulation_running