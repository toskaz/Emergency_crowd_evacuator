import pygame
from map.cell import CellType
from map.image_loader import generate_map_matrix
from config import *
from input import handle_input
from renderer import *

def initialize_pygame(map_matrix, drawing_size):
    pygame.init()

    screen_width = len(map_matrix[0]) * drawing_size
    screen_height = len(map_matrix) * drawing_size

    screen = pygame.display.set_mode((screen_width, screen_height))

    font = pygame.font.Font(None, 24)

    return screen, font


def exit():
    pygame.quit()


def draw(screen, font, map_matrix, drawing_size, selected_cell_type, show_grid):
    draw_map(
        screen,
        map_matrix,
        drawing_size,
        show_grid
    )

    draw_ui(
        screen,
        font,
        selected_cell_type
    )

    pygame.display.flip()


def main():
    map_matrix = generate_map_matrix(image_path)

    screen, font = initialize_pygame(
        map_matrix,
        drawing_size
    )

    selected_cell_type = CellType.WALL
    show_grid = True

    working = True

    while working:
        working, selected_cell_type, show_grid = handle_input(
            map_matrix,
            drawing_size,
            selected_cell_type,
            show_grid
        )

        draw(
            screen,
            font,
            map_matrix,
            drawing_size,
            selected_cell_type,
            show_grid
        )
    exit()

if __name__ == "__main__":
    main()