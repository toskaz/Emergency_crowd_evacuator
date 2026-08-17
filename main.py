import pygame
from map.cell import CellType
from map.image_loader import generate_map_matrix
from config import *
from input import handle_input
from renderer import *
from window import Window

def main():
    map_matrix = generate_map_matrix(image_path)

    window = Window(map_matrix,drawing_size)

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

        window.draw(map_matrix, drawing_size, selected_cell_type, show_grid)
        
    window.close()

if __name__ == "__main__":
    main()