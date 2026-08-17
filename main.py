from map.cell import CellType
from map.image_loader import generate_map_matrix
from config import *
from input import handle_input
from renderer import *
from window import Window
from evacuee import Evacuee
from grid import Grid
from editor.tool import Tool

def main():
    grid = Grid.from_image(image_path)

    window = Window(grid,drawing_size)

    selected_tool = Tool.WALL
    show_grid = True

    evacuees = []

    while window.is_open:
        selected_tool, show_grid = handle_input(
            window,
            evacuees,
            grid,
            drawing_size,
            selected_tool,
            show_grid
        )

        window.draw(grid, drawing_size, selected_tool, show_grid)

if __name__ == "__main__":
    main()