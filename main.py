from config import *
from input import handle_input
from renderer import *
from window import Window
from grid import Grid
from editor.tool import Tool
from action import Action

def touching_exit(evacuee, grid):
    return grid.get(evacuee.grid_x, evacuee.grid_y) == CellType.EXIT

ACTION_DIRECTIONS = {
    Action.STAY:  (0, 0),
    Action.UP:    (0, -1),
    Action.DOWN:  (0, 1),
    Action.LEFT:  (-1, 0),
    Action.RIGHT: (1, 0),
}

def update_evacuees(evacuees, grid, action):

    dx, dy = ACTION_DIRECTIONS[action]

    for evacuee in evacuees:

        new_x = evacuee.grid_x + dx
        new_y = evacuee.grid_y + dy

        if not grid.in_bounds(new_x, new_y):
            continue

        cell = grid.get(new_x, new_y)

        if cell == CellType.WALL:
            continue

        evacuee.grid_x = new_x
        evacuee.grid_y = new_y

    evacuees[:] = [
        evacuee
        for evacuee in evacuees
        if not touching_exit(evacuee, grid)
    ]
def main():
    grid = Grid.from_image(image_path)

    window = Window(grid,drawing_size)

    selected_tool = Tool.WALL
    show_grid = True

    evacuees = []
    clock = pygame.time.Clock()
    simulation_running = False
    while window.is_open:
        
        selected_tool, show_grid, simulation_running, action = handle_input(
            window,
            evacuees,
            grid,
            drawing_size,
            selected_tool,
            show_grid,
            simulation_running
        )


        if simulation_running:
            update_evacuees(
                evacuees,
                grid,
                action
            )

        window.draw(grid, evacuees, drawing_size, selected_tool, show_grid)

if __name__ == "__main__":
    main()