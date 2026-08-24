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

def update_evacuees(evacuees, grid, occupancy, action):

    dx, dy = ACTION_DIRECTIONS[action]

    for evacuee in evacuees:

        old_x = evacuee.grid_x
        old_y = evacuee.grid_y

        new_x = old_x + dx
        new_y = old_y + dy

        if not grid.in_bounds(new_x, new_y):
            continue

        if grid.get(new_x, new_y) == CellType.WALL:
            continue

        if occupancy.get(new_x, new_y) is not None:
            continue

        occupancy.set(old_x, old_y, None)
        occupancy.set(new_x, new_y, evacuee)

        evacuee.grid_x = new_x
        evacuee.grid_y = new_y

def main():
    map_grid = Grid.from_image(image_path)

    occupancy_grid = Grid(map_grid.width, map_grid.height, None)

    window = Window(map_grid,drawing_size)

    selected_tool = Tool.WALL
    show_grid = True

    evacuees = []
    clock = pygame.time.Clock()
    simulation_running = False
    while window.is_open:
        
        selected_tool, show_grid, simulation_running, action = handle_input(
            window,
            evacuees,
            map_grid,
            occupancy_grid,
            drawing_size,
            selected_tool,
            show_grid,
            simulation_running
        )


        if simulation_running:
            update_evacuees(
                evacuees,
                map_grid,
                occupancy_grid,
                action
            )

        window.draw(map_grid, evacuees, drawing_size, selected_tool, show_grid)

if __name__ == "__main__":
    main()