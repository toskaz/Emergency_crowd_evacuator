from config import *
from input import handle_input
from renderer import *
from window import Window
from evacuee import Evacuee
from grid import Grid
from editor.tool import Tool

def touching_exit(evacuee, grid, drawing_size):
    grid_x = int(evacuee.position.x // drawing_size)
    grid_y = int(evacuee.position.y // drawing_size)

    return grid.get(grid_x, grid_y) == CellType.EXIT

def update_evacuees(evacuees, grid, drawing_size, delta_time):
    keys = pygame.key.get_pressed()

    direction = pygame.Vector2()

    if keys[pygame.K_w]:
        direction.y -= 1

    if keys[pygame.K_s]:
        direction.y += 1

    if keys[pygame.K_a]:
        direction.x -= 1

    if keys[pygame.K_d]:
        direction.x += 1

    if direction.length_squared() > 0:
        direction = direction.normalize()

    for evacuee in evacuees:

        movement = direction * evacuee.speed * delta_time

        new_position = evacuee.position + movement

        grid_x = int(new_position.x // drawing_size)
        grid_y = int(new_position.y // drawing_size)

        cell = grid.get(grid_x, grid_y)

        if cell == CellType.WALL:
            continue

        evacuee.position = new_position

    evacuees[:] = [
        evacuee
        for evacuee in evacuees
        if not touching_exit(
            evacuee,
            grid,
            drawing_size
        )
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
        
        selected_tool, show_grid, simulation_running = handle_input(
            window,
            evacuees,
            grid,
            drawing_size,
            selected_tool,
            show_grid,
            simulation_running
        )

        delta_time = clock.tick(60) / 1000.0

        if simulation_running:
            update_evacuees(evacuees, grid, drawing_size, delta_time)
        #temporary for testing

        window.draw(grid, evacuees, drawing_size, selected_tool, show_grid)

if __name__ == "__main__":
    main()