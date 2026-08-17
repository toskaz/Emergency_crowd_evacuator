import pygame
from map.cell import CellType
from map.image_loader import generate_map_matrix
from colors import Color
from config import *

CELL_COLORS = {
    CellType.EMPTY: Color.EMPTY.value,
    CellType.WALL: Color.WALL.value,
    CellType.EVACUEE: Color.EVACUEE.value,
    CellType.EXIT: Color.EXIT.value,
}


def initialize_pygame(map_matrix, drawing_size):
    pygame.init()

    screen_width = len(map_matrix[0]) * drawing_size
    screen_height = len(map_matrix) * drawing_size

    screen = pygame.display.set_mode((screen_width, screen_height))

    font = pygame.font.Font(None, 24)

    return screen, font



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


def handle_map_editor(
    map_matrix,
    mouse_pos,
    button,
    drawing_size,
    selected_cell_type
):
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


def draw_map(screen, map_matrix, drawing_size, show_grid):
    screen.fill(Color.BACKGROUND.value)

    for y, row in enumerate(map_matrix):
        for x, value in enumerate(row):
            pos_x = x * drawing_size
            pos_y = y * drawing_size

            color = CELL_COLORS[value]

            # Draw the cell
            pygame.draw.rect(
                screen,
                color,
                (pos_x, pos_y, drawing_size, drawing_size)
            )

            # Draw the grid line
            if show_grid:
                pygame.draw.rect(
                    screen,
                    Color.GRID.value,
                    (pos_x, pos_y, drawing_size, drawing_size),
                    1
                )


def draw_ui(screen, font, selected_cell_type):
    tool_names = {
        CellType.WALL: "WALL",
        CellType.EVACUEE: "EVACUEE",
        CellType.EXIT: "EXIT",
    }

    tool_name = tool_names[selected_cell_type]
    tool_color = CELL_COLORS[selected_cell_type]

    text = font.render(
        f"Tool: {tool_name}",
        True,
        tool_color
    )

    padding = 10

    text_rect = text.get_rect(
        top=padding,
        right=screen.get_width() - padding
    )

    screen.blit(text, text_rect)


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