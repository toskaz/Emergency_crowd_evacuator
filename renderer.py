from map.cell import CellType
from colors import Color
import pygame

CELL_COLORS = {
    CellType.EMPTY: Color.EMPTY.value,
    CellType.WALL: Color.WALL.value,
    CellType.EVACUEE: Color.EVACUEE.value,
    CellType.EXIT: Color.EXIT.value,
}

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