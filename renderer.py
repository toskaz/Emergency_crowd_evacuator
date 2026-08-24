from map.cell import CellType
from colors import Color
import pygame
from editor.tool import Tool
from coordinates import grid_to_pixel, grid_to_pixel_center


CELL_COLORS = {
    CellType.EMPTY: Color.EMPTY.value,
    CellType.WALL: Color.WALL.value,
    CellType.EXIT: Color.EXIT.value,
}


TOOL_COLORS = {
    Tool.WALL: Color.WALL.value,
    Tool.EVACUEE: Color.EVACUEE.value,
    Tool.EXIT: Color.EXIT.value,
}


def draw_map(screen, grid, drawing_size, show_grid):
    screen.fill(Color.BACKGROUND.value)

    for y in range(grid.height):
        for x in range(grid.width):
            value = grid.get(x, y)

            pos_x, pos_y = grid_to_pixel(x, y, drawing_size)
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


def draw_ui(screen, font, selected_tool):
    tool_names = {
        Tool.WALL: "WALL",
        Tool.EVACUEE: "EVACUEE",
        Tool.EXIT: "EXIT",
    }

    tool_name = tool_names[selected_tool]
    tool_color = TOOL_COLORS[selected_tool]

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

def draw_evacuees(screen, evacuees, drawing_size):

    for evacuee in evacuees:

        pixel_x, pixel_y = grid_to_pixel_center(
            evacuee.grid_x,
            evacuee.grid_y,
            drawing_size
        )

        pygame.draw.circle(
            screen,
            Color.EVACUEE.value,
            (pixel_x, pixel_y),
            drawing_size // 2
        )