from map.cell import CellType
from colors import Color
import pygame
from editor.tool import Tool
from coordinates import grid_to_pixel, grid_to_pixel_center
from config import drawing_size

CELL_COLORS = {
    CellType.EMPTY: Color.EMPTY.value,
    CellType.WALL: Color.WALL.value,
    CellType.EXIT: Color.EXIT.value,
}


TOOL_COLORS = {
    Tool.WALL: Color.WALL.value,
    Tool.EVACUEE: Color.EVACUEE.value,
    Tool.EXIT: Color.EXIT.value,
    Tool.DRONE: Color.DRONE.value,
}


def draw_map(screen, grid, show_grid):
    screen.fill(Color.BACKGROUND.value)

    for y in range(grid.height):
        for x in range(grid.width):
            value = grid.get(x, y)

            pos_x, pos_y = grid_to_pixel(x, y)
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


def draw_ui(screen, font, selected_tool, status_lines=()):
    tool_names = {
        Tool.WALL: "WALL",
        Tool.EVACUEE: "EVACUEE",
        Tool.EXIT: "EXIT",
        Tool.DRONE: "DRONE",
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

    for line in status_lines:
        text = font.render(line, True, Color.WALL.value)

        text_rect = text.get_rect(
            top=text_rect.bottom + 4,
            right=screen.get_width() - padding
        )

        screen.blit(text, text_rect)


def draw_exit_capacities(screen, font, exits):

    for exit_ in exits:
        center_x = sum(x for x, _ in exit_.cells) / len(exit_.cells)
        center_y = sum(y for _, y in exit_.cells) / len(exit_.cells)

        text = font.render(f"{exit_.capacity:g}", True, Color.WALL.value)
        text_rect = text.get_rect(center=grid_to_pixel_center(center_x, center_y))
        text_rect.clamp_ip(screen.get_rect().inflate(-4, -2))

        pygame.draw.rect(screen, Color.EXIT.value, text_rect.inflate(4, 2))
        screen.blit(text, text_rect)


def draw_navigation(screen, navigation, drone_routes):

    for u, v, _ in navigation.edges:
        pygame.draw.line(
            screen,
            Color.NAVIGATION.value,
            grid_to_pixel_center(*navigation.representatives[u]),
            grid_to_pixel_center(*navigation.representatives[v])
        )

    for cell in navigation.representatives:
        pygame.draw.circle(screen, Color.NAVIGATION.value, grid_to_pixel_center(*cell), 2)

    for route in drone_routes.values():
        pygame.draw.lines(
            screen,
            Color.DRONE.value,
            False,
            [grid_to_pixel_center(*cell) for cell in route],
            2
        )


def draw_evacuees(screen, evacuees):

    for evacuee in evacuees:

        pixel_x, pixel_y = grid_to_pixel_center(evacuee.grid_x,evacuee.grid_y)

        pygame.draw.circle(
            screen,
            Color.EVACUEE.value,
            (pixel_x, pixel_y),
            drawing_size // 2
        )


def draw_drones(screen, drones):

    for drone in drones:

        pixel_x, pixel_y = grid_to_pixel_center(drone.grid_x, drone.grid_y)

        pygame.draw.circle(
            screen,
            Color.DRONE.value,
            (pixel_x, pixel_y),
            drawing_size // 2
        )