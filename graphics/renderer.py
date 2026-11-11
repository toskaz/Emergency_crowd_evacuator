from map.cell import CellType
from graphics.colors import Color
import pygame
from editor.tool import Tool
from grids.cluster import ClusterType
from grids.coordinates import grid_to_pixel, grid_to_pixel_center
import utils.config as config
from utils.config import cluster_size

CLUSTER_COLORS = ((40, 170, 225), (245, 170, 60))

CELL_COLORS = {
    CellType.EMPTY: Color.EMPTY.value,
    CellType.WALL: Color.WALL.value,
    CellType.EXIT: Color.EXIT.value,
    CellType.STAIRS: Color.STAIRS.value,
    CellType.EVACUEE: Color.EMPTY.value,
    CellType.DRONE: Color.EMPTY.value,
}

ENTITY_COLORS = {
    CellType.EVACUEE: Color.EVACUEE.value,
    CellType.DRONE: Color.DRONE.value,
}


TOOL_COLORS = {
    Tool.WALL: Color.WALL.value,
    Tool.EVACUEE: Color.EVACUEE.value,
    Tool.EXIT: Color.EXIT.value,
    Tool.DRONE: Color.DRONE.value,
    Tool.STAIRS: Color.STAIRS.value,
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
                (pos_x, pos_y, config.drawing_size, config.drawing_size)
            )

            marker_color = ENTITY_COLORS.get(value)
            if marker_color is not None:
                pygame.draw.circle(
                    screen,
                    marker_color,
                    grid_to_pixel_center(x, y),
                    config.drawing_size // 2,
                )

            # Draw the grid line
            if show_grid:
                pygame.draw.rect(
                    screen,
                    Color.GRID.value,
                    (pos_x, pos_y, config.drawing_size, config.drawing_size),
                    1
                )


def draw_temporary_walls(screen, grid):
    for y in range(grid.height):
        for x in range(grid.width):
            if not grid.get(x, y):
                continue

            pos_x, pos_y = grid_to_pixel(x, y)
            rect = (pos_x, pos_y, config.drawing_size, config.drawing_size)
            pygame.draw.rect(screen, Color.TEMPORARY_WALL.value, rect)
            pygame.draw.line(
                screen,
                Color.WALL.value,
                (pos_x + 2, pos_y + 2),
                (pos_x + config.drawing_size - 3, pos_y + config.drawing_size - 3),
                2,
            )
            pygame.draw.line(
                screen,
                Color.WALL.value,
                (pos_x + config.drawing_size - 3, pos_y + 2),
                (pos_x + 2, pos_y + config.drawing_size - 3),
                2,
            )


def draw_key_bindings(screen, font, key_bindings):
    if not key_bindings:
        return

    padding = 10
    title = font.render("Controls", True, Color.WALL.value)
    title_rect = title.get_rect(topleft=(padding, padding))
    screen.blit(title, title_rect)

    top = title_rect.bottom + 5
    for line in key_bindings:
        text = font.render(line, True, Color.WALL.value)
        text_rect = text.get_rect(topleft=(padding, top))
        screen.blit(text, text_rect)
        top = text_rect.bottom + 3


def draw_ui(screen, font, selected_tool=None, status_lines=(), key_bindings=(),
            key_font=None):
    tool_names = {
        Tool.WALL: "WALL",
        Tool.EVACUEE: "EVACUEE",
        Tool.EXIT: "EXIT",
        Tool.DRONE: "DRONE",
        Tool.STAIRS: "STAIRS",
    }

    padding = 10
    top = padding
    if selected_tool is not None:
        tool_name = tool_names[selected_tool]
        tool_color = TOOL_COLORS[selected_tool]
        text = font.render(f"Tool: {tool_name}", True, tool_color)
        text_rect = text.get_rect(top=padding, right=screen.get_width() - padding)
        screen.blit(text, text_rect)
        top = text_rect.bottom + 4

    for line in status_lines:
        text = font.render(line, True, Color.WALL.value)
        text_rect = text.get_rect(top=top, right=screen.get_width() - padding)
        screen.blit(text, text_rect)
        top = text_rect.bottom + 4

    draw_key_bindings(screen, key_font or font, key_bindings)


def draw_navigation(screen, floors, drone_routes, floor_id):
    for route in drone_routes.values():
        points = []
        for node in route:
            route_floor_id, cluster_id = node
            if route_floor_id != floor_id:
                points.append(None)
                continue
            points.append(tuple(
                round(value)
                for value in grid_to_pixel_center(
                    *floors[floor_id].cluster_grid.clusters[cluster_id].get_centroid()
                )
            ))

        segment = []
        for point in points + [None]:
            if point is None:
                if len(segment) > 1:
                    pygame.draw.lines(screen, Color.DRONE.value, False, segment, 2)
                segment = []
            else:
                segment.append(point)


def cluster_color(x, y):
    cluster_width = 2 * cluster_size + 1
    color_index = (x // cluster_width + y // cluster_width) % 2
    return CLUSTER_COLORS[color_index]


def draw_cluster_edges(screen, cluster_grid):
    for first_cluster, second_cluster in cluster_grid.adjacent_cluster_pairs:
        first_center = grid_to_pixel_center(
            *cluster_grid.clusters[first_cluster].get_centroid()
        )
        second_center = grid_to_pixel_center(
            *cluster_grid.clusters[second_cluster].get_centroid()
        )
        pygame.draw.line(
            screen,
            Color.DRONE.value,
            tuple(round(value) for value in first_center),
            tuple(round(value) for value in second_center),
            2
        )


def draw_clusters(screen, overlay_grid, alpha=110, font=None):
    cells = overlay_grid.grid

    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA) #need for proper transparency

    for y in range(cells.height):
        for x in range(cells.width):
            cluster_id = cells.get(x, y)
            if cluster_id is None or cluster_id == -1:
                continue

            px, py = grid_to_pixel(x, y)
            color = (
                Color.EXIT.value
                if overlay_grid.clusters[cluster_id].type is ClusterType.EXIT
                else Color.STAIRS.value
                if overlay_grid.clusters[cluster_id].type is ClusterType.STAIRS
                else cluster_color(x, y)
            )

            pygame.draw.rect(overlay,(*color, alpha),(px, py, config.drawing_size, config.drawing_size))

    screen.blit(overlay, (0, 0))

    if font is None:
        font = pygame.font.Font(None, 18)

    for cluster_id, cluster in enumerate(overlay_grid.clusters):
        if not cluster.cells:
            continue

        center_x, center_y = cluster.get_centroid()
        pixel_x, pixel_y = grid_to_pixel_center(center_x, center_y)
        label = font.render(str(cluster_id), True, Color.BACKGROUND.value)
        label_rect = label.get_rect(center=(round(pixel_x), round(pixel_y)))
        screen.blit(label, label_rect)


def draw_stair_highlights(screen, world, font=None, link_mode=False):
    if font is None:
        font = pygame.font.Font(None, 18)

    active_floor = world.active_floor
    active_floor_id = world.active_floor_id

    for cluster_id, cluster in enumerate(active_floor.cluster_grid.clusters):
        if cluster.type is not ClusterType.STAIRS or not cluster.cells:
            continue

        is_pending = (world.pending_stair_endpoint == (active_floor_id, cluster_id))

        min_x = min(x for x, y in cluster.cells)
        min_y = min(y for x, y in cluster.cells)
        max_x = max(x for x, y in cluster.cells)
        max_y = max(y for x, y in cluster.cells)

        rect_x, rect_y = grid_to_pixel(min_x, min_y)
        rect_w = (max_x - min_x + 1) * config.drawing_size
        rect_h = (max_y - min_y + 1) * config.drawing_size
        box = pygame.Rect(rect_x - 2, rect_y - 2, rect_w + 4, rect_h + 4)

        if is_pending:
            pygame.draw.rect(screen, (255, 230, 50), box, 3)
            badge = font.render("WYBRANE", True, (255, 230, 50))
            screen.blit(badge, (box.centerx - badge.get_width() // 2, max(2, box.top - 14)))
        elif link_mode:
            pygame.draw.rect(screen, (80, 200, 255), box, 2)

        if cluster.linked_to is not None:
            target_floor_num = cluster.linked_to[0] + 1
            badge = font.render(f"P{target_floor_num}", True, (100, 255, 140))
            screen.blit(badge, (box.centerx - badge.get_width() // 2, min(screen.get_height() - 14, box.bottom + 2)))