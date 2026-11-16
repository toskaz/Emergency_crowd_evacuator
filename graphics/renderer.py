from map.cell import CellType
from graphics.colors import Color, assignment_color
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
            color = CELL_COLORS.get(value, Color.EMPTY.value)

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


def draw_assignment_groups(screen, simulation, floor_id):
    for evacuee in simulation.evacuees:
        if evacuee.floor_id != floor_id:
            continue
        pygame.draw.circle(
            screen,
            assignment_color(evacuee.assigned_drone_id),
            grid_to_pixel_center(evacuee.grid_x, evacuee.grid_y),
            config.drawing_size // 2,
        )

    for drone in simulation.drones:
        if drone.floor_id != floor_id:
            continue
        pixel_x, pixel_y = grid_to_pixel(drone.grid_x, drone.grid_y)
        rect = pygame.Rect(pixel_x, pixel_y, config.drawing_size, config.drawing_size)
        pygame.draw.rect(screen, assignment_color(drone.drone_id), rect)
        pygame.draw.rect(screen, Color.WALL.value, rect, 1)


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

    padding_x = 16
    line_h = 18
    total_h = len(key_bindings) * line_h + 8
    start_y = screen.get_height() - total_h - 12

    badge_surface = pygame.Surface((260, total_h), pygame.SRCALPHA)
    badge_surface.fill((25, 25, 25, 180))
    pygame.draw.rect(badge_surface, (55, 55, 55, 180), badge_surface.get_rect(), 1, border_radius=6)
    screen.blit(badge_surface, (padding_x, start_y))

    cur_y = start_y + 4
    for line in key_bindings:
        text = font.render(line, True, (180, 180, 180))
        screen.blit(text, (padding_x + 8, cur_y))
        cur_y += line_h


def draw_ui(screen, font, selected_tool=None, status_lines=(), key_bindings=(),
            key_font=None):
    alert_lines = [
        line for line in status_lines
        if line.startswith(" ") or line.startswith("★") or line.startswith("Error:") or line.startswith("Notice:")
    ]
    if alert_lines:
        screen_w = screen.get_width()
        canvas_center_x = (screen_w - 280) // 2
        banner_y = 96
        for line in alert_lines:
            display_text = line.strip()
            if not display_text:
                continue
            if display_text.startswith("Notice:"):
                color = (16, 185, 129)
            elif display_text.startswith("Error:") or "Error" in display_text:
                color = (245, 158, 11)
            else:
                color = (59, 130, 246)
            text = (key_font or font).render(display_text, True, (240, 240, 240))
            pad_x, pad_y = 12, 6
            pill_rect = pygame.Rect(
                canvas_center_x - text.get_width() // 2 - pad_x,
                banner_y,
                text.get_width() + pad_x * 2,
                text.get_height() + pad_y * 2
            )
            pill_surf = pygame.Surface((pill_rect.width, pill_rect.height), pygame.SRCALPHA)
            pill_surf.fill((30, 30, 30, 220))
            pygame.draw.rect(pill_surf, color, pill_surf.get_rect(), 1, border_radius=6)
            screen.blit(pill_surf, pill_rect.topleft)
            screen.blit(text, (pill_rect.x + pad_x, pill_rect.y + pad_y))
            banner_y += pill_rect.height + 6

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

    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

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
                else Color.DOOR.value
                if overlay_grid.clusters[cluster_id].type is ClusterType.DOOR
                else cluster_color(x, y)
            )

            pygame.draw.rect(overlay, (*color, alpha), (px, py, config.drawing_size, config.drawing_size))

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
        font = pygame.font.Font(None, 16)

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
            box_color = (255, 230, 50)
            box_width = 3
            badge_text = f"C{cluster_id}: SELECTED"
        elif cluster.linked_to is not None:
            box_color = (16, 185, 129)
            box_width = 2
            t_floor, t_cid = cluster.linked_to
            badge_text = f"C{cluster_id} -> F{t_floor + 1}:C{t_cid}"
        else:
            box_color = (80, 200, 255) if link_mode else (239, 68, 68)
            box_width = 2
            badge_text = f"C{cluster_id}: UNLINKED"

        pygame.draw.rect(screen, box_color, box, box_width)

        badge_surf = font.render(badge_text, True, box_color)
        pad_x, pad_y = 6, 2
        badge_w = badge_surf.get_width() + pad_x * 2
        badge_h = badge_surf.get_height() + pad_y * 2

        badge_x = box.centerx - badge_w // 2
        badge_y = box.top - badge_h - 2
        if badge_y < 90:
            badge_y = box.bottom + 2

        pill_rect = pygame.Rect(badge_x, badge_y, badge_w, badge_h)
        pill_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pill_surf.fill((20, 20, 20, 230))
        pygame.draw.rect(pill_surf, (45, 45, 45), pill_surf.get_rect(), 1, border_radius=4)
        screen.blit(pill_surf, pill_rect.topleft)
        screen.blit(badge_surf, (badge_x + pad_x, badge_y + pad_y))