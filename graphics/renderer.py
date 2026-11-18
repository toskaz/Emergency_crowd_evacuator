from collections import deque

from map.cell import CellType
from graphics.colors import Color, assignment_color
import pygame
import pygame.gfxdraw
import math
from editor.tool import Tool
from grids.cluster import ClusterType
from grids.coordinates import grid_to_pixel, grid_to_pixel_center
import utils.config as config
from utils.config import cluster_size, drone_vision_radius

CLUSTER_COLORS = (Color.DRONE.value, Color.STAIRS.value)

CELL_COLORS = {
    CellType.WALL: Color.WALL.value,
    CellType.EXIT: Color.EXIT.value,
    CellType.STAIRS: Color.STAIRS.value,
}


TOOL_COLORS = {
    Tool.WALL: Color.WALL.value,
    Tool.EVACUEE: Color.EVACUEE.value,
    Tool.EXIT: Color.EXIT.value,
    Tool.DRONE: Color.DRONE.value,
    Tool.STAIRS: Color.STAIRS.value,
}

_vision_overlay = None
_vision_overlay_size = None


def draw_map(screen, grid, show_grid):
    screen.fill(Color.EXTERIOR.value)
    exterior = _find_exterior_cells(grid)

    for y in range(grid.height):
        for x in range(grid.width):
            value = grid.get(x, y)
            color = CELL_COLORS.get(value)
            pos_x, pos_y = grid_to_pixel(x, y)
            if color is None:
                if (x, y) not in exterior:
                    pygame.draw.rect(
                        screen,
                        Color.EMPTY.value,
                        (pos_x, pos_y, config.drawing_size, config.drawing_size),
                    )
            else:
                pygame.draw.rect(
                    screen,
                    color,
                    (pos_x, pos_y, config.drawing_size, config.drawing_size),
                )
                if value == CellType.WALL and config.drawing_size >= 5:
                    pygame.draw.line(
                        screen,
                        (226, 232, 240),
                        (pos_x + 1, pos_y + 1),
                        (pos_x + config.drawing_size - 2, pos_y + 1),
                    )
                elif value == CellType.EXIT:
                    _draw_exit_marker(screen, pos_x, pos_y)

            if value == CellType.EVACUEE:
                _draw_evacuee_marker(
                    screen,
                    tuple(round(coord) for coord in grid_to_pixel_center(x, y)),
                )
            elif value == CellType.DRONE:
                _draw_drone_marker(
                    screen,
                    tuple(round(coord) for coord in grid_to_pixel_center(x, y)),
                )

    _draw_exit_labels(screen, grid)

    if show_grid:
        left, top = grid_to_pixel(0, 0)
        right = left + grid.width * config.drawing_size
        bottom = top + grid.height * config.drawing_size
        for x in range(grid.width + 1):
            px = left + x * config.drawing_size
            pygame.draw.line(screen, Color.GRID_BRIGHT.value, (px, top), (px, bottom))
        for y in range(grid.height + 1):
            py = top + y * config.drawing_size
            pygame.draw.line(screen, Color.GRID_BRIGHT.value, (left, py), (right, py))


def _find_exterior_cells(grid):
    exterior = set()
    pending = deque()
    for x in range(grid.width):
        pending.extend(((x, 0), (x, grid.height - 1)))
    for y in range(grid.height):
        pending.extend(((0, y), (grid.width - 1, y)))

    while pending:
        x, y = pending.popleft()
        if (x, y) in exterior or not grid.in_bounds(x, y):
            continue
        if grid.get(x, y) in (CellType.WALL, CellType.EXIT):
            continue
        exterior.add((x, y))
        pending.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
    return exterior


def _draw_exit_marker(screen, pixel_x, pixel_y):
    size = config.drawing_size
    halo = pygame.Surface((size + 4, size + 4), pygame.SRCALPHA)
    halo.fill((*Color.EXIT.value, 55))
    screen.blit(halo, (pixel_x - 2, pixel_y - 2))
    pygame.draw.rect(screen, Color.EXIT.value, (pixel_x, pixel_y, size, size))
    if size >= 4:
        pygame.draw.line(
            screen,
            (167, 243, 208),
            (pixel_x + 1, pixel_y + 1),
            (pixel_x + size - 2, pixel_y + 1),
        )

def _draw_exit_labels(screen, grid):
    remaining = {
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.get(x, y) == CellType.EXIT
    }
    font = pygame.font.Font(None, 14)
    while remaining:
        component = {remaining.pop()}
        pending = deque(component)
        while pending:
            x, y = pending.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        center_x = sum(x for x, _ in component) / len(component)
        center_y = sum(y for _, y in component) / len(component)
        pixel_x, pixel_y = grid_to_pixel_center(center_x, center_y)
        label = font.render("EXIT", True, (167, 243, 208))
        label_rect = label.get_rect(
            midleft=(round(pixel_x + config.drawing_size), round(pixel_y))
        )
        _draw_exit_label_badge(screen, label, label_rect)


def _draw_exit_label_badge(screen, label, label_rect):
    backdrop = label.get_rect().inflate(6, 4)
    backdrop.topleft = (label_rect.x - 3, label_rect.y - 2)
    badge = pygame.Surface(backdrop.size, pygame.SRCALPHA)
    badge.fill((*Color.BACKGROUND.value, 230))
    pygame.draw.rect(badge, Color.EXIT.value, badge.get_rect(), 1, border_radius=3)
    screen.blit(badge, backdrop.topleft)
    screen.blit(label, label_rect)


def draw_entities(screen, simulation, floor_id):
    for evacuee in simulation.evacuees:
        if evacuee.floor_id != floor_id:
            continue
        center = tuple(round(value) for value in grid_to_pixel_center(
            evacuee.grid_x, evacuee.grid_y
        ))
        _draw_evacuee_marker(screen, center)

    drones = [drone for drone in simulation.drones if drone.floor_id == floor_id]
    if drones:
        global _vision_overlay, _vision_overlay_size
        screen_size = screen.get_size()
        if _vision_overlay_size != screen_size:
            _vision_overlay = pygame.Surface(screen_size, pygame.SRCALPHA)
            _vision_overlay_size = screen_size
        else:
            _vision_overlay.fill((0, 0, 0, 0))
        for drone in drones:
            center = tuple(round(value) for value in grid_to_pixel_center(
                drone.grid_x, drone.grid_y
            ))
            pygame.draw.circle(
                _vision_overlay,
                (*Color.DRONE.value, 42),
                center,
                drone_vision_radius * config.drawing_size,
                1,
            )
        screen.blit(_vision_overlay, (0, 0))

    for drone in drones:
        center_x, center_y = (
            round(value)
            for value in grid_to_pixel_center(drone.grid_x, drone.grid_y)
        )
        _draw_drone_marker(screen, (center_x, center_y))


def _draw_evacuee_marker(screen, center):
    radius = max(3, config.drawing_size // 3)
    pygame.gfxdraw.filled_circle(
        screen, *center, radius + 1, Color.BACKGROUND.value
    )
    pygame.gfxdraw.aacircle(screen, *center, radius + 1, Color.BACKGROUND.value)
    pygame.gfxdraw.filled_circle(screen, *center, radius, Color.EVACUEE.value)
    pygame.gfxdraw.aacircle(screen, *center, radius, Color.EVACUEE.value)


def _draw_drone_marker(screen, center):
    center_x, center_y = center
    radius = max(4, config.drawing_size // 2)
    points = [
        (center_x, center_y - radius),
        (center_x + radius, center_y),
        (center_x, center_y + radius),
        (center_x - radius, center_y),
    ]
    pygame.gfxdraw.filled_polygon(screen, points, Color.DRONE.value)
    pygame.gfxdraw.aapolygon(screen, points, Color.DRONE.value)


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
    size = config.drawing_size
    hazard = pygame.Surface((size, size), pygame.SRCALPHA)
    hazard.fill((*Color.TEMPORARY_WALL.value, 95))
    pygame.draw.line(
        hazard,
        (*Color.TEMPORARY_WALL.value, 180),
        (1, 1),
        (size - 2, size - 2),
        1,
    )
    pygame.draw.line(
        hazard,
        (*Color.TEMPORARY_WALL.value, 180),
        (size - 2, 1),
        (1, size - 2),
        1,
    )
    for y in range(grid.height):
        for x in range(grid.width):
            if not grid.get(x, y):
                continue

            pos_x, pos_y = grid_to_pixel(x, y)
            screen.blit(hazard, (pos_x, pos_y))


def draw_key_bindings(screen, font, key_bindings):
    if not key_bindings:
        return

    padding_x = 16
    line_h = 18
    total_h = len(key_bindings) * line_h + 8
    start_y = screen.get_height() - total_h - 12

    badge_surface = pygame.Surface((260, total_h), pygame.SRCALPHA)
    badge_surface.fill((*Color.BACKGROUND.value, 220))
    pygame.draw.rect(
        badge_surface,
        (*Color.CARD_BORDER.value, 220),
        badge_surface.get_rect(),
        1,
        border_radius=6,
    )
    screen.blit(badge_surface, (padding_x, start_y))

    cur_y = start_y + 4
    for line in key_bindings:
        text = font.render(line, True, Color.TEXT_PRIMARY.value)
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
                color = Color.EXIT.value
            elif display_text.startswith("Error:") or "Error" in display_text:
                color = Color.TEMPORARY_WALL.value
            else:
                color = Color.DRONE.value
            text = (key_font or font).render(display_text, True, Color.TEXT_PRIMARY.value)
            pad_x, pad_y = 12, 6
            pill_rect = pygame.Rect(
                canvas_center_x - text.get_width() // 2 - pad_x,
                banner_y,
                text.get_width() + pad_x * 2,
                text.get_height() + pad_y * 2
            )
            pill_surf = pygame.Surface((pill_rect.width, pill_rect.height), pygame.SRCALPHA)
            pill_surf.fill((*Color.CARD_BG.value, 235))
            pygame.draw.rect(pill_surf, color, pill_surf.get_rect(), 1, border_radius=6)
            screen.blit(pill_surf, pill_rect.topleft)
            screen.blit(text, (pill_rect.x + pad_x, pill_rect.y + pad_y))
            banner_y += pill_rect.height + 6



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
                    for start, end in zip(segment, segment[1:]):
                        _draw_dashed_segment(
                            screen, start, end, Color.NAVIGATION.value
                        )
                segment = []
            else:
                segment.append(point)


def _draw_dashed_segment(screen, start, end, color, dash_length=8, gap_length=5):
    start_x, start_y = start
    end_x, end_y = end
    dx, dy = end_x - start_x, end_y - start_y
    length = math.hypot(dx, dy)
    if length == 0:
        return
    direction_x, direction_y = dx / length, dy / length
    offset = 0.0
    while offset < length:
        dash_end = min(offset + dash_length, length)
        pygame.draw.aaline(
            screen,
            color,
            (start_x + direction_x * offset, start_y + direction_y * offset),
            (start_x + direction_x * dash_end, start_y + direction_y * dash_end),
        )
        offset += dash_length + gap_length


def draw_results_chart(screen, simulation, font, small_font):
    canvas_width = screen.get_width() - 280
    panel_width = min(680, canvas_width - 48)
    panel_height = min(310, screen.get_height() - 120)
    if panel_width < 360 or panel_height < 220:
        return

    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((*Color.BACKGROUND.value, 238))
    pygame.draw.rect(
        panel, Color.CARD_BORDER.value, panel.get_rect(), 1, border_radius=8
    )

    history = simulation.evacuation_history or [(0, 0)]
    total = simulation.evacuated_count + len(simulation.evacuees)
    evacuation_rate = (
        100.0 * simulation.evacuated_count / total if total else 0.0
    )
    average_time = (
        sum(simulation.evacuation_times) / len(simulation.evacuation_times)
        if simulation.evacuation_times
        else None
    )
    title = font.render("EVACUATION PROGRESS", True, Color.TEXT_PRIMARY.value)
    panel.blit(title, (18, 12))

    metrics = (
        ("TOTAL TICKS", str(simulation.step_count)),
        ("EVACUATION RATE", f"{evacuation_rate:.1f}%"),
        (
            "AVG. EVACUATION TIME",
            f"{average_time:.1f} ticks" if average_time is not None else "--",
        ),
    )
    metric_width = (panel_width - 36) // len(metrics)
    for index, (label, value) in enumerate(metrics):
        x = 18 + index * metric_width
        panel.blit(small_font.render(label, True, Color.TEXT_MUTED.value), (x, 43))
        panel.blit(font.render(value, True, Color.DRONE.value), (x, 59))

    plot_left, plot_top = 54, 105
    plot_right, plot_bottom = panel_width - 22, panel_height - 34
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    maximum_step = max(1, history[-1][0])
    maximum_count = max(1, total, max(count for _, count in history))

    for division in range(5):
        y = round(plot_bottom - plot_height * division / 4)
        pygame.draw.line(
            panel, Color.GRID.value, (plot_left, y), (plot_right, y), 1
        )
        count_label = round(maximum_count * division / 4)
        panel.blit(
            small_font.render(str(count_label), True, Color.TEXT_MUTED.value),
            (8, y - 7),
        )

    points = [
        (
            round(plot_left + plot_width * step / maximum_step),
            round(plot_bottom - plot_height * count / maximum_count),
        )
        for step, count in history
    ]
    if len(points) > 1:
        pygame.draw.aalines(panel, Color.EXIT.value, False, points)
    if points:
        pygame.gfxdraw.filled_circle(
            panel, points[-1][0], points[-1][1], 3, Color.EXIT.value
        )
        pygame.gfxdraw.aacircle(
            panel, points[-1][0], points[-1][1], 3, Color.EXIT.value
        )

    panel.blit(
        small_font.render("TICK", True, Color.TEXT_MUTED.value),
        (plot_right - 30, plot_bottom + 8),
    )
    screen.blit(panel, (24, 108))


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


def draw_clusters(screen, overlay_grid, alpha=42, font=None):
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

    if config.drawing_size < 18:
        return
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
            box_color = Color.EVACUEE.value
            box_width = 3
            badge_text = f"C{cluster_id}: SELECTED"
        elif cluster.linked_to is not None:
            box_color = Color.EXIT.value
            box_width = 2
            t_floor, t_cid = cluster.linked_to
            badge_text = f"C{cluster_id} -> F{t_floor + 1}:C{t_cid}"
        else:
            box_color = Color.DRONE.value if link_mode else Color.TEMPORARY_WALL.value
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
        pill_surf.fill((*Color.BACKGROUND.value, 230))
        pygame.draw.rect(
            pill_surf, Color.CARD_BORDER.value, pill_surf.get_rect(), 1, border_radius=4
        )
        screen.blit(pill_surf, pill_rect.topleft)
        screen.blit(badge_surf, (badge_x + pad_x, badge_y + pad_y))