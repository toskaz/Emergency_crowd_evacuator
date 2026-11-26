from map.cell import CellType
from graphics.colors import Color, assignment_color
import pygame
import pygame.gfxdraw
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


def draw_map(screen, grid, show_grid, plan_overlay_surface=None):
    screen.fill(Color.BACKGROUND.value)

    d_size = config.drawing_size
    px0, py0 = grid_to_pixel(0, 0)
    map_rect = pygame.Rect(px0, py0, grid.width * d_size, grid.height * d_size)
    pygame.draw.rect(screen, Color.EMPTY.value, map_rect)

    if plan_overlay_surface is not None:
        target_w = max(1, grid.width * d_size)
        target_h = max(1, grid.height * d_size)
        if plan_overlay_surface.get_size() != (target_w, target_h):
            scaled = pygame.transform.smoothscale(plan_overlay_surface, (target_w, target_h))
        else:
            scaled = plan_overlay_surface
        scaled.set_alpha(72)
        screen.blit(scaled, (px0, py0))

    for y in range(grid.height):
        for x in range(grid.width):
            value = grid.get(x, y)
            if value == CellType.EMPTY:
                continue

            pos_x, pos_y = grid_to_pixel(x, y)
            color = CELL_COLORS.get(value, Color.EMPTY.value)

            if value not in (CellType.EVACUEE, CellType.DRONE):
                pygame.draw.rect(
                    screen,
                    color,
                    (pos_x, pos_y, d_size, d_size)
                )

            marker_color = ENTITY_COLORS.get(value)
            if marker_color is not None:
                cx, cy = grid_to_pixel_center(x, y)
                radius = max(2, d_size // 2)
                pygame.draw.circle(screen, marker_color, (cx, cy), radius)

    if show_grid:
        end_x = px0 + grid.width * d_size
        end_y = py0 + grid.height * d_size
        for x in range(grid.width + 1):
            lx = px0 + x * d_size
            pygame.draw.line(screen, Color.GRID.value, (lx, py0), (lx, end_y), 1)
        for y in range(grid.height + 1):
            ly = py0 + y * d_size
            pygame.draw.line(screen, Color.GRID.value, (px0, ly), (end_x, ly), 1)


def draw_assignment_groups(screen, simulation, floor_id):
    for evacuee in simulation.evacuees:
        if evacuee.floor_id != floor_id:
            continue
        cx, cy = grid_to_pixel_center(evacuee.grid_x, evacuee.grid_y)
        radius = max(2, config.drawing_size // 2)
        color = assignment_color(evacuee.assigned_drone_id)
        try:
            pygame.gfxdraw.filled_circle(screen, cx, cy, radius, color)
            pygame.gfxdraw.aacircle(screen, cx, cy, radius, color)
        except Exception:
            pygame.draw.circle(screen, color, (cx, cy), radius)

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


def draw_map_scale(screen, font, top_left=None, bottom_left=None, cell_size_m=0.6):
    box_w = 154
    box_h = 56

    cells_count = 5
    max_bar_px = box_w - 44
    if cells_count * config.drawing_size > max_bar_px:
        cells_count = max(1, max_bar_px // config.drawing_size)
    real_dist_m = cells_count * cell_size_m
    line_px = int(cells_count * config.drawing_size)

    if top_left is not None:
        x, y = top_left
    elif bottom_left is not None:
        x, bl_y = bottom_left
        y = bl_y - box_h
    else:
        x, y = 14, 240

    bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    bg.fill((20, 20, 20, 220))
    pygame.draw.rect(bg, (55, 55, 55, 220), bg.get_rect(), 1, border_radius=6)
    screen.blit(bg, (x, y))

    if cell_size_m >= 1.0:
        scale_title = f"SCALE: 1 cell = {cell_size_m:.2f}m"
    else:
        scale_title = f"SCALE: 1 cell = {cell_size_m * 100:.0f}cm"
    title_lbl = font.render(scale_title, True, (220, 220, 220))
    screen.blit(title_lbl, (x + 14, y + 6))

    bar_x = x + 14
    bar_y = y + 28
    bar_h = 6

    mid_px = line_px // 2
    pygame.draw.rect(screen, (240, 240, 240), (bar_x, bar_y, mid_px, bar_h))
    pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, mid_px, bar_h), 1)

    pygame.draw.rect(screen, (40, 40, 40), (bar_x + mid_px, bar_y, line_px - mid_px, bar_h))
    pygame.draw.rect(screen, (200, 200, 200), (bar_x + mid_px, bar_y, line_px - mid_px, bar_h), 1)

    pygame.draw.line(screen, (240, 240, 240), (bar_x, bar_y - 2), (bar_x, bar_y + bar_h + 2), 1)
    pygame.draw.line(screen, (240, 240, 240), (bar_x + mid_px, bar_y - 2), (bar_x + mid_px, bar_y + bar_h + 2), 1)
    pygame.draw.line(screen, (240, 240, 240), (bar_x + line_px, bar_y - 2), (bar_x + line_px, bar_y + bar_h + 2), 1)

    def _fmt_dist(v):
        if v < 1.0:
            return f"{v * 100:.0f}cm"
        if abs(round(v, 2) - round(v, 1)) < 1e-4:
            return f"{v:.1f}m"
        return f"{v:.2f}m"

    t0 = font.render("0", True, (220, 220, 220))
    t_end = font.render(_fmt_dist(real_dist_m), True, (220, 220, 220))

    screen.blit(t0, (bar_x - t0.get_width() // 2, bar_y + bar_h + 3))
    screen.blit(t_end, (bar_x + line_px - t_end.get_width() // 2, bar_y + bar_h + 3))

    if line_px >= 90:
        mid_val = real_dist_m / 2
        t_mid = font.render(_fmt_dist(mid_val), True, (180, 180, 180))
        screen.blit(t_mid, (bar_x + mid_px - t_mid.get_width() // 2, bar_y + bar_h + 3))

    return box_h


def draw_heatmap(screen, floor):
    if floor is None or not hasattr(floor, "heatmap_grid") or floor.heatmap_grid is None:
        return
    grid = floor.heatmap_grid
    overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    d_size = config.drawing_size

    max_val = 1
    for row in grid.cells:
        for val in row:
            if val and val > max_val:
                max_val = val

    r, g, b = Color.HEATMAP.value

    for y in range(grid.height):
        for x in range(grid.width):
            val = grid.get(x, y)
            if not val or val <= 0:
                continue
            px, py = grid_to_pixel(x, y)
            ratio = min(1.0, val / max(5, max_val))
            alpha = int(40 + 175 * (ratio ** 0.65))
            pygame.draw.rect(overlay, (r, g, b, alpha), (px, py, d_size, d_size))

    screen.blit(overlay, (0, 0))


def draw_heatmap_legend(screen, font, top_left=None, bottom_left=None):
    box_w = 154
    box_h = 52
    if top_left is not None:
        x, y = top_left
    elif bottom_left is not None:
        x, bl_y = bottom_left
        y = bl_y - box_h
    else:
        x, y = 14, 304

    bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    bg.fill((20, 20, 20, 220))
    pygame.draw.rect(bg, (55, 55, 55, 220), bg.get_rect(), 1, border_radius=6)
    screen.blit(bg, (x, y))

    title = font.render("HEATMAP", True, (170, 170, 170))
    screen.blit(title, (x + 12, y + 6))

    bar_x = x + 12
    bar_y = y + 23
    bar_w = box_w - 24
    bar_h = 8

    r, g, b = Color.HEATMAP.value
    grad_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
    for col in range(bar_w):
        col_ratio = col / max(1, bar_w - 1)
        col_alpha = int(40 + 175 * (col_ratio ** 0.65))
        pygame.draw.line(grad_surf, (r, g, b, col_alpha), (col, 0), (col, bar_h))
    screen.blit(grad_surf, (bar_x, bar_y))
    pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h), 1)

    lbl_low = font.render("Low", True, (210, 210, 210))
    lbl_high = font.render("High traffic", True, (210, 210, 210))
    screen.blit(lbl_low, (bar_x, bar_y + bar_h + 3))
    screen.blit(lbl_high, (bar_x + bar_w - lbl_high.get_width(), bar_y + bar_h + 3))

    return box_h


def draw_legend(screen, font, top_left=None, bottom_left=None):
    legend_items = [
        ("Wall", Color.WALL.value, False),
        ("Exit", Color.EXIT.value, False),
        ("Stairs", Color.STAIRS.value, False),
        ("Evacuee", Color.EVACUEE.value, True),
        ("Guide Agent", Color.DRONE.value, True),
        ("Hazard", Color.TEMPORARY_WALL.value, False),
    ]

    pad_x, pad_y = 12, 8
    item_h = 20
    total_h = len(legend_items) * item_h + pad_y * 2 + 14
    total_w = 154

    if top_left is not None:
        x, y = top_left
    elif bottom_left is not None:
        x, bl_y = bottom_left
        y = bl_y - total_h
    else:
        x, y = 14, 96

    bg = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
    bg.fill((22, 22, 22, 215))
    pygame.draw.rect(bg, (55, 55, 55, 215), bg.get_rect(), 1, border_radius=6)
    screen.blit(bg, (x, y))

    title = font.render("LEGEND", True, (170, 170, 170))
    screen.blit(title, (x + pad_x, y + 6))

    cur_y = y + 24
    for name, color, is_circle in legend_items:
        center_y = cur_y + item_h // 2
        if is_circle:
            pygame.draw.circle(screen, color, (x + pad_x + 5, center_y), 5)
            pygame.draw.circle(screen, (255, 255, 255), (x + pad_x + 5, center_y), 5, 1)
        elif name == "Hazard":
            h_rect = pygame.Rect(x + pad_x, center_y - 5, 11, 11)
            pygame.draw.rect(screen, color, h_rect, border_radius=2)
            pygame.draw.line(screen, Color.WALL.value, (h_rect.x + 1, h_rect.y + 1), (h_rect.right - 2, h_rect.bottom - 2), 1)
            pygame.draw.line(screen, Color.WALL.value, (h_rect.right - 2, h_rect.y + 1), (h_rect.x + 1, h_rect.bottom - 2), 1)
            pygame.draw.rect(screen, (160, 60, 40), h_rect, 1, border_radius=2)
        else:
            pygame.draw.rect(screen, color, (x + pad_x, center_y - 5, 11, 11), border_radius=2)
            pygame.draw.rect(screen, (80, 80, 80), (x + pad_x, center_y - 5, 11, 11), 1, border_radius=2)

        txt = font.render(name, True, (225, 225, 225))
        txt_y = center_y - txt.get_height() // 2
        screen.blit(txt, (x + pad_x + 20, txt_y))
        cur_y += item_h

    return total_h


def draw_building_elevation(screen, font, small_font, world, top_left=None, bottom_left=None, min_top_y=None):
    if world is None or len(world.floors) <= 1:
        return 0

    num_floors = len(world.floors)
    box_w = 154
    floor_row_h = 32
    box_h = 42 + (num_floors - 1) * floor_row_h + 18

    if bottom_left is not None:
        x, bl_y = bottom_left
        y = bl_y - box_h
    elif top_left is not None:
        x, y = top_left
    else:
        x = 14
        y = screen.get_height() - box_h - 14
        if min_top_y is not None and y < min_top_y:
            y = min_top_y

    bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    bg.fill((20, 20, 20, 225))
    pygame.draw.rect(bg, (55, 55, 55, 225), bg.get_rect(), 1, border_radius=7)
    screen.blit(bg, (x, y))

    title = small_font.render("BUILDING PROFILE", True, (170, 170, 170))
    screen.blit(title, (x + 12, y + 8))

    active_id = getattr(world, "active_floor_id", 0)

    line_start_x = x + 38
    line_end_x = x + box_w - 14
    base_line_y = y + box_h - 18

    for f_idx in range(num_floors):
        f_id = f_idx
        line_y = base_line_y - f_idx * floor_row_h
        is_active = (f_id == active_id)

        if is_active:
            pygame.draw.line(screen, (16, 185, 129), (line_start_x, line_y), (line_end_x, line_y), 3)
            lbl = font.render(f"F{f_id + 1}", True, (16, 185, 129))
            screen.blit(lbl, (x + 12, line_y - lbl.get_height() // 2))
        else:
            pygame.draw.line(screen, (90, 90, 90), (line_start_x, line_y), (line_end_x, line_y), 1)
            lbl = small_font.render(f"F{f_id + 1}", True, (160, 160, 160))
            screen.blit(lbl, (x + 12, line_y - lbl.get_height() // 2))

        if f_idx < num_floors - 1:
            mid_y = line_y - floor_row_h // 2
            floors_map = getattr(world, "floors", {})
            if isinstance(floors_map, dict):
                floor_obj = floors_map.get(f_idx + 1)
            elif isinstance(floors_map, (list, tuple)) and f_idx + 1 < len(floors_map):
                floor_obj = floors_map[f_idx + 1]
            else:
                floor_obj = None
            h_m = getattr(floor_obj, "floor_height_m", 3.0) if floor_obj else 3.0
            h_str = f"{h_m:.1f} m"
            h_lbl = small_font.render(h_str, True, (190, 190, 190))
            screen.blit(h_lbl, (line_start_x + (line_end_x - line_start_x - h_lbl.get_width()) // 2, mid_y - h_lbl.get_height() // 2))

    return box_h


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
        banner_y = 128
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


def draw_stair_highlights(screen, world, font=None, link_mode=False, is_sim=False, simulation=None):
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

        s_badge = world.get_stair_badge(active_floor_id, cluster_id)

        if is_sim:
            in_transit = 0
            if simulation:
                for e in simulation.evacuees:
                    if getattr(e, "in_stair_transit", False):
                        target = getattr(e, "stair_target_cluster", None)
                        origin = getattr(e, "stair_origin_cluster", None)
                        if e.floor_id == active_floor_id and (target == cluster_id or origin == cluster_id):
                            in_transit += 1

            if in_transit > 0:
                box_color = (245, 158, 11)
                box_width = 2
                badge_text = f"{s_badge} ({in_transit})"
            elif cluster.linked_to is not None:
                box_color = (16, 185, 129)
                box_width = 1
                badge_text = s_badge
            else:
                box_color = (120, 120, 120)
                box_width = 1
                badge_text = s_badge
        else:
            if is_pending:
                box_color = (255, 230, 50)
                box_width = 3
                badge_text = f"{s_badge}: SELECTED"
            elif cluster.linked_to is not None:
                box_color = (16, 185, 129)
                box_width = 2
                t_floor, t_cid = cluster.linked_to
                t_badge = world.get_stair_badge(t_floor, t_cid)
                badge_text = f"{s_badge} <-> F{t_floor + 1}:{t_badge}"
            else:
                box_color = (80, 200, 255) if link_mode else (239, 68, 68)
                box_width = 2
                badge_text = f"{s_badge}: UNLINKED"

        pygame.draw.rect(screen, box_color, box, box_width)

        badge_surf = font.render(badge_text, True, (255, 255, 255))
        badge_w = max(24, badge_surf.get_width() + 8)
        badge_h = 16

        badge_x = box.centerx - badge_w // 2
        badge_y = box.top - badge_h - 2
        if badge_y < 90:
            badge_y = box.bottom + 2

        pill_rect = pygame.Rect(badge_x, badge_y, badge_w, badge_h)
        pill_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pill_surf.fill((20, 24, 30, 175))
        pygame.draw.rect(pill_surf, box_color, pill_surf.get_rect(), 1, border_radius=4)
        screen.blit(pill_surf, pill_rect.topleft)
        tx = badge_x + (badge_w - badge_surf.get_width()) // 2
        ty = badge_y + (badge_h - badge_surf.get_height()) // 2
        screen.blit(badge_surf, (tx, ty))


def draw_exit_badges(screen, world, font=None):
    if font is None:
        font = pygame.font.Font(None, 15)

    active_floor = world.active_floor
    active_floor_id = world.active_floor_id

    for cluster_id, cluster in enumerate(active_floor.cluster_grid.clusters):
        if cluster.type is not ClusterType.EXIT or not cluster.cells:
            continue

        min_x = min(x for x, y in cluster.cells)
        min_y = min(y for x, y in cluster.cells)
        max_x = max(x for x, y in cluster.cells)
        max_y = max(y for x, y in cluster.cells)

        rect_x, rect_y = grid_to_pixel(min_x, min_y)
        rect_w = (max_x - min_x + 1) * config.drawing_size
        rect_h = (max_y - min_y + 1) * config.drawing_size
        box = pygame.Rect(rect_x - 1, rect_y - 1, rect_w + 2, rect_h + 2)

        e_badge = world.get_exit_badge(active_floor_id, cluster_id)

        badge_surf = font.render(e_badge, True, (255, 255, 255))
        badge_w = max(24, badge_surf.get_width() + 8)
        badge_h = 16

        badge_x = box.centerx - badge_w // 2
        badge_y = box.top - badge_h - 2
        if badge_y < 90:
            badge_y = box.bottom + 2

        pill_rect = pygame.Rect(badge_x, badge_y, badge_w, badge_h)
        pill_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pill_surf.fill((20, 24, 30, 175))
        pygame.draw.rect(pill_surf, (16, 185, 129), pill_surf.get_rect(), 1, border_radius=4)
        screen.blit(pill_surf, pill_rect.topleft)
        tx = badge_x + (badge_w - badge_surf.get_width()) // 2
        ty = badge_y + (badge_h - badge_surf.get_height()) // 2
        screen.blit(badge_surf, (tx, ty))