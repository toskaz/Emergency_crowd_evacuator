from asyncio import selector_events
from pathlib import Path
from asyncio import selector_events
import pygame
from graphics.colors import Color
from grids.cluster import ClusterType
from graphics.renderer import (
    draw_map,
    draw_ui,
    draw_key_bindings,
    draw_navigation,
    draw_clusters,
    draw_cluster_edges,
    draw_temporary_walls,
    draw_stair_highlights,
    draw_assignment_groups,
)


class Window:

    def __init__(self, grid):
        pygame.init()

        self.screen = pygame.display.get_surface()
        if self.screen is None:
            from utils.config import drawing_size
            self.screen = pygame.display.set_mode((1280, 800))

        self.title_font = pygame.font.Font(None, 20)
        self.font = pygame.font.Font(None, 18)
        self.small_font = pygame.font.Font(None, 16)
        self.tiny_font = pygame.font.Font(None, 14)
        self.is_open = True

    def draw_editor(self, context, selected_tool, key_bindings):
        self._draw_view(
            context,
            context.simulation.grid_world.map_grid,
            key_bindings,
            selected_tool,
        )

    def draw_overview(self, context, key_bindings):
        self._draw_view(
            context,
            context.simulation.grid_world.map_grid,
            key_bindings,
        )

    def draw_simulation(self, context, key_bindings):
        self._draw_view(
            context,
            context.simulation.grid_world.simulation_grid,
            key_bindings,
        )

    def _draw_view(self, context, grid, key_bindings, selected_tool=None):
        screen_w, screen_h = self.screen.get_size()
        sidebar_w = 280
        top_bar_h = 88
        canvas_w = max(100, screen_w - sidebar_w)
        canvas_h = max(100, screen_h - top_bar_h)
        canvas_rect = pygame.Rect(0, top_bar_h, canvas_w, canvas_h)

        self.screen.set_clip(canvas_rect)
        self._draw_grid(context, grid)
        if context.simulation and context.simulation.started:
            draw_temporary_walls(
                self.screen, context.simulation.grid_world.temporary_wall_grid
            )
        if context.state.render_options_enabled:
            self._draw_overlays(context)
        self._draw_ui(context, key_bindings, selected_tool)
        self.screen.set_clip(None)

        self._draw_figma_chrome(context, selected_tool)

    def _draw_grid(self, context, grid):
        draw_map(self.screen, grid, context.view_options.show_grid)

    def _draw_overlays(self, context):
        world = context.simulation.grid_world
        options = context.view_options
        if options.show_cluster_connections:
            draw_cluster_edges(self.screen, world.cluster_grid)
        if options.show_clusters:
            draw_clusters(self.screen, world.cluster_grid, font=self.small_font)
        draw_stair_highlights(
            self.screen,
            world,
            font=self.small_font,
            link_mode=context.link_stairs_mode,
        )
        if options.show_drone_routes:
            draw_navigation(
                self.screen,
                world.floors,
                context.drone_routes,
                world.active_floor_id,
            )
        if options.show_assignments:
            draw_assignment_groups(
                self.screen, context.simulation, world.active_floor_id
            )

    def _draw_ui(self, context, key_bindings, selected_tool=None):
        draw_ui(
            self.screen,
            self.font,
            selected_tool,
            context.state.status_lines(context),
            key_bindings,
            self.small_font,
        )

    def _draw_figma_chrome(self, context, selected_tool=None):
        screen = self.screen
        screen_w, screen_h = screen.get_size()
        sidebar_x = screen_w - 280

        pygame.draw.rect(screen, Color.TOPNAV_BG.value, (0, 0, screen_w, 48))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 47), (screen_w, 47), 1)

        mark_rect = pygame.Rect(16, 12, 24, 24)
        
        pygame.draw.rect(screen, Color.ACCENT_BLUE.value, mark_rect, border_radius=5)

        pygame.draw.polygon(screen, (255, 255, 255), [
            (mark_rect.centerx - 4, mark_rect.centery - 5),
            (mark_rect.centerx + 4, mark_rect.centery),
            (mark_rect.centerx - 4, mark_rect.centery + 5)
        ])

        title_surf = self.title_font.render("Emergency Crowd Evacuator", True, Color.TEXT_PRIMARY.value)
        screen.blit(title_surf, (48, 15))

        workflow_surf = self.tiny_font.render("WORKFLOW", True, Color.TEXT_MUTED.value)
        screen.blit(workflow_surf, (screen_w - 530 - 75, 18))

        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        pygame.draw.rect(screen, (21, 21, 21), (switcher_x, 7, switcher_w, 34), border_radius=6)

        pygame.draw.rect(screen, Color.FLOORTABS_BG.value, (0, 48, sidebar_x, 40))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 87), (sidebar_x, 87), 1)

        floor_plan_surf = self.tiny_font.render("FLOOR PLAN", True, Color.TEXT_MUTED.value)
        screen.blit(floor_plan_surf, (sidebar_x - 90, 62))

        pygame.draw.rect(screen, Color.SIDEBAR_BG.value, (sidebar_x, 48, 280, screen_h - 48))
        pygame.draw.line(screen, Color.DIVIDER.value, (sidebar_x, 48), (sidebar_x, screen_h), 1)

        stats_header = self.tiny_font.render("SIMULATION STATS", True, Color.TEXT_MUTED.value)
        screen.blit(stats_header, (sidebar_x + 16, 62))

        card_rect = pygame.Rect(sidebar_x + 16, 84, 248, 118)
        pygame.draw.rect(screen, Color.CARD_BG.value, card_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)

        evacuated = context.simulation.evacuated_count if context.simulation else 0
        total = evacuated + len(context.simulation.evacuees) if context.simulation else 0
        step = context.simulation.step_count if context.simulation else 0

        r1_y = card_rect.y + 11
        lbl_step = self.font.render("Step:", True, (170, 170, 170))
        val_step = self.font.render(str(step), True, (240, 240, 240))
        screen.blit(lbl_step, (card_rect.x + 14, r1_y))
        screen.blit(val_step, (card_rect.right - val_step.get_width() - 14, r1_y))

        pygame.draw.line(screen, (52, 52, 52), (card_rect.x + 14, r1_y + 26), (card_rect.right - 14, r1_y + 26), 1)

        r2_y = r1_y + 36
        lbl_evac = self.font.render("Evacuated:", True, (170, 170, 170))
        val_evac = self.font.render(f"{evacuated} / {total}", True, (240, 240, 240))
        screen.blit(lbl_evac, (card_rect.x + 14, r2_y))
        screen.blit(val_evac, (card_rect.right - val_evac.get_width() - 14, r2_y))

        pygame.draw.line(screen, (52, 52, 52), (card_rect.x + 14, r2_y + 26), (card_rect.right - 14, r2_y + 26), 1)

        r3_y = r2_y + 36
        lbl_scale = self.font.render("Scale:", True, (170, 170, 170))
        val_scale = self.font.render("1 px = 60 cm", True, (240, 240, 240))
        screen.blit(lbl_scale, (card_rect.x + 14, r3_y))
        screen.blit(val_scale, (card_rect.right - val_scale.get_width() - 14, r3_y))

        pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, 216), (screen_w - 16, 216), 1)

        tools_header = self.tiny_font.render("TOOLS", True, (181, 181, 181))
        screen.blit(tools_header, (sidebar_x + 16, 226))

        mode_name = "Edit Mode" if context.state.name == "editor" else context.state.name.replace("_", " ").title()
        mode_label = self.tiny_font.render(mode_name, True, (141, 141, 141))
        screen.blit(mode_label, (screen_w - 16 - mode_label.get_width(), 226))

        if context.state.name == "overview":
            self._draw_sidebar_stairs_info(context, sidebar_x, screen_w)

        playback_div_y = 538
        pygame.draw.line(screen, (56, 56, 56), (sidebar_x + 16, playback_div_y), (screen_w - 16, playback_div_y), 1)

        playback_header = self.tiny_font.render("SIMULATION PLAYBACK", True, (170, 170, 170))
        screen.blit(playback_header, (sidebar_x + 16, playback_div_y + 12))

    def _draw_sidebar_stairs_info(self, context, sidebar_x, screen_w):
        if not context.simulation or not context.simulation.grid_world:
            return
        world = context.simulation.grid_world
        if not world.floors:
            return
        floor = world.active_floor
        floor_id = world.active_floor_id

        section_y = 392
        pygame.draw.line(self.screen, (52, 52, 52), (sidebar_x + 16, section_y), (screen_w - 16, section_y), 1)
        header = self.tiny_font.render(f"STAIRS (FLOOR {floor_id + 1})", True, (181, 181, 181))
        self.screen.blit(header, (sidebar_x + 16, section_y + 10))

        card_rect = pygame.Rect(sidebar_x + 16, section_y + 28, 248, 102)
        pygame.draw.rect(self.screen, Color.CARD_BG.value, card_rect, border_radius=7)
        pygame.draw.rect(self.screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)

        stair_clusters = [
            (cid, c) for cid, c in enumerate(floor.cluster_grid.clusters)
            if c.type is ClusterType.STAIRS and c.cells
        ]

        if not stair_clusters:
            empty_txt = self.small_font.render("No stairs on this floor", True, (130, 130, 130))
            self.screen.blit(empty_txt, (card_rect.x + 12, card_rect.y + 12))
            return

        row_y = card_rect.y + 10
        for cid, cluster in stair_clusters[:4]:
            cx, cy = cluster.get_centroid()
            is_pending = (world.pending_stair_endpoint == (floor_id, cid))
            if is_pending:
                status_text = "Selected..."
                status_color = (255, 230, 50)
            elif cluster.linked_to is not None:
                t_floor, t_cid = cluster.linked_to
                status_text = f"-> Floor {t_floor + 1} (C{t_cid})"
                status_color = (16, 185, 129)
            else:
                status_text = "Unlinked"
                status_color = (239, 68, 68)

            left_lbl = self.small_font.render(f"C{cid} ({int(cx)},{int(cy)})", True, (220, 220, 220))
            right_lbl = self.small_font.render(status_text, True, status_color)
            self.screen.blit(left_lbl, (card_rect.x + 12, row_y))
            self.screen.blit(right_lbl, (card_rect.right - right_lbl.get_width() - 12, row_y))
            row_y += 22

    def close(self):
        self.is_open = False
        pygame.quit()
