import pygame
from graphics.colors import Color
from grids.cluster import ClusterType
from graphics.renderer import (
    draw_map,
    draw_ui,
    draw_navigation,
    draw_clusters,
    draw_cluster_edges,
    draw_temporary_walls,
    draw_stair_highlights,
    draw_assignment_groups,
    draw_map_scale,
    draw_legend,
    draw_heatmap,
    draw_heatmap_legend,
    draw_building_elevation,
)


class Window:

    def __init__(self, grid):
        pygame.init()

        self.screen = pygame.display.get_surface()
        if self.screen is None:
            self.screen = pygame.display.set_mode((1280, 800))

        self.title_font = pygame.font.Font(None, 24)
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 15)
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
        if context.simulation and (context.simulation.started or getattr(context.state, "name", "") in ("simulation_paused", "simulation_editor")):
            draw_temporary_walls(
                self.screen, context.simulation.grid_world.temporary_wall_grid
            )
        if context.state.render_options_enabled:
            self._draw_overlays(context)
        self._draw_ui(context, key_bindings, selected_tool)
        legend_top_y = 96
        legend_h = draw_legend(self.screen, self.tiny_font, top_left=(14, legend_top_y))
        active_floor = context.simulation.grid_world.active_floor if context.simulation and context.simulation.grid_world else None
        cell_m = getattr(active_floor, 'cell_size_m', 0.6) if active_floor else 0.6
        scale_top_y = legend_top_y + legend_h + 6
        scale_h = draw_map_scale(self.screen, self.tiny_font, top_left=(14, scale_top_y), cell_size_m=cell_m)

        next_stack_y = scale_top_y + scale_h + 6
        if getattr(context.view_options, "show_heatmap", False):
            heat_h = draw_heatmap_legend(self.screen, self.tiny_font, top_left=(14, next_stack_y))
            next_stack_y += heat_h + 6

        if context.simulation and context.simulation.grid_world:
            draw_building_elevation(
                self.screen,
                self.font,
                self.small_font,
                context.simulation.grid_world,
                min_top_y=next_stack_y + 8,
            )
        self.screen.set_clip(None)

        self._draw_figma_chrome(context, selected_tool)

    def _draw_grid(self, context, grid):
        draw_map(self.screen, grid, context.view_options.show_grid)

    def _draw_overlays(self, context):
        world = context.simulation.grid_world
        options = context.view_options
        if getattr(options, "show_heatmap", False):
            draw_heatmap(self.screen, world.active_floor)
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

        title_text = "Emergency Crowd Evacuator" if screen_w >= 1050 else "ARIADNE"
        title_surf = self.title_font.render(title_text, True, Color.TEXT_PRIMARY.value)
        screen.blit(title_surf, (48, 15))

        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        pygame.draw.rect(screen, (21, 21, 21), (switcher_x, 7, switcher_w, 34), border_radius=6)

        if screen_w >= 1150:
            workflow_surf = self.tiny_font.render("WORKFLOW", True, Color.TEXT_MUTED.value)
            screen.blit(workflow_surf, (switcher_x - workflow_surf.get_width() - 12, 18))

        pygame.draw.rect(screen, Color.FLOORTABS_BG.value, (0, 48, sidebar_x, 40))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 87), (sidebar_x, 87), 1)

        floor_plan_surf = self.tiny_font.render("FLOOR PLAN", True, Color.TEXT_MUTED.value)
        screen.blit(floor_plan_surf, (sidebar_x - 90, 62))

        pygame.draw.rect(screen, Color.SIDEBAR_BG.value, (sidebar_x, 48, 280, screen_h - 48))
        pygame.draw.line(screen, Color.DIVIDER.value, (sidebar_x, 48), (sidebar_x, screen_h), 1)

        active_floor = context.simulation.grid_world.active_floor if context.simulation and context.simulation.grid_world else None
        cell_m = getattr(active_floor, 'cell_size_m', 0.6) if active_floor else 0.6
        if cell_m >= 1.0:
            val_scale = self.font.render(f"1 cell = {cell_m:.2f} m", True, (240, 240, 240))
        else:
            val_scale = self.font.render(f"1 cell = {round(cell_m * 100)} cm", True, (240, 240, 240))

        if context.state.name in ("editor", "overview"):
            scale_header = self.tiny_font.render("SCALE", True, Color.TEXT_MUTED.value)
            screen.blit(scale_header, (sidebar_x + 16, 60))
            card_rect = pygame.Rect(sidebar_x + 16, 80, 248, 42)
            pygame.draw.rect(screen, Color.CARD_BG.value, card_rect, border_radius=7)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)
            lbl_scale = self.font.render("Scale:", True, (170, 170, 170))
            screen.blit(lbl_scale, (card_rect.x + 14, card_rect.y + 13))
            screen.blit(val_scale, (card_rect.right - val_scale.get_width() - 14, card_rect.y + 13))

            divider_y = 134
            pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, divider_y), (screen_w - 16, divider_y), 1)

            if context.state.name == "editor":
                tools_header = self.tiny_font.render("TOOLS", True, (181, 181, 181))
                screen.blit(tools_header, (sidebar_x + 16, divider_y + 8))
                mode_label = self.tiny_font.render("Edit Mode", True, (141, 141, 141))
                screen.blit(mode_label, (screen_w - 16 - mode_label.get_width(), divider_y + 8))

                act_div_y = 372
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, act_div_y), (screen_w - 16, act_div_y), 1)
                actions_header = self.tiny_font.render("ACTIONS", True, (181, 181, 181))
                screen.blit(actions_header, (sidebar_x + 16, act_div_y + 8))
            elif context.state.name == "overview":
                self._draw_sidebar_stairs_info(context, sidebar_x, screen_w)
                opt_div_y = 320
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, opt_div_y), (screen_w - 16, opt_div_y), 1)
                view_header = self.tiny_font.render("VIEW OPTIONS", True, (181, 181, 181))
                screen.blit(view_header, (sidebar_x + 16, opt_div_y + 8))

        else:
            stats_header = self.tiny_font.render("SIMULATION STATS", True, Color.TEXT_MUTED.value)
            screen.blit(stats_header, (sidebar_x + 16, 60))
            evacuated = context.simulation.evacuated_count if context.simulation else 0
            total = evacuated + len(context.simulation.evacuees) if context.simulation else 0
            step = context.simulation.step_count if context.simulation else 0
            pct = (evacuated / total) if total > 0 else 0.0

            top_pad = 11
            r1_y = 80 + top_pad
            lbl_step = self.font.render("Step:", True, (170, 170, 170))
            val_step = self.font.render(str(step), True, (240, 240, 240))
            r2_y = r1_y + 32
            lbl_evac = self.font.render("Evacuated:", True, (170, 170, 170))
            val_evac = self.font.render(f"{evacuated} / {total} ({pct * 100:.0f}%)", True, (240, 240, 240))
            r3_y = r2_y + 39
            lbl_scale = self.font.render("Scale:", True, (170, 170, 170))
            card_h = (r3_y + lbl_scale.get_height() + top_pad) - 80
            card_rect = pygame.Rect(sidebar_x + 16, 80, 248, card_h)
            pygame.draw.rect(screen, Color.CARD_BG.value, card_rect, border_radius=7)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)

            screen.blit(lbl_step, (card_rect.x + 14, r1_y))
            screen.blit(val_step, (card_rect.right - val_step.get_width() - 14, r1_y))

            pygame.draw.line(screen, (52, 52, 52), (card_rect.x + 14, r1_y + 24), (card_rect.right - 14, r1_y + 24), 1)

            screen.blit(lbl_evac, (card_rect.x + 14, r2_y))
            screen.blit(val_evac, (card_rect.right - val_evac.get_width() - 14, r2_y))

            bar_rect = pygame.Rect(card_rect.x + 14, r2_y + 17, card_rect.width - 28, 6)
            pygame.draw.rect(screen, (38, 38, 38), bar_rect, border_radius=3)
            fill_w = int(bar_rect.width * pct)
            if fill_w > 0:
                pygame.draw.rect(screen, (16, 185, 129), (bar_rect.x, bar_rect.y, fill_w, bar_rect.height), border_radius=3)

            pygame.draw.line(screen, (52, 52, 52), (card_rect.x + 14, r2_y + 31), (card_rect.right - 14, r2_y + 31), 1)

            screen.blit(lbl_scale, (card_rect.x + 14, r3_y))
            screen.blit(val_scale, (card_rect.right - val_scale.get_width() - 14, r3_y))

            if context.state.name in ("simulation_paused", "simulation_running"):
                tools_div_y = 216
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, tools_div_y), (screen_w - 16, tools_div_y), 1)
                tools_header = self.tiny_font.render("TOOLS", True, (181, 181, 181))
                screen.blit(tools_header, (sidebar_x + 16, tools_div_y + 8))

                opt_div_y = 292
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, opt_div_y), (screen_w - 16, opt_div_y), 1)
                view_header = self.tiny_font.render("VIEW OPTIONS", True, (181, 181, 181))
                screen.blit(view_header, (sidebar_x + 16, opt_div_y + 8))

                pb_div_y = 428
                pygame.draw.line(screen, (56, 56, 56), (sidebar_x + 16, pb_div_y), (screen_w - 16, pb_div_y), 1)
                pb_header = self.tiny_font.render("SIMULATION PLAYBACK", True, (170, 170, 170))
                screen.blit(pb_header, (sidebar_x + 16, pb_div_y + 8))

    def _draw_sidebar_stairs_info(self, context, sidebar_x, screen_w):
        if not context.simulation or not context.simulation.grid_world:
            return
        world = context.simulation.grid_world
        if not world.floors:
            return
        floor = world.active_floor
        floor_id = world.active_floor_id

        header = self.tiny_font.render(f"STAIRS (FLOOR {floor_id + 1})", True, (181, 181, 181))
        self.screen.blit(header, (sidebar_x + 16, 142))

        card_rect = pygame.Rect(sidebar_x + 16, 164, 248, 102)
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
