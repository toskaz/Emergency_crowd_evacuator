from pathlib import Path
import pygame
from graphics.colors import Color
from graphics.fonts import get_font
from grids.cluster import ClusterType
from graphics.renderer import (
    draw_map,
    draw_ui,
    draw_navigation,
    draw_clusters,
    draw_cluster_edges,
    draw_temporary_walls,
    draw_stair_highlights,
    draw_exit_badges,
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

        self.title_font = get_font(22, bold=True)
        self.font = get_font(15)
        self.small_font = get_font(13)
        self.tiny_font = get_font(11, bold=True)
        self.is_open = True

        logo_path = Path(__file__).resolve().parent.parent / "logo" / "logoSquare.png"
        self.logo_surf = None
        if logo_path.exists():
            try:
                raw_logo = pygame.image.load(str(logo_path)).convert_alpha()
                self.logo_surf = pygame.transform.smoothscale(raw_logo, (20, 20))
            except Exception:
                self.logo_surf = None

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
        else:
            if context.simulation and context.simulation.grid_world:
                draw_stair_highlights(
                    self.screen,
                    context.simulation.grid_world,
                    font=self.small_font,
                    link_mode=context.link_stairs_mode,
                )
                draw_exit_badges(
                    self.screen,
                    context.simulation.grid_world,
                    font=self.tiny_font,
                )
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
        pygame.draw.rect(self.screen, (45, 48, 54), canvas_rect, 1)

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
            is_sim=getattr(context.state, "name", "").startswith("simulation"),
            simulation=context.simulation,
        )
        draw_exit_badges(
            self.screen,
            world,
            font=self.tiny_font,
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

        mark_rect = pygame.Rect(16, (48 - 28) // 2, 28, 28)
        pygame.draw.rect(screen, (255, 255, 255), mark_rect, border_radius=6)
        if getattr(self, "logo_surf", None):
            lx = mark_rect.centerx - self.logo_surf.get_width() // 2
            ly = mark_rect.centery - self.logo_surf.get_height() // 2
            screen.blit(self.logo_surf, (lx, ly))

        title_text = "ARIADNE"
        title_surf = self.title_font.render(title_text, True, Color.TEXT_PRIMARY.value)
        title_rect = title_surf.get_rect(left=mark_rect.right + 12, centery=24)
        screen.blit(title_surf, title_rect)

        switcher_w = 260
        switcher_x = screen_w - switcher_w - 16
        pygame.draw.rect(screen, (21, 21, 21), (switcher_x, 7, switcher_w, 34), border_radius=6)

        if screen_w >= 1050:
            workflow_surf = self.tiny_font.render("WORKFLOW", True, Color.TEXT_MUTED.value)
            workflow_rect = workflow_surf.get_rect(right=switcher_x - 12, centery=24)
            screen.blit(workflow_surf, workflow_rect)

        pygame.draw.rect(screen, Color.FLOORTABS_BG.value, (0, 48, sidebar_x, 40))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 87), (sidebar_x, 87), 1)

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
            screen.blit(scale_header, (sidebar_x + 16, 56))
            card_rect = pygame.Rect(sidebar_x + 16, 76, 248, 38)
            pygame.draw.rect(screen, Color.CARD_BG.value, card_rect, border_radius=7)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)
            lbl_scale = self.font.render("Scale:", True, (170, 170, 170))
            lbl_scale_rect = lbl_scale.get_rect(left=card_rect.x + 14, centery=card_rect.centery)
            screen.blit(lbl_scale, lbl_scale_rect)
            val_scale_rect = val_scale.get_rect(right=card_rect.right - 14, centery=card_rect.centery)
            screen.blit(val_scale, val_scale_rect)

            divider_y = 124
            pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, divider_y), (screen_w - 16, divider_y), 1)

            if context.state.name == "editor":
                tools_header = self.tiny_font.render("TOOLS", True, (181, 181, 181))
                screen.blit(tools_header, (sidebar_x + 16, divider_y + 8))
                mode_label = self.tiny_font.render("Edit Mode", True, (141, 141, 141))
                screen.blit(mode_label, (screen_w - 16 - mode_label.get_width(), divider_y + 8))

                act_div_y = 266
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, act_div_y), (screen_w - 16, act_div_y), 1)
                actions_header = self.tiny_font.render("ACTIONS", True, (181, 181, 181))
                screen.blit(actions_header, (sidebar_x + 16, act_div_y + 8))

                world = context.simulation.grid_world if context.simulation else None
                if world and len(world.floors) > 1:
                    act_floor_id = world.active_floor_id
                    stair_div_y = 438 if act_floor_id > 0 else 406
                    pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, stair_div_y), (screen_w - 16, stair_div_y), 1)
                    stair_hdr = self.tiny_font.render("STAIR LINKS", True, (181, 181, 181))
                    screen.blit(stair_hdr, (sidebar_x + 16, stair_div_y + 8))
            elif context.state.name == "overview":
                self._draw_sidebar_stairs_info(context, sidebar_x, screen_w)
                opt_div_y = 320
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, opt_div_y), (screen_w - 16, opt_div_y), 1)
                view_header = self.tiny_font.render("VIEW OPTIONS", True, (181, 181, 181))
                screen.blit(view_header, (sidebar_x + 16, opt_div_y + 8))

        else:
            stats_header = self.tiny_font.render("SIMULATION STATS", True, Color.TEXT_MUTED.value)
            screen.blit(stats_header, (sidebar_x + 16, 56))

            if hasattr(context.simulation, "metrics"):
                hud = context.simulation.metrics.get_live_hud_data(context.simulation)
            else:
                evac = context.simulation.evacuated_count if context.simulation else 0
                tot = evac + len(context.simulation.evacuees) if context.simulation else 0
                hud = {
                    "progress_pct": (evac / tot) if tot > 0 else 0.0,
                    "progress_str": f"{int((evac / tot) * 100) if tot > 0 else 0}%",
                    "remaining_str": f"{len(context.simulation.evacuees) if context.simulation else 0} / {tot}",
                    "sim_time_str": "00:00",
                    "exit_flow_str": "0.0 people/s",
                    "on_stairs_str": "0",
                    "bottlenecks_str": "None",
                }

            tbl_x = sidebar_x + 16
            tbl_y = 78
            tbl_w = 248
            hdr_h = 24
            row_h = 26
            rows_data = [
                ("Progress", hud["progress_str"], (240, 240, 240), hud["progress_pct"]),
                ("Remaining", hud["remaining_str"], (240, 240, 240), None),
                ("Sim Time", hud["sim_time_str"], (240, 240, 240), None),
                ("Exit Flow", hud["exit_flow_str"], (240, 240, 240), None),
                ("On Stairs", hud["on_stairs_str"], (240, 240, 240), None),
            ]
            tbl_h = hdr_h + len(rows_data) * row_h
            table_rect = pygame.Rect(tbl_x, tbl_y, tbl_w, tbl_h)

            pygame.draw.rect(screen, (24, 25, 28), table_rect, border_radius=6)
            pygame.draw.rect(screen, (60, 64, 72), table_rect, 1, border_radius=6)

            hdr_rect = pygame.Rect(tbl_x, tbl_y, tbl_w, hdr_h)
            pygame.draw.rect(screen, (34, 38, 46), hdr_rect, border_top_left_radius=6, border_top_right_radius=6)
            pygame.draw.line(screen, (60, 64, 72), (tbl_x, tbl_y + hdr_h), (tbl_x + tbl_w, tbl_y + hdr_h), 1)

            col_split_x = tbl_x + 96
            pygame.draw.line(screen, (52, 56, 64), (col_split_x, tbl_y + hdr_h), (col_split_x, tbl_y + tbl_h), 1)

            hdr_metric = self.tiny_font.render("METRIC", True, (140, 155, 180))
            hdr_value = self.tiny_font.render("VALUE", True, (140, 155, 180))
            screen.blit(hdr_metric, (tbl_x + 8, tbl_y + 4))
            screen.blit(hdr_value, (tbl_x + tbl_w - 8 - hdr_value.get_width(), tbl_y + 4))

            for idx, (label, val_str, val_col, prg_pct) in enumerate(rows_data):
                ry = tbl_y + hdr_h + idx * row_h
                row_bg = (30, 31, 35) if idx % 2 == 0 else (24, 25, 28)
                if idx == len(rows_data) - 1:
                    row_rect = pygame.Rect(tbl_x + 1, ry, tbl_w - 2, row_h - 1)
                    pygame.draw.rect(screen, row_bg, row_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
                else:
                    row_rect = pygame.Rect(tbl_x + 1, ry, tbl_w - 2, row_h)
                    pygame.draw.rect(screen, row_bg, row_rect)
                    pygame.draw.line(screen, (44, 46, 52), (tbl_x, ry + row_h), (tbl_x + tbl_w, ry + row_h), 1)

                l_surf = self.small_font.render(label, True, (170, 175, 185))
                l_rect = l_surf.get_rect(left=tbl_x + 8, centery=ry + row_h // 2)
                screen.blit(l_surf, l_rect)

                v_surf = self.small_font.render(val_str, True, val_col)
                if prg_pct is not None:
                    bar_w = 64
                    bar_h = 4
                    bar_x = tbl_x + tbl_w - 8 - bar_w
                    bar_y = ry + (row_h - bar_h) // 2
                    pygame.draw.rect(screen, (45, 45, 48), (bar_x, bar_y, bar_w, bar_h), border_radius=2)
                    fill_w = int(bar_w * prg_pct)
                    if fill_w > 0:
                        pygame.draw.rect(screen, (16, 185, 129), (bar_x, bar_y, fill_w, bar_h), border_radius=2)
                    v_rect = v_surf.get_rect(right=bar_x - 6, centery=ry + row_h // 2)
                    screen.blit(v_surf, v_rect)
                else:
                    v_rect = v_surf.get_rect(right=tbl_x + tbl_w - 8, centery=ry + row_h // 2)
                    screen.blit(v_surf, v_rect)

            if context.state.name in ("simulation_paused", "simulation_running"):
                tools_div_y = 242
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, tools_div_y), (screen_w - 16, tools_div_y), 1)
                tools_header = self.tiny_font.render("TOOLS", True, (181, 181, 181))
                screen.blit(tools_header, (sidebar_x + 16, tools_div_y + 8))

                opt_div_y = 310
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, opt_div_y), (screen_w - 16, opt_div_y), 1)
                view_header = self.tiny_font.render("VIEW OPTIONS", True, (181, 181, 181))
                screen.blit(view_header, (sidebar_x + 16, opt_div_y + 8))

                algo_div_y = 434
                pygame.draw.line(screen, (52, 52, 52), (sidebar_x + 16, algo_div_y), (screen_w - 16, algo_div_y), 1)
                algo_header = self.tiny_font.render("ROUTING ALGORITHM", True, (181, 181, 181))
                screen.blit(algo_header, (sidebar_x + 16, algo_div_y + 8))

                pb_div_y = 504
                pygame.draw.line(screen, (56, 56, 56), (sidebar_x + 16, pb_div_y), (screen_w - 16, pb_div_y), 1)
                pb_header = self.tiny_font.render("SIMULATION PLAYBACK", True, (170, 170, 170))
                screen.blit(pb_header, (sidebar_x + 16, pb_div_y + 8))

                speed_div_y = 608
                pygame.draw.line(screen, (56, 56, 56), (sidebar_x + 16, speed_div_y), (screen_w - 16, speed_div_y), 1)
                speed_header = self.tiny_font.render("SPEED", True, (170, 170, 170))
                screen.blit(speed_header, (sidebar_x + 16, speed_div_y + 8))

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
            s_name = world.get_stair_name(floor_id, cid)
            is_pending = (world.pending_stair_endpoint == (floor_id, cid))
            if is_pending:
                status_text = "Selected..."
                status_color = (255, 230, 50)
            elif cluster.linked_to is not None:
                t_floor, t_cid = cluster.linked_to
                t_name = world.get_stair_name(t_floor, t_cid)
                status_text = f"-> Floor {t_floor + 1} ({t_name})"
                status_color = (16, 185, 129)
            else:
                status_text = "Unlinked"
                status_color = (239, 68, 68)

            left_lbl = self.small_font.render(s_name, True, (220, 220, 220))
            right_lbl = self.small_font.render(status_text, True, status_color)
            self.screen.blit(left_lbl, (card_rect.x + 12, row_y))
            self.screen.blit(right_lbl, (card_rect.right - right_lbl.get_width() - 12, row_y))
            row_y += 22

    def close(self):
        self.is_open = False
        pygame.quit()
