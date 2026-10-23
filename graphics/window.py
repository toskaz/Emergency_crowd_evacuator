import pygame
from utils.config import drawing_size
from graphics.renderer import draw_map, draw_ui, draw_key_bindings, draw_render_entities, draw_spawn_markers, draw_navigation, draw_clusters, draw_cluster_edges


class Window:

    def __init__(self, grid):
        pygame.init()

        screen_width = grid.width * drawing_size
        screen_height = grid.height * drawing_size

        self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.is_open = True

    def draw(self, grid_world, evacuees, drones, selected_tool, show_grid,
            status_lines=(), show_drone_routes=False, drone_routes=None,
            show_cluster_overlay=True, show_cluster_connections=True,
            simulation_view=False, key_bindings=(), show_tool=False):
        grid = grid_world.simulation_grid if simulation_view else grid_world.map_grid
        draw_map(
            self.screen,
            grid,
            show_grid
        )
        if show_cluster_connections:
            draw_cluster_edges(self.screen, grid_world.cluster_grid)
        if show_cluster_overlay:
            draw_clusters(self.screen, grid_world.cluster_grid, font=self.small_font)

        if show_drone_routes:
            draw_navigation(self.screen, grid_world.navigation_graph, drone_routes or {})

        draw_ui(
            self.screen,
            self.font,
            selected_tool,
            status_lines,
            key_bindings,
            show_tool,
            self.small_font,
        )

        if not simulation_view:
            draw_spawn_markers(self.screen, grid_world.map_grid)

        if simulation_view:
            draw_render_entities(self.screen, grid_world.simulation_grid)

        pygame.display.flip()

    def close(self):
        self.is_open = False
        pygame.quit()
