import pygame
from utils.config import drawing_size
from graphics.renderer import draw_map, draw_ui, draw_key_bindings, draw_navigation, draw_clusters, draw_cluster_edges


class Window:

    def __init__(self, grid):
        pygame.init()

        screen_width = grid.width * drawing_size
        screen_height = grid.height * drawing_size

        self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
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
        self._draw_grid(context, grid)
        if context.state.render_options_enabled:
            self._draw_overlays(context)
        self._draw_ui(context, key_bindings, selected_tool)
        pygame.display.flip()

    def _draw_grid(self, context, grid):
        draw_map(self.screen, grid, context.view_options.show_grid)

    def _draw_overlays(self, context):
        world = context.simulation.grid_world
        options = context.view_options
        if options.show_cluster_connections:
            draw_cluster_edges(self.screen, world.cluster_grid)
        if options.show_clusters:
            draw_clusters(self.screen, world.cluster_grid, font=self.small_font)
        if options.show_drone_routes:
            draw_navigation(self.screen, world.navigation_graph, context.drone_routes)

    def _draw_ui(self, context, key_bindings, selected_tool=None):
        draw_ui(
            self.screen,
            self.font,
            selected_tool,
            context.state.status_lines(context),
            key_bindings,
            self.small_font,
        )

    def close(self):
        self.is_open = False
        pygame.quit()
