import pygame

from renderer import draw_map, draw_ui


class Window:

    def __init__(self, grid, drawing_size):
        pygame.init()

        screen_width = grid.width * drawing_size
        screen_height = grid.height * drawing_size

        self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.font = pygame.font.Font(None, 24)
        self.is_open = True

    def draw(self, grid, drawing_size, selected_cell_type, show_grid):
        draw_map(
            self.screen,
            grid,
            drawing_size,
            show_grid
        )

        draw_ui(
            self.screen,
            self.font,
            selected_cell_type
        )

        pygame.display.flip()

    def close(self):
        self.is_open = False
        pygame.quit()