import pygame

from renderer import draw_map, draw_ui


class Window:

    def __init__(self, map_matrix, drawing_size):
        pygame.init()

        screen_width = len(map_matrix[0]) * drawing_size
        screen_height = len(map_matrix) * drawing_size

        self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.font = pygame.font.Font(None, 24)

    def draw(self, map_matrix, drawing_size, selected_cell_type, show_grid):
        draw_map(
            self.screen,
            map_matrix,
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
        pygame.quit()