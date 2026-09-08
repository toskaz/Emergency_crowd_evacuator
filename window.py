import pygame
from config import drawing_size
from renderer import draw_map, draw_ui, draw_evacuees, draw_drones


class Window:

    def __init__(self, grid):
        pygame.init()

        screen_width = grid.width * drawing_size
        screen_height = grid.height * drawing_size

        self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.font = pygame.font.Font(None, 24)
        self.is_open = True

    def draw(self, grid, evacuees, drones, selected_cell_type, show_grid):
        draw_map(
            self.screen,
            grid,
            show_grid
        )

        draw_ui(
            self.screen,
            self.font,
            selected_cell_type
        )

        draw_evacuees(self.screen, evacuees)
        draw_drones(self.screen, drones)

        pygame.display.flip()

    def close(self):
        self.is_open = False
        pygame.quit()