import pygame
import sys
import cv2
from enum import IntEnum, Enum


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    EVACUEE = 2
    EXIT = 3


class Color(Enum):
    EMPTY = (255, 255, 255)
    WALL = (40, 40, 40)
    EVACUEE = (0, 100, 255)
    EXIT = (0, 255, 0)
    GRID = (200, 200, 200)
    BACKGROUND = (0, 0, 0)


IMAGE_PATH = "plan_black_white.jpg"
BLOCK_SIZE = 25
DRAWING_SIZE = 8
ACCEPTANCE_THRESHOLD = 0.40


def initialize_pygame(map_matrix, drawing_size):
    pygame.init()

    screen_width = len(map_matrix[0]) * drawing_size
    screen_height = len(map_matrix) * drawing_size

    screen = pygame.display.set_mode((screen_width, screen_height))

    return screen


def generate_map_matrix(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    height, width = thresh.shape
    map_matrix = []

    for y in range(0, height, BLOCK_SIZE):
        row = []

        for x in range(0, width, BLOCK_SIZE):
            block = thresh[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
            area = BLOCK_SIZE * BLOCK_SIZE

            number_of_white_pixels = cv2.countNonZero(block)
            number_of_black_pixels = area - number_of_white_pixels

            if number_of_black_pixels > (area * ACCEPTANCE_THRESHOLD):
                row.append(CellType.WALL)
            else:
                row.append(CellType.EMPTY)

        map_matrix.append(row)

    return map_matrix


def handle_input(map_matrix, drawing_size, mode):
    for event in pygame.event.get():

        # Close window
        if event.type == pygame.QUIT:
            return False, mode

        # Keyboard input
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_1:
                mode = CellType.WALL

            elif event.key == pygame.K_2:
                mode = CellType.EVACUEE

            elif event.key == pygame.K_3:
                mode = CellType.EXIT

        # Mouse input
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_map_editor(
                map_matrix,
                event.pos,
                event.button,
                drawing_size,
                mode
            )

    return True, mode


def handle_map_editor(map_matrix, mouse_pos, button, drawing_size, mode):
    pos_x, pos_y = mouse_pos

    grid_x = pos_x // drawing_size
    grid_y = pos_y // drawing_size

    max_y = len(map_matrix)
    max_x = len(map_matrix[0])

    if 0 <= grid_y < max_y and 0 <= grid_x < max_x:

        # LMB = place current mode
        if button == 1:
            map_matrix[grid_y][grid_x] = mode

        # RMB = erase
        elif button == 3:
            map_matrix[grid_y][grid_x] = CellType.EMPTY


def draw_map(screen, map_matrix, drawing_size):
    screen.fill(Color.BACKGROUND.value)

    for y, row in enumerate(map_matrix):
        for x, value in enumerate(row):
            pos_x = x * drawing_size
            pos_y = y * drawing_size

            if value == CellType.EMPTY:
                color = Color.EMPTY.value

            elif value == CellType.WALL:
                color = Color.WALL.value

            elif value == CellType.EVACUEE:
                color = Color.EVACUEE.value

            elif value == CellType.EXIT:
                color = Color.EXIT.value

            # Draw the cell
            pygame.draw.rect(
                screen,
                color,
                (pos_x, pos_y, drawing_size, drawing_size)
            )

            # Draw the grid line
            pygame.draw.rect(
                screen,
                Color.GRID.value,
                (pos_x, pos_y, drawing_size, drawing_size),
                1
            )


def main():
    map_matrix = generate_map_matrix(IMAGE_PATH)

    screen = initialize_pygame(map_matrix, DRAWING_SIZE)

    mode = CellType.WALL

    working = True

    while working:
        working, mode = handle_input(
            map_matrix,
            DRAWING_SIZE,
            mode
        )

        draw_map(
            screen,
            map_matrix,
            DRAWING_SIZE
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
