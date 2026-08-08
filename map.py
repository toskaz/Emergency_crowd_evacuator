import pygame
import sys
import cv2

image_path = "plan_black_white.jpg"
block_size = 25
drawing_size = 8
acceptance_threshold = 0.40

def generate_map_matrix(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('test_threshold.png', thresh)

    height, width = thresh.shape
    map_matrix = []
    for y in range(0, height, block_size):
        row = []
        for x in range(0, width, block_size):
            block = thresh[y : y + block_size, x : x + block_size]
            area = block_size * block_size

            number_of_white_pixels = cv2.countNonZero(block)
            number_of_black_pixels = area - number_of_white_pixels
            
            if number_of_black_pixels > (area * acceptance_threshold):
                row.append(1)
            else:
                row.append(0) 
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
                mode = 1

            elif event.key == pygame.K_2:
                mode = 2

            elif event.key == pygame.K_3:
                mode = 3

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


# if left mouse button (1) we add a wall
# if right mouse button (3) we remove the wall
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
            map_matrix[grid_y][grid_x] = 0

def draw_map(screen, map_matrix, drawing_size):
    screen.fill((0, 0, 0))
    for y, row in enumerate(map_matrix):
        for x, value in enumerate(row):
            pos_x = x * drawing_size
            pos_y = y * drawing_size

            if value == 0:
                color = (255, 255, 255)    # empty

            elif value == 1:
                color = (40, 40, 40)       # wall

            elif value == 2:
                color = (0, 100, 255)      # blue escapist

            elif value == 3:
                color = (0, 255, 0)        # green exit

            # Draw the cell
            pygame.draw.rect(screen, color, (pos_x, pos_y, drawing_size, drawing_size))

            # Draw the grid line
            pygame.draw.rect(screen, (200, 200, 200), (pos_x, pos_y, drawing_size, drawing_size), 1)

def main():
    map_matrix = generate_map_matrix(image_path)

    pygame.init()

    SCREEN_WIDTH = len(map_matrix[0]) * drawing_size
    SCREEN_HEIGHT = len(map_matrix) * drawing_size

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    mode = 1
    working = True
    while working:
        working, mode = handle_input(map_matrix, drawing_size, mode)

        draw_map(screen, map_matrix, drawing_size)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()