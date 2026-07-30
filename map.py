import pygame
import sys
import cv2

image_path = "regular-office-floor-plan.png"
block_size = 25
drawing_size = 15
acceptance_threshold = 0.35

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

def main():
    map_matrix = generate_map_matrix(image_path)

    pygame.init()

    screen = pygame.display.set_mode()

    working = True
    while working:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                working = False

        screen.fill((0, 0, 0)) 

        for y, row in enumerate(map_matrix):
            for x, value in enumerate(row):
                pos_x = x * drawing_size
                pos_y = y * drawing_size

                if value == 1:
                    color = (40, 40, 40) 
                else:
                    color = (255, 255, 255) 

                pygame.draw.rect(screen, color, (pos_x, pos_y, drawing_size, drawing_size))
                pygame.draw.rect(screen, (200, 200, 200), (pos_x, pos_y, drawing_size, drawing_size), 1)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()