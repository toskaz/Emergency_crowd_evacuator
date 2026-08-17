import cv2
from config import block_size, acceptance_threshold
from map.cell import CellType

def generate_map_matrix(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    height, width = thresh.shape
    map_matrix = []

    for y in range(0, height, block_size):
        row = []

        for x in range(0, width, block_size):
            block = thresh[y : y + block_size, x : x + block_size]

            area = block.shape[0] * block.shape[1]

            number_of_white_pixels = cv2.countNonZero(block)
            number_of_black_pixels = area - number_of_white_pixels

            if number_of_black_pixels > (area * acceptance_threshold):
                row.append(CellType.WALL)
            else:
                row.append(CellType.EMPTY)

        map_matrix.append(row)

    return map_matrix