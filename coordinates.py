from config import drawing_size

def grid_to_pixel(grid_x, grid_y):
    pixel_x = grid_x * drawing_size
    pixel_y = grid_y * drawing_size

    return pixel_x, pixel_y


def grid_to_pixel_center(grid_x, grid_y):
    pixel_x = grid_x * drawing_size + drawing_size // 2
    pixel_y = grid_y * drawing_size + drawing_size // 2

    return pixel_x, pixel_y


def pixel_to_grid(pixel_x, pixel_y):
    grid_x = pixel_x // drawing_size
    grid_y = pixel_y // drawing_size

    return grid_x, grid_y