from utils.config import cluster_size
import utils.config as config

def grid_to_pixel(grid_x, grid_y):
    pixel_x = grid_x * config.drawing_size + getattr(config, 'camera_offset_x', 0)
    pixel_y = grid_y * config.drawing_size + getattr(config, 'camera_offset_y', 0)

    return pixel_x, pixel_y


def grid_to_pixel_center(grid_x, grid_y):
    pixel_x = grid_x * config.drawing_size + config.drawing_size // 2 + getattr(config, 'camera_offset_x', 0)
    pixel_y = grid_y * config.drawing_size + config.drawing_size // 2 + getattr(config, 'camera_offset_y', 0)

    return pixel_x, pixel_y


def pixel_to_grid(pixel_x, pixel_y):
    grid_x = (pixel_x - getattr(config, 'camera_offset_x', 0)) // config.drawing_size
    grid_y = (pixel_y - getattr(config, 'camera_offset_y', 0)) // config.drawing_size

    return grid_x, grid_y