from colorsys import hsv_to_rgb
from enum import Enum

class Color(Enum):
    EMPTY = (255, 255, 255)
    WALL = (40, 40, 40)
    EVACUEE = (0, 100, 255)
    DRONE = (220, 20, 60)
    EXIT = (0, 255, 0)
    STAIRS = (255, 200, 0)
    TEMPORARY_WALL = (220, 85, 55)
    GRID = (200, 200, 200)
    NAVIGATION = (240, 150, 40)
    BACKGROUND = (255,255,255)


def assignment_color(drone_id):
    if drone_id is None:
        return (135, 135, 135)
    hue = (0.58 + drone_id * 0.61803398875) % 1.0
    return tuple(round(channel * 255) for channel in hsv_to_rgb(hue, 0.75, 0.85))
