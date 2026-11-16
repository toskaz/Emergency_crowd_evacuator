from colorsys import hsv_to_rgb
from enum import Enum

class Color(Enum):
    EMPTY = (244, 245, 245)
    WALL = (32, 34, 37)
    EVACUEE = (59, 130, 246)
    DRONE = (220, 20, 60)
    EXIT = (16, 185, 129)
    STAIRS = (245, 158, 11)
    DOOR = (160, 90, 210)
    TEMPORARY_WALL = (220, 85, 55)
    GRID = (225, 226, 229)
    NAVIGATION = (240, 150, 40)
    BACKGROUND = (22, 22, 22)
    TOPNAV_BG = (30, 30, 30)
    SIDEBAR_BG = (30, 30, 30)
    FLOORTABS_BG = (37, 37, 37)
    CARD_BG = (37, 37, 37)
    CARD_BORDER = (53, 53, 53)
    DIVIDER = (54, 54, 54)
    TEXT_MUTED = (163, 163, 163)
    TEXT_PRIMARY = (242, 242, 242)
    ACCENT_BLUE = (22, 101, 52)

def assignment_color(drone_id):
    if drone_id is None:
        return (135, 135, 135)
    hue = (0.58 + drone_id * 0.61803398875) % 1.0
    return tuple(round(channel * 255) for channel in hsv_to_rgb(hue, 0.75, 0.85))
