from colorsys import hsv_to_rgb
from enum import Enum

class Color(Enum):
    EMPTY = (17, 24, 39)
    EXTERIOR = (9, 13, 22)
    WALL = (203, 213, 225)
    EVACUEE = (245, 158, 11)
    DRONE = (56, 189, 248)
    EXIT = (16, 185, 129)
    STAIRS = (129, 140, 248)
    DOOR = (167, 139, 250)
    TEMPORARY_WALL = (239, 68, 68)
    GRID = (30, 41, 59)
    GRID_BRIGHT = (51, 65, 85)
    NAVIGATION = (56, 189, 248)
    BACKGROUND = (9, 13, 22)
    TOPNAV_BG = (9, 13, 22)
    SIDEBAR_BG = (17, 24, 39)
    FLOORTABS_BG = (17, 24, 39)
    CARD_BG = (30, 41, 59)
    CARD_BORDER = (51, 65, 85)
    DIVIDER = (51, 65, 85)
    TEXT_MUTED = (148, 163, 184)
    TEXT_PRIMARY = (226, 232, 240)
    ACCENT_BLUE = (56, 189, 248)

def assignment_color(drone_id):
    if drone_id is None:
        return (135, 135, 135)
    hue = (0.58 + drone_id * 0.61803398875) % 1.0
    return tuple(round(channel * 255) for channel in hsv_to_rgb(hue, 0.75, 0.85))
