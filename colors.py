from enum import Enum

class Color(Enum):
    EMPTY = (255, 255, 255)
    WALL = (40, 40, 40)
    EVACUEE = (0, 100, 255)
    EXIT = (0, 255, 0)
    GRID = (200, 200, 200)
    BACKGROUND = (0, 0, 0)