from enum import IntEnum

class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    EXIT = 2
    EVACUEE = 3
    DRONE = 4