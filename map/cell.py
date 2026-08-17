from enum import IntEnum

class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    EVACUEE = 2
    EXIT = 3