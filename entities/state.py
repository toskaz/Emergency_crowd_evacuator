from enum import Enum

class EvacueeState(Enum):
    MOVING = "moving"
    WAITING_AT_EXIT = "waiting at exit"
    EVACUATED = "evacuated"

class DroneState(Enum):
    FLYING = "flying"
    AT_EXIT = "at exit"
