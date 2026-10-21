from entities.state import EvacueeState


class Evacuee:

    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.waiting_steps = 0
        self.state = EvacueeState.MOVING
