class Evacuee:

    def __init__(self, grid_x, grid_y, floor_id=0, assigned_drone_id=None):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.floor_id = floor_id
        self.assigned_drone_id = assigned_drone_id