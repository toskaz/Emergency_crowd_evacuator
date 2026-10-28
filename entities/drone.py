class Drone:

    def __init__(self, grid_x, grid_y, floor_id=0):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.floor_id = floor_id

    def move(self, dx, dy):
        self.grid_x += dx
        self.grid_y += dy
