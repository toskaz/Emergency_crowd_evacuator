class Drone:

    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y

    def move(self, dx, dy):
        self.grid_x += dx
        self.grid_y += dy
