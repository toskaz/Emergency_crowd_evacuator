class Grid:

    def __init__(self, width, height, default=None):
        self.width = width
        self.height = height

        self.cells = [
            [default for _ in range(width)]
            for _ in range(height)
        ]

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        return self.cells[y][x]

    def set(self, x, y, value):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        self.cells[y][x] = value

    def copy(self):
        copied_grid = type(self)(self.width, self.height)
        copied_grid.cells = [row.copy() for row in self.cells]
        return copied_grid
