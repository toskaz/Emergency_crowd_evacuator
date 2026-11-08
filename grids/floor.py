import logging

from grids.cluster_grid import ClusterGrid
from grids.grid import Grid
from map.cell import CellType


logger = logging.getLogger(__name__)


class Floor:
    def __init__(self, floor_id, map_grid):
        self.id = floor_id
        self.map_grid = map_grid
        self.evacuee_occupancy_grid = Grid(map_grid.width, map_grid.height, None)
        self.drone_occupancy_grid = Grid(map_grid.width, map_grid.height, None)
        self.rebuild_map_data()

    def rebuild_map_data(self):
        self.wall_and_exit_grid = Grid(
            self.map_grid.width, self.map_grid.height, CellType.EMPTY
        )
        for y in range(self.map_grid.height):
            for x in range(self.map_grid.width):
                cell_type = self.map_grid.get(x, y)
                if cell_type in (CellType.WALL, CellType.EXIT, CellType.STAIRS):
                    self.wall_and_exit_grid.set(x, y, cell_type)

        self.simulation_grid = self.wall_and_exit_grid.copy()
        self.temporary_wall_grid = Grid(
            self.map_grid.width, self.map_grid.height, False
        )
        self.cluster_grid = ClusterGrid(self.map_grid)
        logger.debug("Clusters generated for floor %d", self.id + 1)
        self.distance_map = None
