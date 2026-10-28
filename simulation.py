from grids.grid_world import Grid_World
from grids.grid import Grid
from entities.evacuee import Evacuee
from entities.drone import Drone
from map.cell import CellType


class Simulation:

    def __init__(self):
        self.grid_world = Grid_World()
        self.evacuees = []
        self.drones = []

        self.step_count = 0
        self.evacuated_count = 0
        self.started = False
        self.prepared = False

    def prepare(self):
        self.evacuees.clear()
        self.drones.clear()
        self.step_count = 0
        self.evacuated_count = 0
        self.started = False

        for floor in self.grid_world.floors.values():
            floor.evacuee_occupancy_grid = Grid(
                floor.map_grid.width, floor.map_grid.height, None
            )
            floor.drone_occupancy_grid = Grid(
                floor.map_grid.width, floor.map_grid.height, None
            )
        self.grid_world.rebuild_map_data()

        for floor_id, floor in self.grid_world.floors.items():
            for y in range(floor.map_grid.height):
                for x in range(floor.map_grid.width):
                    cell_type = floor.map_grid.get(x, y)
                    if cell_type == CellType.EVACUEE:
                        evacuee = Evacuee(x, y, floor_id)
                        self.evacuees.append(evacuee)
                        self.grid_world.add_evacuee(evacuee)
                    elif cell_type == CellType.DRONE:
                        drone = Drone(x, y, floor_id)
                        self.drones.append(drone)
                        self.grid_world.add_drone(drone)

        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)
        self.prepared = True

    def start(self):
        if self.started:
            return

        if not self.prepared:
            self.prepare()

        self.started = True

    def touching_exit(self, evacuee):
        return self.grid_world.is_exit(
            evacuee.grid_x, evacuee.grid_y, evacuee.floor_id
        )

    def remove_evacuated(self):
        evacuated = [evacuee for evacuee in self.evacuees if self.touching_exit(evacuee)]

        for evacuee in evacuated:
            self.grid_world.remove_evacuee(evacuee)

        self.evacuees[:] = [evacuee for evacuee in self.evacuees if evacuee not in evacuated]
        self.evacuated_count += len(evacuated)

    def step(self, evacuee_actions, drone_actions=None):
        if not self.started:
            self.start()

        self.grid_world.step_evacuees(self.evacuees, evacuee_actions)
        self.remove_evacuated()

        if drone_actions is not None:
            self.grid_world.step_drones(self.drones, drone_actions)

        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)
        self.step_count += 1