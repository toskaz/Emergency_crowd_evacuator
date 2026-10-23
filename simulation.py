from grids.grid_world import Grid_World
from entities.evacuee import Evacuee
from entities.drone import Drone
from map.cell import CellType


class Simulation:

    def __init__(self, map_grid=None):
        self.grid_world = Grid_World(map_grid)
        self.evacuees = []
        self.drones = []

        self.step_count = 0
        self.evacuated_count = 0
        self.started = False

    def start(self):
        if self.started:
            return

        map_grid = self.grid_world.map_grid
        for y in range(map_grid.height):
            for x in range(map_grid.width):
                cell_type = map_grid.get(x, y)
                if cell_type == CellType.EVACUEE:
                    evacuee = Evacuee(x, y)
                    self.evacuees.append(evacuee)
                    self.grid_world.add_evacuee(evacuee)
                elif cell_type == CellType.DRONE:
                    drone = Drone(x, y)
                    self.drones.append(drone)
                    self.grid_world.add_drone(drone)

        self.started = True
        self.grid_world.rebuild_map_data()
        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)

    def add_evacuee(self, grid_x, grid_y):
        evacuee = Evacuee(grid_x, grid_y)

        self.evacuees.append(evacuee)
        self.grid_world.add_evacuee(evacuee)
        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)

    def remove_evacuee(self, grid_x, grid_y):
        evacuee = self.grid_world.occupancy_grid.get(grid_x, grid_y)

        if evacuee is None:
            return

        self.evacuees.remove(evacuee)
        self.grid_world.remove_evacuee(evacuee)
        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)

    def add_drone(self, grid_x, grid_y):
        drone = Drone(grid_x, grid_y)
        self.drones.append(drone)
        self.grid_world.add_drone(drone)
        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)

    def remove_drone(self, grid_x, grid_y):
        drone = self.grid_world.drone_occupancy_grid.get(grid_x, grid_y)

        if drone is None:
            return

        self.drones.remove(drone)
        self.grid_world.remove_drone(drone)
        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)

    def touching_exit(self, evacuee):
        return self.grid_world.is_exit(evacuee.grid_x, evacuee.grid_y)

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