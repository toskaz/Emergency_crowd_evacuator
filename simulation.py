from controls.drone_assignment import assign_drone_groups
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
        self.last_editor_error = None
        self.last_editor_error_timer = 0

    def prepare(self):
        self.grid_world.rebuild_changed_map_data()
        self.evacuees.clear()
        self.drones.clear()
        self.step_count = 0
        self.evacuated_count = 0
        self.started = False
        self.prepared = False
        self.last_editor_error = None
        self.last_editor_error_timer = 0

        for floor in self.grid_world.floors.values():
            floor.evacuee_occupancy_grid = Grid(
                floor.map_grid.width, floor.map_grid.height, None
            )
            floor.drone_occupancy_grid = Grid(
                floor.map_grid.width, floor.map_grid.height, None
            )
            floor.heatmap_grid = Grid(
                floor.map_grid.width, floor.map_grid.height, 0
            )
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

        assign_drone_groups(self.grid_world, self.drones, self.evacuees)
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

        finished_drones = [drone for drone in self.drones if self.touching_exit(drone)]
        for drone in finished_drones:
            self.grid_world.remove_drone(drone)
        self.drones[:] = [drone for drone in self.drones if drone not in finished_drones]

    def step(self, evacuee_actions, drone_actions=None):
        if not self.started:
            self.start()

        for evacuee in self.evacuees:
            fl = self.grid_world.floors.get(evacuee.floor_id)
            if fl and hasattr(fl, "heatmap_grid"):
                cur_v = fl.heatmap_grid.get(evacuee.grid_x, evacuee.grid_y) or 0
                fl.heatmap_grid.set(evacuee.grid_x, evacuee.grid_y, cur_v + 1)

        self.grid_world.step_evacuees(self.evacuees, evacuee_actions)
        self.remove_evacuated()

        if drone_actions is not None:
            self.grid_world.step_drones(self.drones, drone_actions)
            self.remove_evacuated()

        self.grid_world.rebuild_simulation_grid(self.evacuees, self.drones)
        self.step_count += 1