from controls.action import Action
from controls.evacuee_rl import find_leaders, step_towards_exit


MOVE_EVERY = 2
MAX_WAIT = 30


class ScriptedDroneController:
    def __init__(self):
        self.waited = {}

    def reset(self):
        self.waited.clear()

    def get_actions(self, simulation):
        return [self.choose_action(simulation, drone) for drone in simulation.drones]

    def choose_action(self, simulation, drone):
        world = simulation.grid_world
        if world.is_exit(drone.grid_x, drone.grid_y, drone.floor_id):
            return Action.STAY
        if simulation.step_count % MOVE_EVERY != 0:
            return Action.STAY
        if self.should_wait(simulation, drone) and self.waited.get(drone, 0) < MAX_WAIT:
            self.waited[drone] = self.waited.get(drone, 0) + 1
            return Action.STAY

        self.waited[drone] = 0
        if self.stairs_are_shorter(world, drone):
            return Action.USE_STAIRS
        return step_towards_exit(world, drone)

    def should_wait(self, simulation, drone):
        group = [evacuee for evacuee in simulation.evacuees if evacuee.floor_id == drone.floor_id]
        following = [evacuee for evacuee in find_leaders(simulation) if evacuee.floor_id == drone.floor_id]
        return len(following) < len(group) / 2

    def stairs_are_shorter(self, world, drone):
        arrival = world.stair_arrival(drone, "drone")
        if arrival is None:
            return False
        floor_id, x, y = arrival
        here = world.floors[drone.floor_id].distance_map[drone.grid_y][drone.grid_x]
        return world.floors[floor_id].distance_map[y][x] < here
