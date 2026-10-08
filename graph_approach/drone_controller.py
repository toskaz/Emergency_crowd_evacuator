import math

from action import Action
from config import navigation_block_size, drone_lead_distance
from graph_approach.navigation_graph import NavigationGraph
from simulation import ACTION_DIRECTIONS

DIRECTION_ACTIONS = {direction: action for action, direction in ACTION_DIRECTIONS.items()}


# How the drones behave in the graph approach. Every step each drone takes the closest person
# that nobody else is looking after, works out which exit is best for them and then hovers
# a few cells ahead of them on the way there, so the person can see where to go.
class GraphDroneController:

    def __init__(self, block_size=navigation_block_size, lead_distance=drone_lead_distance):
        self.block_size = block_size
        self.lead_distance = lead_distance

        self.navigation = None
        self.navigation_version = None
        self.routes = {}

    def update_navigation(self, simulation):
        # Building the graph is the slow part, so we only redo it when the map has actually changed.
        if self.navigation_version != simulation.map_version:
            self.navigation = NavigationGraph(simulation.map_grid, self.block_size)
            self.navigation_version = simulation.map_version

        return self.navigation

    def get_actions(self, simulation):
        navigation = self.update_navigation(simulation)
        assignments = self.assign_evacuees(navigation, simulation)

        exit_load = {
            exit_: len(simulation.evacuees_in_exit(exit_))
            for exit_ in simulation.exits
        }

        actions = []
        self.routes = {}

        for drone in simulation.drones:
            evacuee = assignments.get(drone)

            if evacuee is None:
                actions.append(Action.STAY)
                continue

            position = (drone.grid_x, drone.grid_y)
            goal = self.guide_point(navigation, simulation.exits, evacuee, exit_load)

            actions.append(action_towards(position, navigation.next_cell(position, goal)))
            self.routes[drone] = self.route_cells(navigation, position, goal)

        return actions

    # Pairs drones with people, closest pairs first, and nobody gets two drones. People who are
    # already standing in an exit are left out, they're just waiting for their turn to get out.
    def assign_evacuees(self, navigation, simulation):
        evacuees = [
            evacuee for evacuee in simulation.evacuees
            if not simulation.touching_exit(evacuee)
        ]

        candidates = []

        for drone_index, drone in enumerate(simulation.drones):
            drone_cell = (drone.grid_x, drone.grid_y)

            for evacuee_index, evacuee in enumerate(evacuees):
                evacuee_cell = (evacuee.grid_x, evacuee.grid_y)
                distance = navigation.distance(drone_cell, evacuee_cell)

                if math.isfinite(distance):
                    candidates.append((distance, manhattan(drone_cell, evacuee_cell), drone_index, evacuee_index))

        candidates.sort()

        assignments = {}
        taken = set()

        for _, _, drone_index, evacuee_index in candidates:
            drone = simulation.drones[drone_index]

            if drone in assignments or evacuee_index in taken:
                continue

            assignments[drone] = evacuees[evacuee_index]
            taken.add(evacuee_index)

        return assignments

    def guide_point(self, navigation, exits, evacuee, exit_load):
        position = (evacuee.grid_x, evacuee.grid_y)
        exit_cell = self.choose_exit_cell(navigation, exits, position, exit_load)

        if exit_cell is None:
            return position

        point = position

        for _ in range(self.lead_distance):
            point = navigation.next_cell(point, exit_cell)

        return point

    # The closest exit isn't always the quickest one. If people are already queuing there, a further
    # exit can be faster, so we add the expected waiting time (queue / capacity) to the walking
    # distance and go with the exit that has the lowest total.
    def choose_exit_cell(self, navigation, exits, position, exit_load):
        best = None

        for exit_ in exits:
            cell = min(
                exit_.cells,
                key=lambda exit_cell: (navigation.distance(position, exit_cell), manhattan(position, exit_cell))
            )
            distance = navigation.distance(position, cell)

            if not math.isfinite(distance):
                continue

            cost = distance + exit_load[exit_] / exit_.capacity

            if best is None or cost < best[0]:
                best = (cost, exit_, cell)

        if best is None:
            return None

        _, exit_, cell = best
        exit_load[exit_] += 1

        return cell

    def route_cells(self, navigation, start, goal):
        nodes = navigation.route(start, goal)

        return [start] + [navigation.representatives[node] for node in nodes[1:-1]] + [goal]


def action_towards(position, next_cell):
    return DIRECTION_ACTIONS[(next_cell[0] - position[0], next_cell[1] - position[1])]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
