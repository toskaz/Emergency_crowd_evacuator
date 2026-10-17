import pygame

from graphics.window import Window
from editor.tool import Tool
from controls.action import Action
from simulation import Simulation
from controls.input import handle_input
from controls.action_controller import get_actions


class Program:

    def __init__(self):
        self.simulation = Simulation()
        self.drone_routes = {}

        self.window = Window(
            self.simulation.map_grid,
        )

        self.selected_tool = Tool.WALL
        self.show_grid = True
        self.simulation_running = False
        self.action = Action.STAY
        self.autonomous_drones = True
        self.show_navigation = False
        self.show_cluster_overlay = True

    def run(self):
        clock = pygame.time.Clock()

        while self.window.is_open:

            was_running = self.simulation_running
            handle_input(self)
            if not self.window.is_open:
                break

            if self.simulation_running and not was_running:
                self.plan_drone_routes()

            if self.simulation_running:
                evacuee_actions = get_actions(self.simulation.evacuees, self.action)

                if self.autonomous_drones:
                    drone_actions = [Action.STAY] * len(self.simulation.drones)
                else:
                    drone_actions = get_actions(self.simulation.drones, self.action)

                self.simulation.step(evacuee_actions, drone_actions)

            navigation = self.simulation.navigation_graph if self.show_navigation else None

            self.window.draw(
                self.simulation.map_grid,
                self.simulation.evacuees,
                self.simulation.drones,
                self.selected_tool,
                self.show_grid,
                self.simulation.exits,
                self.status_lines(),
                navigation,
                self.drone_routes,
                self.simulation.cluster_grid,
                self.show_cluster_overlay
            )

            clock.tick(60)

    def plan_drone_routes(self):
        graph = self.simulation.navigation_graph
        exit_cells = [cell for exit_ in self.simulation.exits for cell in exit_.cells]
        self.drone_routes = {}

        for drone in self.simulation.drones:
            start = (drone.grid_x, drone.grid_y)
            _, route = graph.distance_to_closest_source(exit_cells, start)
            self.drone_routes[drone] = route

        self.show_navigation = True

    def status_lines(self):
        drone_mode = "cluster route" if self.autonomous_drones else "manual"
        evacuated = self.simulation.evacuated_count
        total = evacuated + len(self.simulation.evacuees)

        return [
            f"Drones: {drone_mode}",
            f"Step: {self.simulation.step_count}",
            f"Evacuated: {evacuated}/{total}",
        ]
