import pygame

from window import Window
from editor.tool import Tool
from action import Action
from simulation import Simulation
from input import handle_input
from action_controller import get_actions
from graph_approach.drone_controller import GraphDroneController


class Program:

    def __init__(self):
        self.simulation = Simulation()
        self.drone_controller = GraphDroneController()

        self.window = Window(
            self.simulation.map_grid,
        )

        self.selected_tool = Tool.WALL
        self.show_grid = True
        self.simulation_running = False
        self.action = Action.STAY
        self.autonomous_drones = True
        self.show_navigation = False

    def run(self):
        clock = pygame.time.Clock()

        while self.window.is_open:

            handle_input(self)
            if not self.window.is_open:
                break

            if self.simulation_running:
                evacuee_actions = get_actions(self.simulation.evacuees, self.action)

                if self.autonomous_drones:
                    drone_actions = self.drone_controller.get_actions(self.simulation)
                else:
                    drone_actions = get_actions(self.simulation.drones, self.action)

                self.simulation.step(evacuee_actions, drone_actions)

            navigation = None
            drone_routes = {}

            if self.show_navigation:
                navigation = self.drone_controller.update_navigation(self.simulation)

                if self.autonomous_drones:
                    drone_routes = {
                        drone: route
                        for drone, route in self.drone_controller.routes.items()
                        if drone in self.simulation.drones
                    }

            self.window.draw(
                self.simulation.map_grid,
                self.simulation.evacuees,
                self.simulation.drones,
                self.selected_tool,
                self.show_grid,
                self.simulation.exits,
                self.status_lines(),
                navigation,
                drone_routes
            )

            clock.tick(60)

    def status_lines(self):
        drone_mode = "Floyd-Warshall" if self.autonomous_drones else "manual"
        evacuated = self.simulation.evacuated_count
        total = evacuated + len(self.simulation.evacuees)

        return [
            f"Drones: {drone_mode}",
            f"Step: {self.simulation.step_count}",
            f"Evacuated: {evacuated}/{total}",
        ]
