import pygame

from graphics.window import Window
from editor.tool import Tool
from controls.action import Action
from controls.drone_dqn import DQNDroneController
from simulation import Simulation
from controls.input import handle_input
from controls.action_controller import get_actions


class Program:

    def __init__(self):
        self.simulation = Simulation()
        self.drone_routes = {}
        self.drone_controller = DQNDroneController()

        self.window = Window(
            self.simulation.grid_world.map_grid,
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
                self.simulation.start()
                self.plan_drone_routes()

            if self.simulation_running:
                evacuee_actions = get_actions(self.simulation.evacuees, self.action)

                if self.autonomous_drones:
                    drone_actions = self.drone_controller.get_actions(
                        self.simulation,
                        self.drone_routes,
                    )
                else:
                    drone_actions = get_actions(self.simulation.drones, self.action)

                self.simulation.step(evacuee_actions, drone_actions)
                if self.autonomous_drones:
                    self.drone_controller.learn(self.simulation, self.drone_routes)

            self.window.draw(
                self.simulation.grid_world,
                self.simulation.evacuees,
                self.simulation.drones,
                self.selected_tool,
                self.show_grid,
                self.status_lines(),
                self.show_navigation,
                self.drone_routes,
                self.show_cluster_overlay,
                not self.simulation.started,
            )

            clock.tick(60)

    def plan_drone_routes(self):
        grid_world = self.simulation.grid_world
        self.drone_routes = grid_world.plan_drone_routes(self.simulation.drones)
        self.show_navigation = True

    def status_lines(self):
        drone_mode = "CNN-DQN + cluster route" if self.autonomous_drones else "manual"
        evacuated = self.simulation.evacuated_count
        total = evacuated + len(self.simulation.evacuees)

        return [
            f"Drones: {drone_mode}",
            f"Step: {self.simulation.step_count}",
            f"Evacuated: {evacuated}/{total}",
        ]
