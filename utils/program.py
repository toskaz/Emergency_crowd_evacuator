import pygame

from graphics.window import Window
from editor.tool import Tool
from controls.action import Action
from controls.drone_dqn import DQNDroneController
from simulation import Simulation
from controls.input import handle_input
from utils.program_state import EditorState


class Program:

    def __init__(self):
        self.simulation = Simulation()
        self.state = EditorState()
        self.drone_routes = {}
        self.drone_controller = DQNDroneController()

        self.window = Window(
            self.simulation.grid_world.map_grid,
        )

        self.selected_tool = Tool.WALL
        self.show_grid = True
        self.action = Action.STAY
        self.autonomous_drones = True
        self.show_cluster_overlay = True
        self.show_cluster_connections = True
        self.show_drone_routes = True

    def run(self):
        clock = pygame.time.Clock()

        while self.window.is_open:
            handle_input(self)
            if not self.window.is_open:
                break

            self.state.update(self)
            self.state.render(self)

            clock.tick(60)

    def transition_to(self, state):
        self.state.exit(self)
        self.state = state
        self.state.enter(self)

    def plan_drone_routes(self):
        grid_world = self.simulation.grid_world
        self.drone_routes = grid_world.plan_drone_routes(self.simulation.drones)

    def status_lines(self):
        drone_mode = "CNN-DQN + cluster route" if self.autonomous_drones else "manual"
        evacuated = self.simulation.evacuated_count
        total = evacuated + len(self.simulation.evacuees)

        return [
            f"State: {self.state.name}",
            f"Drones: {drone_mode}",
            f"Step: {self.simulation.step_count}",
            f"Evacuated: {evacuated}/{total}",
        ]
