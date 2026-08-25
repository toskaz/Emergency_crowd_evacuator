import pygame

from window import Window
from editor.tool import Tool
from action import Action
from simulation import Simulation
from input import handle_input


class Program:

    def __init__(self):
        self.simulation = Simulation()

        self.window = Window(
            self.simulation.map_grid,
        )

        self.selected_tool = Tool.WALL
        self.show_grid = True
        self.simulation_running = False
        self.action = Action.STAY

    def run(self):
        clock = pygame.time.Clock()

        while self.window.is_open:

            handle_input(self)

            if self.simulation_running:
                self.simulation.step(self.action)

            self.window.draw(self.simulation.map_grid, self.simulation.evacuees, self.selected_tool, self.show_grid)

            clock.tick(60)