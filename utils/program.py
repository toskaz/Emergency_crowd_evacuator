import pygame

from controls.input import handle_input
from utils.program_context import ProgramContext


class Program:

    def __init__(self):
        self.context = ProgramContext()

    def run(self):
        clock = pygame.time.Clock()
        context = self.context

        while context.window.is_open:
            handle_input(context)
            if not context.window.is_open:
                break

            context.state.update(context)
            context.state.render(context)

            clock.tick(60)
