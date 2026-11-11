import pygame

from controls.input import handle_input
from utils.program_context import ProgramContext


class Program:

    def __init__(self):
        self.context = ProgramContext()

    def run(self):
        clock = pygame.time.Clock()
        context = self.context

        while context.running:
            time_delta = clock.tick(60) / 1000.0

            handle_input(context)

            if not context.running:
                break

            if context.ui_manager is not None:
                context.ui_manager.update(time_delta)

            context.state.update(context)
            context.state.render(context)

            screen = (
                context.window.screen
                if context.window is not None
                else context.screen
            )
            if context.ui_manager is not None and screen is not None:
                context.ui_manager.draw_ui(screen)

            pygame.display.flip()

        pygame.quit()
