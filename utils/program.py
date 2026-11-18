import pygame

from controls.input import handle_input
from utils.program_context import ProgramContext


BASE_SIMULATION_INTERVAL = 0.25


def simulation_tick_interval(speed_multiplier):
    if speed_multiplier not in ProgramContext.speed_multipliers:
        raise ValueError(f"Unsupported simulation speed: {speed_multiplier}")
    return BASE_SIMULATION_INTERVAL / speed_multiplier


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

            if context.state.name == "simulation_running":
                context.simulation_tick_accumulator += time_delta
                interval = simulation_tick_interval(context.speed_multiplier)
                while (
                    context.simulation_tick_accumulator >= interval
                    and context.state.name == "simulation_running"
                ):
                    context.state.update(context)
                    context.simulation_tick_accumulator -= interval
            else:
                context.simulation_tick_accumulator = 0.0
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
