import pygame
import pygame_gui


def handle_input(context):
    for event in pygame.event.get():
        consumed = False
        if context.ui_manager is not None:
            consumed = bool(context.ui_manager.process_events(event))

        if event.type == pygame.QUIT:
            if context.window is not None:
                context.window.close()
            context.running = False

        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            context.state.handle_ui_event(context, event)

        elif (
            event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION)
            and getattr(context.state, "tracks_mouse_drag", False)
        ):
            context.state.handle_mouse(context, event)

        elif not consumed:
            if event.type == pygame.KEYDOWN:
                context.state.handle_key_event(context, event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                context.state.handle_mouse(context, event)
            elif event.type == pygame.MOUSEWHEEL:
                if hasattr(context.state, 'handle_mouse_wheel'):
                    context.state.handle_mouse_wheel(context, event)