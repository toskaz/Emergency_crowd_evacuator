import pygame
import pygame_gui


def handle_input(context):
    for event in pygame.event.get():
        if event.type == pygame.VIDEORESIZE:
            context.handle_resize(event.w, event.h)
            continue

        if event.type == pygame.MOUSEBUTTONDOWN and hasattr(context.state, 'handle_raw_mouse'):
            context.state.handle_raw_mouse(context, event)

        consumed = False
        if context.ui_manager is not None:
            consumed = bool(context.ui_manager.process_events(event))

        if event.type == pygame.QUIT:
            if context.window is not None:
                context.window.close()
            context.running = False

        elif event.type in (
            pygame_gui.UI_BUTTON_PRESSED,
            pygame_gui.UI_TEXT_ENTRY_FINISHED,
            pygame_gui.UI_DROP_DOWN_MENU_CHANGED,
            pygame_gui.UI_WINDOW_CLOSE,
        ):
            context.state.handle_ui_event(context, event)

        elif not consumed:
            if event.type == pygame.KEYDOWN:
                context.state.handle_key_event(context, event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                context.state.handle_mouse(context, event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if hasattr(context.state, 'handle_mouse_up'):
                    context.state.handle_mouse_up(context, event)
            elif event.type == pygame.MOUSEMOTION:
                if hasattr(context.state, 'handle_mouse_motion'):
                    context.state.handle_mouse_motion(context, event)
            elif event.type == pygame.MOUSEWHEEL:
                if hasattr(context.state, 'handle_mouse_wheel'):
                    context.state.handle_mouse_wheel(context, event)