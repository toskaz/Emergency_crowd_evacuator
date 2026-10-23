import pygame


def handle_input(context):
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            context.window.close()

        elif event.type == pygame.KEYDOWN:
            context.state.handle_key_event(context, event.key)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            context.state.handle_mouse(context, event)