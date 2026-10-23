import pygame


def handle_input(program):
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            program.window.close()

        elif event.type == pygame.KEYDOWN:
            program.state.handle_key_event(program, event.key)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            program.state.handle_mouse(program, event)