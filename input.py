import pygame
from editor.editor import handle_map_editor
from editor.tool import Tool
from action import Action


def handle_input(program):
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            program.window.close()

        elif event.type == pygame.KEYDOWN:
            handle_keyboard(program, event)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_mouse(program, event)


def handle_keyboard(program, event):

    if event.key == pygame.K_1:
        program.selected_tool = Tool.WALL

    elif event.key == pygame.K_2:
        program.selected_tool = Tool.EVACUEE

    elif event.key == pygame.K_3:
        program.selected_tool = Tool.EXIT

    elif event.key == pygame.K_g:
        program.show_grid = not program.show_grid

    elif event.key == pygame.K_p:
        program.simulation_running = not program.simulation_running

    elif event.key == pygame.K_w:
        program.action = Action.UP

    elif event.key == pygame.K_s:
        program.action = Action.DOWN

    elif event.key == pygame.K_a:
        program.action = Action.LEFT

    elif event.key == pygame.K_d:
        program.action = Action.RIGHT


def handle_mouse(program, event):
    handle_map_editor(program.simulation, event.pos, event.button, program.selected_tool)