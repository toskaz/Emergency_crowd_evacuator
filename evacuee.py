import pygame


class Evacuee:

    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.speed = 100