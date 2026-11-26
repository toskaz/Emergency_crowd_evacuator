
import math

import pygame

from graphics.colors import Color, assignment_color
from map.cell import CellType


class TrainingFloorViewer:
    def __init__(self, fps=0, enabled=True):
        pygame.display.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.fps = fps
        self.windows = {}
        self.backgrounds = {}
        self.cell_sizes = {}
        self.world = None
        self.episode_drone = None
        self.paused = False
        self.running = True
        self.rendering = enabled
        self.control = pygame.Window(
            "Training — live view off", size=(420, 105), hidden=enabled,
        )
        self.toggle_button = pygame.Rect(10, 10, 400, 38)
        self.last_control_update = -200

    def toggle_rendering(self):
        self.rendering = not self.rendering
        if self.rendering:
            self.control.hide()
            for window in self.windows.values():
                window.show()
            if self.windows:
                next(iter(self.windows.values())).focus()
        else:
            for window in self.windows.values():
                window.hide()
            self.control.show()
            self.control.focus()
            self.last_control_update = -200

    def _draw_control(self, env, episode, episodes):
        now = pygame.time.get_ticks()
        if now - self.last_control_update < 200:
            return
        self.last_control_update = now
        surface = self.control.get_surface()
        surface.fill((245, 245, 245))
        pygame.draw.rect(surface, (45, 100, 180), self.toggle_button, border_radius=5)
        surface.blit(self.font.render("Show live floors (V)", True, (255, 255, 255)), (115, 20))
        status = "PAUSED" if self.paused else "View off — training runs without FPS limit"
        surface.blit(self.small_font.render(status, True, Color.WALL.value), (10, 57))
        label = "Eval" if getattr(env, "is_evaluation", False) else "Ep"
        progress = f"{episode}/{episodes}" if episodes is not None else str(episode)
        line = f"{label} {progress} | Step {env.step_count} | Space: pause | Esc: stop/save"
        surface.blit(self.small_font.render(line, True, Color.WALL.value), (10, 80))
        self.control.flip()

    def _sync_floors(self, env):
        world = env.simulation.grid_world
        if self.world is world and self.episode_drone is env.drone:
            return
        floor_count_changed = len(self.windows) != len(world.floors)
        for floor_id in list(self.windows):
            if floor_id not in world.floors:
                self.windows.pop(floor_id).destroy()
        desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
        columns = min(2, len(world.floors))
        rows = math.ceil(len(world.floors) / columns)
        tile_w, tile_h = desktop_w // columns, desktop_h // rows
        self.backgrounds.clear()
        self.cell_sizes.clear()
        for index, (floor_id, floor) in enumerate(world.floors.items()):
            grid = floor.map_grid
            cell = max(2, min(12, (tile_w - 32) // grid.width, (tile_h - 140) // grid.height))
            width, height = max(420, grid.width * cell + 20), grid.height * cell + 110
            title = f"Drone training — floor {floor_id + 1}/{len(world.floors)}"
            if floor_id not in self.windows:
                self.windows[floor_id] = pygame.Window(
                    title, size=(min(width, tile_w - 16), min(height, tile_h - 40)),
                    position=((index % columns) * tile_w, (index // columns) * tile_h + 25),
                    resizable=True,
                )
            else:
                self.windows[floor_id].title = title
                if floor_count_changed:
                    self.windows[floor_id].size = min(width, tile_w - 16), min(height, tile_h - 40)
                    self.windows[floor_id].position = (index % columns) * tile_w, (index // columns) * tile_h + 25
            self.cell_sizes[floor_id] = cell
            surface = pygame.Surface((width, height))
            surface.fill((245, 245, 245))
            for y in range(grid.height):
                for x in range(grid.width):
                    color = {
                        CellType.WALL: Color.WALL.value,
                        CellType.EXIT: Color.EXIT.value,
                        CellType.STAIRS: Color.STAIRS.value,
                    }.get(grid.get(x, y), Color.EMPTY.value)
                    pygame.draw.rect(surface, color, (10 + x * cell, 72 + y * cell, cell, cell))
            for drone, route in env.routes.items():
                segment = []
                for node in [*route, None]:
                    if node is None or node[0] != floor_id:
                        if len(segment) > 1:
                            pygame.draw.lines(surface, assignment_color(drone.drone_id), False, segment, 2)
                        segment = []
                    else:
                        x, y = floor.cluster_grid.clusters[node[1]].get_centroid()
                        segment.append((10 + (x + 0.5) * cell, 72 + (y + 0.5) * cell))
            self.backgrounds[floor_id] = surface
        self.world = world
        self.episode_drone = env.drone

    def draw(self, env, episode, episodes, reward):
        if not self.rendering:
            self._draw_control(env, episode, episodes)
            return
        self._sync_floors(env)
        simulation = env.simulation
        for floor_id, window in self.windows.items():
            surface = self.backgrounds[floor_id].copy()
            cell = self.cell_sizes[floor_id]
            people = [person for person in simulation.evacuees if person.floor_id == floor_id]
            drones = [drone for drone in simulation.drones if drone.floor_id == floor_id]
            label = "Evaluation" if getattr(env, "is_evaluation", False) else "Episode"
            stage = getattr(env, "curriculum_stage", None) or env.layout_name
            phase = getattr(env, "navigation_phase", None) or getattr(env, "evaluation_suite", None)
            if phase:
                stage += f" / {phase}"
            progress = f"{episode}/{episodes}" if episodes is not None else str(episode)
            lines = (
                f"{label} {progress} | Step {env.step_count}/{env.max_episode_steps}",
                f"{stage} | P {len(people)} D {len(drones)} | Reward {reward:.1f}",
                "PAUSED — Space resumes" if self.paused else "V: view off | Space: pause | Esc: stop/save",
            )
            for row, text in enumerate(lines):
                surface.blit(self.font.render(text, True, Color.WALL.value), (10, 6 + row * 21))
            for person in people:
                pygame.draw.circle(surface, assignment_color(person.assigned_drone_id),
                                   (10 + person.grid_x * cell + cell // 2, 72 + person.grid_y * cell + cell // 2),
                                   max(1, cell // 2 - 1))
            for drone in drones:
                rect = pygame.Rect(10 + drone.grid_x * cell, 72 + drone.grid_y * cell, cell, cell)
                pygame.draw.rect(surface, assignment_color(drone.drone_id), rect)
                pygame.draw.rect(surface, Color.WALL.value, rect, 1)
                if drone is env.drone:
                    pygame.draw.rect(surface, (220, 20, 60), rect.inflate(4, 4), 2)
            legend = "Circle: person | Square: drone | Yellow: stairs | Green: exit"
            surface.blit(self.small_font.render(legend, True, Color.WALL.value), (10, surface.get_height() - 25))
            target = window.get_surface()
            pygame.transform.smoothscale(surface, target.get_size(), target)
            window.flip()

    def _events(self):
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.last_control_update = -200
                elif event.key == pygame.K_v:
                    self.toggle_rendering()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                window = getattr(event, "window", None)
                if (
                    getattr(window, "id", window) == self.control.id
                    and self.toggle_button.collidepoint(event.pos)
                ):
                    self.toggle_rendering()

    def update(self, env, episode, episodes, reward, should_stop=None):
        self._events()
        while self.running and self.paused:
            if should_stop is not None and should_stop():
                self.running = False
                break
            self.draw(env, episode, episodes, reward)
            self.clock.tick(30)
            self._events()
        if self.running:
            self.draw(env, episode, episodes, reward)
            if self.rendering and self.fps > 0:
                self.clock.tick(self.fps)
        return self.running

    def close(self):
        for window in self.windows.values():
            window.destroy()
        self.windows.clear()
        self.control.destroy()
        pygame.quit()
