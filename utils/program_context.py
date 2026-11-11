from dataclasses import dataclass
import logging

import pygame
import pygame_gui

from controls.action import Action
from controls.drone_dqn import DQNDroneController
from controls.evacuee_rl import QLearningFollowController
from graphics.window import Window
from simulation import Simulation
from utils.config import drawing_size, drone_model_path, q_table_path
from utils.program_state import MainMenuState
import utils.config as config


logger = logging.getLogger(__name__)

MENU_WINDOW_SIZE = (500, 350)
CALIBRATION_WINDOW_SIZE = (1024, 768)


@dataclass
class ViewOptions:
    show_grid: bool = True
    show_clusters: bool = True
    show_cluster_connections: bool = True
    show_drone_routes: bool = True


@dataclass
class SimulationOptions:
    action: Action = Action.STAY
    autonomous_drones: bool = True
    rl_evacuees: bool = True


class ProgramContext:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
        pygame.display.set_caption("Emergency Crowd Evacuator")
        self.ui_manager = pygame_gui.UIManager(MENU_WINDOW_SIZE)

        self.simulation = None
        self.window = None
        self.drone_controller = None
        self.rl_evacuee_controller = None
        self.imported_image_path = None
        self.running = True

        self.drone_routes = {}
        self.link_stairs_mode = False
        self.view_options = ViewOptions()
        self.simulation_options = SimulationOptions()

        self.state = MainMenuState()
        logger.debug("Initial application state: %s", self.state.name)
        self.state.enter(self)

    def _center_camera(self, map_grid, screen_width, screen_height):
        map_pixel_width = map_grid.width * config.drawing_size
        map_pixel_height = map_grid.height * config.drawing_size
        
        config.camera_offset_x = (screen_width - 250 - map_pixel_width) // 2
        config.camera_offset_y = (screen_height - map_pixel_height) // 2

    def _calculate_screen_size(self, map_grid):
            display_info = pygame.display.Info()
            max_w, max_h = display_info.current_w - 50, display_info.current_h - 100
            desired_w = map_grid.width * drawing_size + 250 
            desired_h = max(600, map_grid.height * drawing_size) 
            current_w, current_h = self.screen.get_size() if self.screen else (desired_w, desired_h)
        
            target_w = max(current_w, min(desired_w, max_w))
            target_h = max(current_h, min(desired_h, max_h))
        
            return target_w, target_h

    def _load_controllers(self):
        self.drone_controller = DQNDroneController(training=False)
        if not self.drone_controller.load(drone_model_path):
            logger.warning(
                "Drone model not found at %s; run train_drones.py before simulation",
                drone_model_path,
            )
        self.rl_evacuee_controller = QLearningFollowController()
        self.rl_evacuee_controller.load(q_table_path)

    def initialize_world(self, map_grid):
        if self.simulation is None:
            self.simulation = Simulation()
        self.simulation.grid_world.add_floor_from_map(map_grid)

        screen_width, screen_height = self._calculate_screen_size(map_grid)
    
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.ui_manager = pygame_gui.UIManager((screen_width, screen_height))

        self.window = Window(map_grid)
        self._center_camera(map_grid, screen_width, screen_height)
        if self.drone_controller is None:
            self._load_controllers()

    def initialize_world_from_dict(self, data):
        self.simulation = Simulation()
        self.simulation.grid_world.from_dict(data)

        map_grid = self.simulation.grid_world.map_grid
        screen_width, screen_height = self._calculate_screen_size(map_grid)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.ui_manager = pygame_gui.UIManager((screen_width, screen_height))

        self.window = Window(map_grid)
        self._center_camera(map_grid, screen_width, screen_height)
        if self.drone_controller is None:
            self._load_controllers()

    def update_window_for_active_floor(self):
        if self.simulation is None:
            return
        map_grid = self.simulation.grid_world.map_grid
        screen_width, screen_height = self._calculate_screen_size(map_grid)
        current_w, current_h = self.screen.get_size()
        if (current_w, current_h) != (screen_width, screen_height):
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            if self.ui_manager is not None:
                self.ui_manager.set_window_resolution((screen_width, screen_height))
        if self.window is not None:
            self.window.screen = self.screen

    def transition_to(self, state):
        logger.debug("Application state changed: %s -> %s", self.state.name, state.name)
        self.state.exit(self)
        self.state = state
        self.state.enter(self)

    def plan_drone_routes(self):
        world = self.simulation.grid_world
        self.drone_routes = world.plan_drone_routes(self.simulation.drones)