from dataclasses import dataclass
import logging

from controls.action import Action
from controls.drone_dqn import DQNDroneController
from controls.evacuee_rl import QLearningFollowController
from controls.global_coordinator import GlobalCoordinator
from graphics.window import Window
from simulation import Simulation
from utils.config import drone_model_path, q_table_path
from utils.program_state import EditorState


logger = logging.getLogger(__name__)


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
        self.simulation = Simulation()
        self.window = Window(self.simulation.grid_world.map_grid)
        self.drone_controller = DQNDroneController(training=False)
        if not self.drone_controller.load(drone_model_path):
            logger.warning(
                "Drone model not found at %s; run train_drones.py before simulation",
                drone_model_path,
            )
        self.rl_evacuee_controller = QLearningFollowController()
        self.rl_evacuee_controller.load(q_table_path)
        self.drone_routes = {}
        self.coordinator = GlobalCoordinator(self.simulation.grid_world)
        self.link_stairs_mode = False
        self.view_options = ViewOptions()
        self.simulation_options = SimulationOptions()
        self.state = EditorState()
        logger.debug("Initial application state: %s", self.state.name)
        self.state.enter(self)

    def transition_to(self, state):
        logger.debug("Application state changed: %s -> %s", self.state.name, state.name)
        self.state.exit(self)
        self.state = state
        self.state.enter(self)

    def plan_drone_routes(self):
        self.drone_routes = self.coordinator.plan_initial_routes(
            self.simulation.drones
        )

    def replan_drone_routes_after_hazard(self):
        self.drone_routes, replanned_drones = self.coordinator.on_environment_changed(
            self.simulation.drones
        )
        for drone in replanned_drones:
            self.drone_controller.route_cursors.pop(drone, None)