from dataclasses import dataclass

from controls.action import Action
from controls.drone_dqn import DQNDroneController
from graphics.window import Window
from simulation import Simulation
from utils.program_state import EditorState


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


class ProgramContext:
    def __init__(self):
        self.simulation = Simulation()
        self.window = Window(self.simulation.grid_world.map_grid)
        self.drone_controller = DQNDroneController()
        self.drone_routes = {}
        self.view_options = ViewOptions()
        self.simulation_options = SimulationOptions()
        self.state = EditorState()
        self.state.enter(self)

    def transition_to(self, state):
        self.state.exit(self)
        self.state = state
        self.state.enter(self)

    def plan_drone_routes(self):
        world = self.simulation.grid_world
        self.drone_routes = world.plan_drone_routes(self.simulation.drones)