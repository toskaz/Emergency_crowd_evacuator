import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controls.action import Action
from controls.drone_dqn import (
    CHANNELS,
    PATCH_SIZE,
    DQNDroneController,
)
from simulation import Simulation
from training.scenario import drone_start_cells, place_random_scenario


class DroneNavigationEnv(gym.Env):
    def __init__(self, max_episode_steps=400):
        super().__init__()
        self.simulation = Simulation()
        self.controller = DQNDroneController(training=True)
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(CHANNELS, PATCH_SIZE, PATCH_SIZE),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(Action))
        self.drone = None
        self.routes = {}
        self.route_ids = set()
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = (
            seed
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.uint32).max))
        )
        rng = random.Random(episode_seed)
        floor_ids = list(self.simulation.grid_world.floors)
        requested_floor = (options or {}).get("floor_id")
        floor_id = requested_floor if requested_floor is not None else rng.choice(floor_ids)

        place_random_scenario(self.simulation, rng, evacuee_count=0, floor_id=floor_id)
        self.drone = self.simulation.drones[0]
        self.routes = self.simulation.grid_world.plan_drone_routes(
            self.simulation.drones
        )
        self.controller.pending_transitions.clear()
        self.controller.route_cursors.clear()
        self.step_count = 0
        observation, self.route_ids = self.controller._observe(
            self.simulation, self.drone, self.routes[self.drone]
        )
        return observation, {"action_mask": self.action_mask()}

    def action_mask(self):
        legal_actions = set(
            self.controller._legal_actions(self.simulation, self.drone)
        )
        return np.array(
            [Action(action) in legal_actions for action in range(len(Action))],
            dtype=np.int8,
        )

    def select_action(self, observation):
        legal_actions = self.controller._legal_actions(self.simulation, self.drone)
        return self.controller._select_action(
            observation,
            legal_actions,
            self.drone,
            self.route_ids,
            self.simulation,
        )

    def step(self, action):
        action = Action(int(action))
        legal_actions = self.controller._legal_actions(self.simulation, self.drone)
        if action not in legal_actions:
            raise ValueError(f"Illegal drone action: {action.name}")

        observation, route_ids = self.controller._observe(
            self.simulation, self.drone, self.routes[self.drone]
        )
        previous_distance = self.controller._distance_at(
            self.simulation.grid_world,
            self.drone.floor_id,
            self.drone.grid_x,
            self.drone.grid_y,
        )
        previous_position = (
            self.drone.floor_id,
            self.drone.grid_x,
            self.drone.grid_y,
        )
        self.controller.pending_transitions[self.drone] = (
            observation,
            int(action),
            previous_distance,
            route_ids,
            previous_position,
        )

        self.simulation.step([], [action])
        self.step_count += 1
        reward = self.controller._transition_reward(
            self.simulation,
            self.drone,
            action,
            previous_distance,
            route_ids,
            previous_position,
        )
        terminated = self.simulation.grid_world.is_exit(
            self.drone.grid_x, self.drone.grid_y, self.drone.floor_id
        )
        truncated = self.step_count >= self.max_episode_steps and not terminated

        next_observation, self.route_ids = self.controller._observe(
            self.simulation, self.drone, self.routes[self.drone]
        )
        self.controller.learn(self.simulation, self.routes)
        return next_observation, reward, terminated, truncated, {
            "action_mask": self.action_mask(),
            "floor_id": self.drone.floor_id,
            "position": (self.drone.grid_x, self.drone.grid_y),
        }