import random
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controls.action import Action
from controls.drone_dqn import (
    CHANNELS,
    PATCH_SIZE,
    N_STEP,
    DQNDroneController,
)
from controls.evacuee_rl import QLearningFollowController
from controls.global_coordinator import GlobalCoordinator
from controls.scripted_drone import ScriptedDroneController
from grids.cluster import ClusterType
from map.cell import CellType
from simulation import Simulation
from training.layouts import (
    CURRICULUM_STAGES, LAYOUT_KINDS,
    generate_curriculum_building, generate_training_building,
)
from utils.config import q_table_path
from training.scenario import (
    CurriculumSpawnError,
    add_training_stairs,
    configure_training_exits,
    load_default_map,
    place_building_scenario,
    place_curriculum_scenario,
    place_random_scenario,
)


DEFAULT_MAX_EPISODE_STEPS = 2000
MAX_LAYOUT_ATTEMPTS = 8


class DroneNavigationEnv(gym.Env):
    def __init__(self, max_episode_steps=None, layout_mode="mixed", training=True, path_only=False, n_step=N_STEP,
                 path_following=False):
        super().__init__()
        if path_following and not path_only:
            raise ValueError("Path-following rewards require path-only pretraining")
        if layout_mode not in ("mixed", "image", *LAYOUT_KINDS):
            raise ValueError(f"Unknown training layout mode: {layout_mode}")
        self.layout_mode = layout_mode
        self.path_only = path_only
        self.layout_name = "image"
        self.curriculum_stage = None
        self.is_evaluation = not training
        self.simulation = Simulation()
        load_default_map(self.simulation)
        self.stair_links = add_training_stairs(self.simulation.grid_world)
        self.image_world_data = self.simulation.grid_world.to_dict()
        self.original_exits = {
            floor_id: [cell for exit_cells in floor.map_grid.exits for cell in exit_cells]
            for floor_id, floor in self.simulation.grid_world.floors.items()
        }
        self.controller = DQNDroneController(training=training, n_step=n_step, path_following=path_following)
        self.controller.path_only = path_only
        self.coordinator = GlobalCoordinator(self.simulation.grid_world)
        self.evacuee_controller = QLearningFollowController()
        self.evacuee_controller.load(q_table_path)
        self.other_drone_controller = ScriptedDroneController()
        self.fixed_episode_steps = max_episode_steps
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else DEFAULT_MAX_EPISODE_STEPS
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
        options = options or {}
        for attempt in range(MAX_LAYOUT_ATTEMPTS):
            try:
                observation, info = self._reset_episode(rng, options)
                info["layout_attempts"] = attempt + 1
                info["episode_seed"] = episode_seed
                return observation, info
            except CurriculumSpawnError:
                if options.get("curriculum_stage") not in CURRICULUM_STAGES[:-1] or attempt == MAX_LAYOUT_ATTEMPTS - 1:
                    raise

    def _reset_episode(self, rng, options):
        if self.path_only:
            if options.get("evacuee_count", 0) != 0 or options.get("drone_count", 1) != 1:
                raise ValueError("Path-only episodes require zero evacuees and exactly one drone")
            options = dict(options, evacuee_count=0, drone_count=1)
        self.curriculum_stage = options.get("curriculum_stage")
        if self.curriculum_stage is not None and self.curriculum_stage not in CURRICULUM_STAGES:
            raise ValueError(f"Unknown curriculum stage: {self.curriculum_stage}")
        introductory = self.curriculum_stage in CURRICULUM_STAGES[:-1]
        if introductory or self.curriculum_stage == "mixed" or self.layout_mode != "image" or "layout" in options:
            self._build_episode_layout(rng, options)
        floor_ids = list(self.simulation.grid_world.floors)
        requested_floor = options.get("floor_id")
        floor_id = requested_floor if requested_floor is not None else rng.choice(floor_ids)
        self.requires_stairs = options.get(
            "require_stairs", self.curriculum_stage == "stairs" if introductory
            else bool(self.stair_links) and rng.random() < 0.8
        )
        self.exit_floor_id = None
        if self.requires_stairs:
            exit_floors = [
                candidate for candidate in floor_ids
                if candidate != floor_id and self.original_exits[candidate]
            ]
            if not self.stair_links or not exit_floors:
                raise ValueError("Stair training requires linked floors and an exit on another floor")
            self.exit_floor_id = options.get("exit_floor_id")
            if self.exit_floor_id is not None and self.exit_floor_id not in exit_floors:
                raise ValueError("The exit floor must be a different floor with an exit")
            if self.exit_floor_id is None:
                self.exit_floor_id = (
                    max(exit_floors, key=lambda candidate: abs(candidate - floor_id))
                    if rng.random() < 0.6 else rng.choice(exit_floors)
                )
        configure_training_exits(
            self.simulation.grid_world, self.original_exits, self.exit_floor_id
        )
        self.stair_links = self.simulation.grid_world._snapshot_stair_links()

        varied = self.layout_mode != "image" or "layout" in options or self.curriculum_stage == "mixed"
        evacuee_count = options.get(
            "evacuee_count",
            rng.randint(1, 4) if self.curriculum_stage == "basic" else
            rng.randint(5, 12) if introductory else
            rng.choice((1, 3, 6, 12, 18, 24, 36, 48)) if varied else rng.randint(1, 12),
        )
        drone_count = options.get(
            "drone_count",
            1 if introductory else
            rng.randint(1, min(4, max(1, evacuee_count))) if varied else rng.choice((1, 2)),
        )
        self.crowd_distribution = options.get(
            "distribution", "clustered" if introductory else rng.choice(("clustered", "scattered", "line"))
        )
        if introductory:
            if drone_count != 1 or self.crowd_distribution != "clustered":
                raise ValueError("Introductory lessons use one drone and a visible group")
            distances = {"basic": (20, 30), "turns": (30, 65), "stairs": (30, 80)}
            place_curriculum_scenario(
                self.simulation, rng, evacuee_count, floor_id,
                *distances[self.curriculum_stage],
                require_detour=self.curriculum_stage == "turns",
            )
        elif varied:
            place_building_scenario(
                self.simulation, rng, evacuee_count=evacuee_count, floor_id=floor_id,
                drone_count=drone_count, distribution=self.crowd_distribution,
            )
        else:
            place_random_scenario(
                self.simulation, rng, evacuee_count=evacuee_count,
                floor_id=floor_id, spread_out=True, drone_count=drone_count,
            )
        self.evacuee_controller.reset()
        self.other_drone_controller.reset()
        self.drone = self.simulation.drones[0]
        self.initial_group = self.controller._assigned_group(self.simulation, self.drone)
        self.initial_evacuee_count = len(self.initial_group)
        self.routes = self.coordinator.plan_initial_routes(
            self.simulation.drones
        )
        self.controller.reset()
        self.step_count = 0
        self.position_history = deque([
            (self.drone.floor_id, self.drone.grid_x, self.drone.grid_y)
        ], maxlen=2)
        self.stair_transitions = 0
        self.drone_exit_step = None
        self.group_exit_step = None
        self.route_changes = 0
        self.subgoals_completed = 0
        scheduled_change = (rng.randint(8, 35) if self.curriculum_stage in ("turns", "mixed")
                            and rng.random() < 0.3 else None)
        self.route_change_step = options.get("route_change_step", scheduled_change)
        if self.route_change_step is not None and self.route_change_step < 1:
            raise ValueError("route_change_step must be positive or None")
        observation, self.route_ids = self.controller._observe(
            self.simulation, self.drone, self.routes[self.drone]
        )
        route = self.routes[self.drone]
        if not route:
            raise ValueError("Training scenario has no assigned exit route")
        self.initial_route_distance = 0
        for first, second in zip(route, route[1:]):
            if first[0] != second[0]:
                self.initial_route_distance += 1
            else:
                clusters = self.simulation.grid_world.floors[first[0]].cluster_grid.clusters
                x1, y1 = clusters[first[1]].get_centroid()
                x2, y2 = clusters[second[1]].get_centroid()
                self.initial_route_distance += abs(x2 - x1) + abs(y2 - y1)
        if self.fixed_episode_steps is None:
            self.max_episode_steps = max(
                DEFAULT_MAX_EPISODE_STEPS,
                int(8 * self.initial_route_distance + 20 * self.initial_evacuee_count),
            )
            if self.curriculum_stage in ("basic", "turns"):
                self.max_episode_steps = 400 if self.curriculum_stage == "basic" else 500
        return observation, {
            "action_mask": self.action_mask(),
            "requires_stairs": self.requires_stairs,
            "exit_floor_id": self.exit_floor_id,
            "layout": self.layout_name,
            "floor_count": len(floor_ids),
            "evacuee_count": len(self.simulation.evacuees),
            "drone_count": len(self.simulation.drones),
            "distribution": self.crowd_distribution,
            "max_episode_steps": self.max_episode_steps,
            "route_distance": self.initial_route_distance,
            "curriculum_stage": self.curriculum_stage,
            "path_only": self.path_only,
            "route_change_step": self.route_change_step,
        }

    def replan_for_blocked_passage(self):
        if self.drone not in self.simulation.drones:
            return False
        world = self.simulation.grid_world
        original = {drone: list(route) for drone, route in self.routes.items()}
        route = original[self.drone]
        cursor = max(0, self.controller.route_cursors.get(self.drone, 0) - 1)
        for first, second in list(zip(route, route[1:]))[cursor:cursor + 3]:
            if first[0] != second[0]:
                continue
            floor = world.floors[first[0]]
            clusters = floor.cluster_grid.clusters
            if clusters[second[1]].type not in (ClusterType.NORMAL, ClusterType.DOOR):
                continue
            first_cells = set(clusters[first[1]].cells)
            border = [(x, y) for x, y in clusters[second[1]].cells
                      if world.can_move_to(x, y, first[0]) and any(
                          (x + dx, y + dy) in first_cells
                          for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
            if not border or any(
                floor.map_grid.get(x, y) != CellType.EMPTY
                or floor.evacuee_occupancy_grid.get(x, y) is not None
                or floor.drone_occupancy_grid.get(x, y) is not None
                for x, y in border
            ):
                continue
            blocked = []
            accepted = False
            try:
                for x, y in border:
                    if world.set_temporary_wall(first[0], x, y, True):
                        blocked.append((x, y))
                if len(blocked) != len(border):
                    continue
                actors = [*self.simulation.drones, *self.simulation.evacuees]
                if any(not np.isfinite(world.floors[actor.floor_id].distance_map[actor.grid_y][actor.grid_x])
                       for actor in actors):
                    continue
                self.coordinator.update_graph_passability()
                exits = self.coordinator.get_exit_nodes()
                if any(not world.navigation_graph.distance_to_closest_source(
                    exits, self.coordinator.get_drone_cluster(drone)
                )[1] for drone in self.simulation.drones):
                    continue
                routes, changed = self.coordinator.on_environment_changed(self.simulation.drones)
                if (self.drone in changed and routes[self.drone] and routes[self.drone] != original[self.drone]
                        and all(routes.get(drone) for drone in self.simulation.drones)):
                    self.routes = routes
                    _, self.route_ids = self.controller._observe(self.simulation, self.drone, self.routes[self.drone])
                    self.route_changes += 1
                    accepted = True
                    return True
            finally:
                if not accepted:
                    for x, y in blocked:
                        world.set_temporary_wall(first[0], x, y, False)
                    self.coordinator.drone_routes = {drone: list(path) for drone, path in original.items()}
                    self.routes = self.coordinator.drone_routes
                    self.coordinator.update_graph_passability()
        return False

    def _build_episode_layout(self, rng, options):
        stage = self.curriculum_stage
        kind = stage if stage in CURRICULUM_STAGES[:-1] else options.get("layout", self.layout_mode)
        if kind == "mixed":
            kind = rng.choices(("image", *LAYOUT_KINDS), weights=(2, 3, 2, 3))[0]
        self.simulation = Simulation()
        if kind in CURRICULUM_STAGES[:-1]:
            generate_curriculum_building(self.simulation, rng, kind)
        elif kind == "image":
            self.simulation.grid_world.from_dict(self.image_world_data)
        else:
            generate_training_building(
                self.simulation, rng, kind=kind, floor_count=options.get("floor_count")
            )
        self.layout_name = kind
        world = self.simulation.grid_world
        self.stair_links = world._snapshot_stair_links()
        self.original_exits = {
            floor_id: [cell for exit_cells in floor.map_grid.exits for cell in exit_cells]
            for floor_id, floor in world.floors.items()
        }
        self.coordinator = GlobalCoordinator(world)

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
        transition = self.controller.record_transition(
            self.simulation, self.drone, observation, action, route_ids
        )
        previous_floor_id = self.drone.floor_id

        evacuee_actions = self.evacuee_controller.get_actions(self.simulation)
        drone_actions = [
            action if drone is self.drone
            else self.other_drone_controller.choose_action(self.simulation, drone)
            for drone in self.simulation.drones
        ]
        self.simulation.step(evacuee_actions, drone_actions)
        self.stair_transitions += self.drone.floor_id != previous_floor_id
        self.step_count += 1
        components = self.controller.reward_components(self.simulation, self.drone, transition)
        reward = sum(components.values())
        subgoal_reached = (self.controller.path_following
                           and self.controller.reached_route_subgoal(self.simulation, self.drone, transition))
        self.subgoals_completed += subgoal_reached
        success = self.controller.mission_finished(self.simulation, self.drone)
        terminated = success or self.controller.mission_failed(self.simulation, self.drone)
        truncated = self.step_count >= self.max_episode_steps and not terminated
        if self.controller.training:
            self.controller.learn(self.simulation, self.routes)
            if truncated:
                self.controller.finish_episode()
        else:
            self.controller._commit_transition(self.simulation, self.drone, transition)
            self.controller.pending_transitions.clear()
        if self.drone_exit_step is None and self.controller._at_assigned_exit(
                self.simulation.grid_world, self.drone, self.routes[self.drone]):
            self.drone_exit_step = self.step_count
        if self.initial_group and self.group_exit_step is None and self.controller.group_finished(self.simulation, self.drone):
            self.group_exit_step = self.step_count
        route_changed = (not terminated and not truncated and self.step_count == self.route_change_step
                         and self.replan_for_blocked_passage())
        next_observation, self.route_ids = self.controller._observe(
            self.simulation, self.drone, self.routes[self.drone]
        )
        position = (self.drone.floor_id, self.drone.grid_x, self.drone.grid_y)
        stationary = position == self.position_history[-1]
        oscillation = len(self.position_history) == 2 and position == self.position_history[0] and not stationary
        self.position_history.append(position)
        return next_observation, reward, terminated, truncated, {
            "action_mask": self.action_mask(),
            "floor_id": self.drone.floor_id,
            "position": (self.drone.grid_x, self.drone.grid_y),
            "success": success,
            "requires_stairs": self.requires_stairs,
            "exit_floor_id": self.exit_floor_id,
            "stair_transitions": self.stair_transitions,
            "layout": self.layout_name,
            "floor_count": len(self.simulation.grid_world.floors),
            "max_episode_steps": self.max_episode_steps,
            "curriculum_stage": self.curriculum_stage,
            "path_only": self.path_only,
            "route_changed": route_changed,
            "route_changes": self.route_changes,
            "learning_objective": self.controller.learning_objective,
            "subgoal_completed": subgoal_reached,
            "subgoals_completed": self.subgoals_completed,
            "drone_exit_step": self.drone_exit_step,
            "group_exit_step": self.group_exit_step,
            "reward_components": components,
            "stay_action": action == Action.STAY,
            "stationary": stationary,
            "oscillation": oscillation,
            "evacuation_rate": (
                sum(evacuee not in self.simulation.evacuees for evacuee in self.initial_group)
                / self.initial_evacuee_count
                if self.initial_evacuee_count else float(success)
            ),
        }