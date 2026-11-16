from collections import deque
from dataclasses import dataclass
import logging
import random

import numpy as np
import torch
from torch import nn

from controls.action import Action
from controls.line_of_sight import can_see
from map.cell import CellType
from utils.config import drone_visible_evacuee_target, drone_vision_radius


logger = logging.getLogger(__name__)


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

PATCH_RADIUS = drone_vision_radius
PATCH_SIZE = 2 * PATCH_RADIUS + 1
CHANNELS = 12
ASSIGNED_EVACUEES_CHANNEL = 10
VISIBLE_GROUP_COUNT_CHANNEL = 11
OBSERVATION_VERSION = 3
WALL_CLEARANCE_RADIUS = 3
WALL_PROXIMITY_PENALTY = 0.005
DRONE_PROGRESS_REWARD = 0.05
DRONE_EXIT_REWARD = 10.0
MISSION_FAILURE_PENALTY = 20.0
REWARD_VERSION = 3
PATH_FOLLOWING_REWARD_VERSION = 4
PATH_FOLLOWING_PROGRESS_REWARD = 1.0
PATH_FOLLOWING_CLUSTER_REWARD = 10.0
TARGET_SYNC_STEPS = 2000
MAX_TRAINING_Q = 100.0
STEP_COST = 0.06
VISIBILITY_REWARD = 0.002
FOLLOWER_PROGRESS_REWARD = 0.025
FOLLOWER_EXIT_REWARD = 10.0
GROUP_COMPLETION_REWARD = 10.0
FAILED_MOVE_GRACE = 2
FAILED_MOVE_PENALTY = 0.1
DISCOUNT = 0.995
N_STEP = 5


@dataclass
class DroneTransition:
    state: np.ndarray
    action: int
    position: tuple
    target_node: tuple | None
    route: tuple
    route_cursor: int
    visible_followers: dict
    initial_group: tuple
    evacuated_before: frozenset
    active: bool
    previous_failure: tuple | None
    distance_fields: dict | None = None
    drone_distance: float = float("inf")


class DroneQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * PATCH_SIZE * PATCH_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, len(Action)),
        )

    def forward(self, state):
        return self.features(state)


class DQNDroneController:
    def __init__(self, training=True, n_step=N_STEP, path_following=False):
        if not isinstance(n_step, int) or n_step < 1:
            raise ValueError("n_step must be a positive integer")
        self.device = torch.device("cpu")
        self.training = training
        self.epsilon = 0.15
        self.policy_network = DroneQNetwork().to(self.device)
        self.target_network = None
        self.optimizer = None
        if training:
            self.target_network = DroneQNetwork().to(self.device)
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.target_network.eval()
            self.optimizer = torch.optim.Adam(
                self.policy_network.parameters(), lr=0.0001
            )
        else:
            self.policy_network.eval()
        self.replay_buffer = deque(maxlen=20000)
        self.successful_replay = deque(maxlen=5000)
        self.episode_replay = {}
        self.n_step = 1 if path_following else n_step
        self.discount = 0.0 if path_following else DISCOUNT
        self.return_queues = {}
        self.pending_transitions = {}
        self.route_cursors = {}
        self.route_targets = {}
        self.assigned_routes = {}
        self.initial_groups = {}
        self.failed_moves = {}
        self.decision_count = 0
        self.training_steps = 0
        self.last_loss = None
        self.last_q_abs_max = None
        self.last_target_abs_max = None
        self.max_training_q = MAX_TRAINING_Q if training else None
        self.target_sync_steps = TARGET_SYNC_STEPS
        self.path_only = False
        self.path_following = path_following
        self.reward_distance_fields = {}
        self.reward_world = None
        self.reward_topology = ()

    def reset(self):
        self.finish_episode()
        self.pending_transitions.clear()
        self.route_cursors.clear()
        self.route_targets.clear()
        self.assigned_routes.clear()
        self.initial_groups.clear()
        self.failed_moves.clear()
        self.episode_replay.clear()
        self.reward_distance_fields.clear()
        self.reward_world = None
        self.reward_topology = ()

    def get_actions(self, simulation, routes):
        actions = []

        for drone in simulation.drones:
            world = simulation.grid_world
            floor = world.floors[drone.floor_id]
            if floor.map_grid.get(drone.grid_x, drone.grid_y) == CellType.EXIT:
                self.pending_transitions.pop(drone, None)
                actions.append(Action.STAY)
                continue

            route = routes.get(drone, [])
            state, route_ids = self._observe(simulation, drone, route)
            legal_actions = self._legal_actions(simulation, drone)
            action = self._select_action(state, legal_actions, drone, route_ids, simulation)
            if self.training:
                self.record_transition(simulation, drone, state, action, route_ids)
            actions.append(action)

        return actions

    def _assigned_group(self, simulation, drone):
        return [
            evacuee for evacuee in simulation.evacuees
            if drone.drone_id is not None and evacuee.assigned_drone_id == drone.drone_id
        ]

    def _visible_evacuees(self, simulation, drone):
        if drone not in simulation.drones:
            return []
        return [
            evacuee for evacuee in simulation.evacuees
            if evacuee.floor_id == drone.floor_id
            and can_see(
                simulation.grid_world,
                drone.floor_id,
                drone.grid_x,
                drone.grid_y,
                evacuee.grid_x,
                evacuee.grid_y,
                radius=PATCH_RADIUS,
            )
        ]

    def record_transition(self, simulation, drone, state, action, route_ids):
        target_node = self.route_targets.get(drone)
        distance_fields = self._reward_distances(simulation.grid_world, target_node)
        initial_group = self.initial_groups.setdefault(drone, tuple(self._assigned_group(simulation, drone)))
        visible_followers = {
            evacuee: self._field_distance(distance_fields, (evacuee.floor_id, evacuee.grid_x, evacuee.grid_y))
            for evacuee in self._visible_evacuees(simulation, drone)
            if drone.drone_id is not None and evacuee.assigned_drone_id == drone.drone_id
        }
        position = (drone.floor_id, drone.grid_x, drone.grid_y)
        transition = DroneTransition(
            state=state, action=int(action), position=position, target_node=target_node,
            route=self.assigned_routes.get(drone, ()),
            route_cursor=self.route_cursors.get(drone, 0),
            visible_followers=visible_followers, initial_group=initial_group,
            evacuated_before=self._evacuated_group(simulation, initial_group),
            active=drone in simulation.drones,
            previous_failure=self.failed_moves.get(drone),
            distance_fields=distance_fields,
            drone_distance=self._field_distance(distance_fields, position),
        )
        self.pending_transitions[drone] = transition
        return transition

    def learn(self, simulation, routes):
        if not self.training:
            return
        if self.path_following and not self.path_only:
            raise ValueError("Path-following rewards are only supported for path-only pretraining")

        for drone, transition in list(self.pending_transitions.items()):
            reward = self._transition_reward(simulation, drone, transition)
            success = self.mission_finished(simulation, drone)
            mission_done = success or self.mission_failed(simulation, drone)
            self._commit_transition(simulation, drone, transition)

            next_state, _ = self._observe(simulation, drone, routes.get(drone, []))
            route_changed = tuple(routes.get(drone, ())) != transition.route
            subgoal_reached = (self.path_following and not route_changed
                               and self.reached_route_subgoal(simulation, drone, transition))
            done = mission_done or subgoal_reached or (self.path_following and route_changed)
            next_legal = self._legal_actions(simulation, drone)
            legal_mask = np.zeros(len(Action), dtype=np.float32)
            legal_mask[[int(candidate) for candidate in next_legal]] = 1.0
            experience = (
                transition.state, transition.action, reward, next_state, done, legal_mask
            )
            queue = self.return_queues.setdefault(drone, deque())
            queue.append(experience)
            if len(queue) >= self.n_step:
                self._emit_return(drone)
            if done:
                self._flush_returns(drone)
                if success or subgoal_reached:
                    self.successful_replay.extend(self.episode_replay.get(drone, ()))
                self.episode_replay.pop(drone, None)

        self.pending_transitions.clear()

        if len(self.replay_buffer) >= 64:
            self.training_steps += 1
            if self.training_steps % 4 == 0:
                self._optimize()

    def _emit_return(self, drone):
        queue = self.return_queues[drone]
        window = list(queue)[:self.n_step]
        first, last = window[0], window[-1]
        reward = sum(self.discount ** index * item[2] for index, item in enumerate(window))
        experience = (first[0], first[1], reward, last[3], last[4], last[5], self.discount ** len(window))
        self.replay_buffer.append(experience)
        self.episode_replay.setdefault(drone, []).append(experience)
        queue.popleft()

    def _flush_returns(self, drone, *, terminal=False):
        queue = self.return_queues.get(drone)
        if terminal and queue:
            last = queue[-1]
            queue[-1] = (*last[:4], True, last[5])
        while self.return_queues.get(drone):
            self._emit_return(drone)
        self.return_queues.pop(drone, None)

    def finish_episode(self):
        for drone in list(self.return_queues):
            self._flush_returns(drone)
        self.episode_replay.clear()

    @property
    def learning_objective(self):
        return "immediate_path_following" if self.path_following else "mission"

    @property
    def reward_version(self):
        return PATH_FOLLOWING_REWARD_VERSION if self.path_following else REWARD_VERSION

    @staticmethod
    def reached_route_subgoal(simulation, drone, transition):
        if not transition.active or not transition.route:
            return False
        floor = simulation.grid_world.floors[drone.floor_id]
        node = (drone.floor_id, floor.cluster_grid.get(drone.grid_x, drone.grid_y))
        return node in transition.route[transition.route_cursor:]

    def _commit_transition(self, simulation, drone, transition):
        failure_count = self._failure_count(simulation, drone, transition)
        if failure_count:
            self.failed_moves[drone] = (
                (transition.position, transition.action), simulation.step_count, failure_count
            )
        else:
            self.failed_moves.pop(drone, None)

    def _transition_reward(self, simulation, drone, transition):
        return sum(self.reward_components(simulation, drone, transition).values())

    def reward_components(self, simulation, drone, transition):
        components = dict(time=-STEP_COST, visibility=0.0, drone_progress=0.0,
                          followers=0.0, evacuation=0.0, completion=0.0, exit=0.0,
                          failure=0.0, mission_failure=0.0, wall=0.0)
        if self.path_following:
            components["waypoint"] = 0.0
        group = self._assigned_group(simulation, drone)
        visible = {
            evacuee for evacuee in self._visible_evacuees(simulation, drone)
            if drone.drone_id is not None and evacuee.assigned_drone_id == drone.drone_id
        }
        target_count = min(len(group), drone_visible_evacuee_target)
        visibility_score = min(len(visible), target_count) / target_count if target_count else 0.0
        components["visibility"] = VISIBILITY_REWARD * visibility_score

        escorting = bool(transition.initial_group) and not self.path_only
        if transition.active:
            distance = self._field_distance(transition.distance_fields,
                (drone.floor_id, drone.grid_x, drone.grid_y))
            gain = self._distance_gain(transition.drone_distance, distance)
            if gain > 0 and escorting:
                gain *= visibility_score
            progress_reward = PATH_FOLLOWING_PROGRESS_REWARD if self.path_following else DRONE_PROGRESS_REWARD
            components["drone_progress"] = progress_reward * gain * (0.5 if escorting else 1.0)
        if (
            transition.active
            and self._at_assigned_exit(simulation.grid_world, drone, transition.route)
            and not simulation.grid_world.is_exit(
                transition.position[1], transition.position[2], transition.position[0]
            )
        ):
            components["exit"] = DRONE_EXIT_REWARD
        if (self.path_following and self.reached_route_subgoal(simulation, drone, transition)
                and not components["exit"]):
            components["waypoint"] = PATH_FOLLOWING_CLUSTER_REWARD
        if transition.active and self.mission_failed(simulation, drone):
            components["mission_failure"] = -MISSION_FAILURE_PENALTY

        follower_progress = []
        for evacuee, previous_distance in transition.visible_followers.items():
            distance = self._field_distance(transition.distance_fields,
                (evacuee.floor_id, evacuee.grid_x, evacuee.grid_y))
            follower_progress.append(self._distance_gain(previous_distance, distance))
        if follower_progress:
            components["followers"] = FOLLOWER_PROGRESS_REWARD * sum(follower_progress) / len(follower_progress)

        if transition.initial_group:
            evacuated = self._evacuated_group(simulation, transition.initial_group)
            components["evacuation"] = FOLLOWER_EXIT_REWARD * len(evacuated - transition.evacuated_before) / len(transition.initial_group)
            if len(evacuated) == len(transition.initial_group) and len(transition.evacuated_before) < len(evacuated):
                components["completion"] = GROUP_COMPLETION_REWARD

        if self._failure_count(simulation, drone, transition) > FAILED_MOVE_GRACE:
            components["failure"] = -FAILED_MOVE_PENALTY

        if drone in simulation.drones and not simulation.touching_exit(drone):
            components["wall"] = -self._wall_proximity_penalty(simulation, drone.floor_id, drone.grid_x, drone.grid_y)

        return components

    @staticmethod
    def _evacuated_group(simulation, initial_group):
        inside = set(simulation.evacuees)
        return frozenset(person for person in initial_group if person not in inside and simulation.touching_exit(person))

    def group_finished(self, simulation, drone):
        if self.path_only:
            return self._at_assigned_exit(simulation.grid_world, drone, self.assigned_routes.get(drone, ()))
        initial_group = self.initial_groups.get(drone, ())
        if not initial_group:
            return simulation.touching_exit(drone)
        return len(self._evacuated_group(simulation, initial_group)) == len(initial_group)

    def mission_finished(self, simulation, drone):
        return (
            self._at_assigned_exit(simulation.grid_world, drone, self.assigned_routes.get(drone, ()))
            and self.group_finished(simulation, drone)
        )

    def mission_failed(self, simulation, drone):
        return drone not in simulation.drones and not self._at_assigned_exit(
            simulation.grid_world, drone, self.assigned_routes.get(drone, ())
        )

    @staticmethod
    def _at_assigned_exit(world, drone, route):
        if not route or not world.is_exit(drone.grid_x, drone.grid_y, drone.floor_id):
            return False
        return route[-1] == (drone.floor_id, world.floors[drone.floor_id].cluster_grid.get(drone.grid_x, drone.grid_y))

    @staticmethod
    def _failure_count(simulation, drone, transition):
        position = (drone.floor_id, drone.grid_x, drone.grid_y)
        if not transition.active or transition.action == int(Action.STAY) or position != transition.position:
            return 0
        previous = transition.previous_failure
        signature = (transition.position, transition.action)
        return (
            previous[2] + 1 if previous is not None and previous[0] == signature
            and previous[1] == simulation.step_count - 1 else 1
        )

    def save(self, path):
        torch.save({
            "observation_version": OBSERVATION_VERSION,
            "reward_version": self.reward_version,
            "learning_objective": self.learning_objective,
            "n_step": self.n_step,
            "discount": self.discount,
            "observation_shape": (CHANNELS, PATCH_SIZE, PATCH_SIZE),
            "state_dict": self.policy_network.state_dict(),
        }, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            return False

        if (
            checkpoint.get("observation_version") != OBSERVATION_VERSION
            or tuple(checkpoint.get("observation_shape", ())) != (CHANNELS, PATCH_SIZE, PATCH_SIZE)
        ):
            logger.warning("Drone model at %s uses incompatible observations; retrain with train_drones.py", path)
            return False
        state_dict = checkpoint["state_dict"]
        self.policy_network.load_state_dict(state_dict)
        if self.target_network is not None:
            self.target_network.load_state_dict(state_dict)
        return True

    @staticmethod
    def _distance_gain(previous, current):
        if np.isfinite(previous) and np.isfinite(current):
            return previous - current
        if np.isfinite(previous):
            return -1.0
        if np.isfinite(current):
            return 1.0
        return 0.0

    def _target_distance(self, world, target_node, floor_id, x, y):
        return self._field_distance(self._reward_distances(world, target_node), (floor_id, x, y))

    @staticmethod
    def _field_distance(fields, position):
        floor_id, x, y = position
        if not fields or floor_id not in fields:
            return float("inf")
        distances = fields[floor_id]
        if not (0 <= y < distances.shape[0] and 0 <= x < distances.shape[1]):
            return float("inf")
        return float(distances[y, x])

    def _reward_distances(self, world, target_node):
        if target_node is None:
            return None
        topology = tuple((floor_id, floor.cluster_grid, floor.distance_map)
                         for floor_id, floor in world.floors.items())
        same_topology = len(topology) == len(self.reward_topology) and all(
            first[0] == second[0] and first[1] is second[1] and first[2] is second[2]
            for first, second in zip(topology, self.reward_topology)
        )
        if self.reward_world is not world or not same_topology:
            self.reward_distance_fields.clear()
            self.reward_world, self.reward_topology = world, topology
        if target_node in self.reward_distance_fields:
            return self.reward_distance_fields[target_node]
        distances = {floor_id: np.full((floor.map_grid.height, floor.map_grid.width), np.inf, dtype=np.float32)
                     for floor_id, floor in world.floors.items()}
        pending = deque()
        floor_id, cluster_id = target_node
        for x, y in world.floors[floor_id].cluster_grid.clusters[cluster_id].cells:
            if world.can_move_to(x, y, floor_id):
                distances[floor_id][y, x] = 0
                pending.append((floor_id, x, y))
        expanded_stairs = set()
        while pending:
            floor_id, x, y = pending.popleft()
            next_distance = distances[floor_id][y, x] + 1
            for dx, dy in ACTION_DIRECTIONS.values():
                nx, ny = x + dx, y + dy
                if (dx or dy) and world.can_move_to(nx, ny, floor_id) and distances[floor_id][ny, nx] > next_distance:
                    distances[floor_id][ny, nx] = next_distance
                    pending.append((floor_id, nx, ny))
            node = world.stair_cluster_at(floor_id, x, y)
            if node is not None and node not in expanded_stairs:
                expanded_stairs.add(node)
                cluster = world._get_stair_cluster(node)
                if cluster.linked_to is not None:
                    target_floor, _ = cluster.linked_to
                    for nx, ny in world._get_stair_cluster(cluster.linked_to).cells:
                        if world.can_move_to(nx, ny, target_floor) and distances[target_floor][ny, nx] > next_distance:
                            distances[target_floor][ny, nx] = next_distance
                            pending.append((target_floor, nx, ny))
        if len(self.reward_distance_fields) >= 8:
            self.reward_distance_fields.pop(next(iter(self.reward_distance_fields)))
        self.reward_distance_fields[target_node] = distances
        return distances

    def _sample_replay(self, batch_size=64):
        successful_count = min(batch_size // 4, len(self.successful_replay))
        batch = random.sample(self.successful_replay, successful_count)
        batch.extend(random.sample(self.replay_buffer, batch_size - successful_count))
        random.shuffle(batch)
        return batch

    def _wall_proximity_penalty(self, simulation, floor_id, x, y):
        floor = simulation.grid_world.floors[floor_id]
        grid = floor.map_grid
        closest_wall = WALL_CLEARANCE_RADIUS + 1

        for delta_y in range(-WALL_CLEARANCE_RADIUS, WALL_CLEARANCE_RADIUS + 1):
            max_delta_x = WALL_CLEARANCE_RADIUS - abs(delta_y)
            for delta_x in range(-max_delta_x, max_delta_x + 1):
                distance = abs(delta_x) + abs(delta_y)
                neighbor_x = x + delta_x
                neighbor_y = y + delta_y

                if not grid.in_bounds(neighbor_x, neighbor_y):
                    closest_wall = min(closest_wall, distance)
                elif (
                    grid.get(neighbor_x, neighbor_y) == CellType.WALL
                    or floor.temporary_wall_grid.get(neighbor_x, neighbor_y)
                ):
                    closest_wall = min(closest_wall, distance)

        if closest_wall > WALL_CLEARANCE_RADIUS:
            return 0.0

        return WALL_PROXIMITY_PENALTY * (WALL_CLEARANCE_RADIUS + 1 - closest_wall) / WALL_CLEARANCE_RADIUS

    def _observe(self, simulation, drone, route):
        world = simulation.grid_world
        floor = world.floors[drone.floor_id]
        cluster_grid = floor.cluster_grid
        route = list(route)
        if self.assigned_routes.get(drone) != tuple(route):
            self._flush_returns(drone, terminal=self.path_following)
            if self.path_following:
                self.episode_replay.pop(drone, None)
            self.route_cursors.pop(drone, None)
        self.assigned_routes[drone] = tuple(route)
        route_cursor = self.route_cursors.get(drone, 0)
        current_cluster = cluster_grid.get(drone.grid_x, drone.grid_y)
        current_node = (drone.floor_id, current_cluster)

        if current_node in route:
            matching_indices = [
                index for index, node in enumerate(route)
                if node == current_node and index >= route_cursor
            ]
            if matching_indices:
                current_index = matching_indices[-1]
                route_cursor = min(current_index + 1, len(route))
                self.route_cursors[drone] = route_cursor

        remaining_route = route[route_cursor:] if route else []
        route_ids = set(route[max(0, route_cursor - 1):]) if route else set()
        target_node = remaining_route[0] if remaining_route else (route[-1] if route else None)
        self.route_targets[drone] = target_node

        target_cells = []
        target_floor_id = drone.floor_id
        if target_node is not None:
            target_floor_id, target_cluster_id = target_node
            target_cells = world.floors[target_floor_id].cluster_grid.clusters[
                target_cluster_id
            ].cells

        state = np.zeros((CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        visible = self._visible_evacuees(simulation, drone)
        visible_cells = {(evacuee.grid_x, evacuee.grid_y): evacuee for evacuee in visible}
        visible_assigned_count = sum(
            drone.drone_id is not None and evacuee.assigned_drone_id == drone.drone_id
            for evacuee in visible
        )
        state[VISIBLE_GROUP_COUNT_CHANNEL].fill(
            min(visible_assigned_count, drone_visible_evacuee_target) / drone_visible_evacuee_target
        )
        if target_cells and target_floor_id == drone.floor_id:
            target_x = sum(x for x, _ in target_cells) / len(target_cells)
            target_y = sum(y for _, y in target_cells) / len(target_cells)
            normalization = max(floor.map_grid.width, floor.map_grid.height, 1)
            direction_x = (target_x - drone.grid_x) / normalization
            direction_y = (target_y - drone.grid_y) / normalization
        else:
            direction_x = direction_y = 0.0

        floor_channel = drone.floor_id / max(len(world.floors) - 1, 1)
        for patch_y in range(PATCH_SIZE):
            for patch_x in range(PATCH_SIZE):
                x = drone.grid_x + patch_x - PATCH_RADIUS
                y = drone.grid_y + patch_y - PATCH_RADIUS
                if not floor.map_grid.in_bounds(x, y):
                    state[0, patch_y, patch_x] = 1.0
                    state[9, patch_y, patch_x] = floor_channel
                    continue

                cell_type = floor.simulation_grid.get(x, y)
                terrain_type = floor.map_grid.get(x, y)
                if cell_type == CellType.WALL or floor.temporary_wall_grid.get(x, y):
                    state[0, patch_y, patch_x] = 1.0
                evacuee = visible_cells.get((x, y))
                if evacuee is not None:
                    channel = (
                        ASSIGNED_EVACUEES_CHANNEL
                        if drone.drone_id is not None and evacuee.assigned_drone_id == drone.drone_id
                        else 1
                    )
                    state[channel, patch_y, patch_x] = 1.0
                if (
                    cell_type == CellType.DRONE
                    and (x, y) != (drone.grid_x, drone.grid_y)
                ):
                    state[2, patch_y, patch_x] = 1.0

                cluster_id = cluster_grid.get(x, y)
                cluster_node = (drone.floor_id, cluster_id)
                if cluster_id is not None and cluster_id >= 0 and cluster_node in route_ids:
                    state[3, patch_y, patch_x] = 1.0
                if (
                    target_node is not None
                    and target_node[0] == drone.floor_id
                    and cluster_id == target_node[1]
                ):
                    state[4, patch_y, patch_x] = 1.0
                if terrain_type == CellType.STAIRS:
                    state[5, patch_y, patch_x] = 1.0
                if terrain_type == CellType.EXIT:
                    state[6, patch_y, patch_x] = 1.0

                state[7, patch_y, patch_x] = direction_x
                state[8, patch_y, patch_x] = direction_y
                state[9, patch_y, patch_x] = floor_channel

        return state, route_ids

    def _legal_actions(self, simulation, drone):
        if drone not in simulation.drones or simulation.touching_exit(drone):
            return [Action.STAY]
        world = simulation.grid_world
        legal_actions = [Action.STAY]

        for action, (dx, dy) in ACTION_DIRECTIONS.items():
            if action == Action.STAY:
                continue

            x = drone.grid_x + dx
            y = drone.grid_y + dy
            if not world.can_move_to(x, y, drone.floor_id):
                continue
            legal_actions.append(action)

        if world.stair_arrival(drone, "drone") is not None:
            legal_actions.append(Action.USE_STAIRS)

        return legal_actions

    def _select_action(self, state, legal_actions, drone, route_ids, simulation):
        self.decision_count += 1
        epsilon = self.epsilon if self.training else 0.0
        if random.random() < epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(tensor)[0].cpu().numpy()
        legal_values = q_values[[int(action) for action in legal_actions]]
        if not np.isfinite(legal_values).all() or (
            self.max_training_q is not None and np.max(np.abs(legal_values)) > self.max_training_q
        ):
            raise RuntimeError("Invalid or divergent DQN action values; start from fresh weights")

        scores = np.full(len(Action), -np.inf, dtype=np.float32)
        for action in legal_actions:
            scores[int(action)] = q_values[int(action)]

        return Action(int(np.argmax(scores)))

    def _optimize(self):
        if self.target_network is None or self.optimizer is None:
            raise RuntimeError("DQN optimization requires a training controller")

        batch = self._sample_replay()
        states, actions, rewards, next_states, dones, next_legal_masks, discounts = zip(*batch)
        states = torch.from_numpy(np.stack(states)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_legal_masks = torch.from_numpy(np.stack(next_legal_masks)).to(self.device)
        discounts = torch.tensor(discounts, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            targets = rewards.clone()
            if torch.any(discounts > 0):
                next_actions = self.policy_network(next_states).masked_fill(
                    next_legal_masks == 0, -torch.inf
                ).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                targets += discounts * next_q_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(current_q_values, targets)
        self.last_loss = float(loss.detach())
        self.last_q_abs_max = float(current_q_values.detach().abs().max())
        self.last_target_abs_max = float(targets.abs().max())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite DQN loss; stopping training")
        if self.max_training_q is not None and max(self.last_q_abs_max, self.last_target_abs_max) > self.max_training_q:
            raise RuntimeError(f"DQN values exceeded {self.max_training_q:g}; stopping divergent training")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        if self.training_steps % self.target_sync_steps == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())