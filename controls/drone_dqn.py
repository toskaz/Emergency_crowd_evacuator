from collections import deque
import random

import numpy as np
import torch
from torch import nn

from controls.action import Action
from map.cell import CellType
from grids.cluster import ClusterType


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

PATCH_RADIUS = 4
PATCH_SIZE = 2 * PATCH_RADIUS + 1
CHANNELS = 10
WALL_CLEARANCE_RADIUS = 3
WALL_PROXIMITY_PENALTY = 0.08


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
    def __init__(self, training=True):
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
                self.policy_network.parameters(), lr=0.0005
            )
        else:
            self.policy_network.eval()
        self.replay_buffer = deque(maxlen=20000)
        self.pending_transitions = {}
        self.route_cursors = {}
        self.decision_count = 0
        self.training_steps = 0

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
            distance = self._distance_at(
                world, drone.floor_id, drone.grid_x, drone.grid_y
            )
            if self.training:
                self.pending_transitions[drone] = (
                    state,
                    int(action),
                    distance,
                    route_ids,
                    (drone.floor_id, drone.grid_x, drone.grid_y),
                )
            actions.append(action)

        return actions

    def learn(self, simulation, routes):
        if not self.training:
            return

        for drone, transition in list(self.pending_transitions.items()):
            state, action, previous_distance, route_ids, previous_position = transition
            reward = self._transition_reward(
                simulation,
                drone,
                action,
                previous_distance,
                route_ids,
                previous_position,
            )
            done = simulation.grid_world.is_exit(
                drone.grid_x, drone.grid_y, drone.floor_id
            )

            next_state, _ = self._observe(simulation, drone, routes.get(drone, []))
            next_legal = self._legal_actions(simulation, drone)
            legal_mask = np.zeros(len(Action), dtype=np.float32)
            legal_mask[[int(candidate) for candidate in next_legal]] = 1.0
            self.replay_buffer.append((state, action, reward, next_state, done, legal_mask))

        self.pending_transitions.clear()

        if len(self.replay_buffer) >= 64:
            self.training_steps += 1
            if self.training_steps % 4 == 0:
                self._optimize()

    def _transition_reward(
        self,
        simulation,
        drone,
        action,
        previous_distance,
        route_ids,
        previous_position,
    ):
        current_position = (drone.floor_id, drone.grid_x, drone.grid_y)
        current_distance = self._distance_at(
            simulation.grid_world, *current_position
        )
        reward = -0.04

        if previous_distance < float("inf") and current_distance < float("inf"):
            reward += 0.35 * (previous_distance - current_distance)

        if current_position == previous_position and action != Action.STAY:
            reward -= 0.5

        reward -= self._wall_proximity_penalty(
            simulation, drone.floor_id, drone.grid_x, drone.grid_y
        )
        cluster_id = simulation.grid_world.floors[drone.floor_id].cluster_grid.get(
            drone.grid_x, drone.grid_y
        )
        if (drone.floor_id, cluster_id) in route_ids:
            reward += 0.06
        else:
            reward -= 0.05

        if simulation.grid_world.is_exit(
            drone.grid_x, drone.grid_y, drone.floor_id
        ):
            reward += 2.0

        return reward

    def save(self, path):
        torch.save(self.policy_network.state_dict(), path)

    def load(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            return False

        self.policy_network.load_state_dict(state_dict)
        if self.target_network is not None:
            self.target_network.load_state_dict(state_dict)
        return True

    def _distance_at(self, world, floor_id, x, y):
        distances = world.floors[floor_id].distance_map
        if not (0 <= y < len(distances)) or not (0 <= x < len(distances[y])):
            return float("inf")
        return distances[y][x]

    def _wall_proximity_penalty(self, simulation, floor_id, x, y):
        grid = simulation.grid_world.floors[floor_id].map_grid
        closest_wall = WALL_CLEARANCE_RADIUS + 1

        for delta_y in range(-WALL_CLEARANCE_RADIUS, WALL_CLEARANCE_RADIUS + 1):
            max_delta_x = WALL_CLEARANCE_RADIUS - abs(delta_y)
            for delta_x in range(-max_delta_x, max_delta_x + 1):
                distance = abs(delta_x) + abs(delta_y)
                neighbor_x = x + delta_x
                neighbor_y = y + delta_y

                if not grid.in_bounds(neighbor_x, neighbor_y):
                    closest_wall = min(closest_wall, distance)
                elif grid.get(neighbor_x, neighbor_y) == CellType.WALL:
                    closest_wall = min(closest_wall, distance)

        if closest_wall > WALL_CLEARANCE_RADIUS:
            return 0.0

        return WALL_PROXIMITY_PENALTY * (WALL_CLEARANCE_RADIUS + 1 - closest_wall)

    def _observe(self, simulation, drone, route):
        world = simulation.grid_world
        floor = world.floors[drone.floor_id]
        cluster_grid = floor.cluster_grid
        route = list(route)
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
        target_node = remaining_route[0] if remaining_route else None

        if target_node is not None:
            target_floor_id, target_cluster_id = target_node
            target_cells = world.floors[target_floor_id].cluster_grid.clusters[
                target_cluster_id
            ].cells
        else:
            target_cells = [
                cell
                for cluster in cluster_grid.clusters
                if cluster.type is ClusterType.EXIT
                for cell in cluster.cells
            ]
            target_floor_id = drone.floor_id

        if not target_cells:
            target_cells = [
                cell
                for cluster in cluster_grid.clusters
                if cluster.type is ClusterType.EXIT
                for cell in cluster.cells
            ]

        state = np.zeros((CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
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
                if cell_type == CellType.WALL:
                    state[0, patch_y, patch_x] = 1.0
                if cell_type == CellType.EVACUEE:
                    state[1, patch_y, patch_x] = 1.0
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

        world = simulation.grid_world
        current_distance = self._distance_at(
            world, drone.floor_id, drone.grid_x, drone.grid_y
        )
        scores = np.full(len(Action), -np.inf, dtype=np.float32)
        for action in legal_actions:
            if action is Action.USE_STAIRS:
                next_floor_id, x, y = world.stair_arrival(drone, "drone")
            else:
                dx, dy = ACTION_DIRECTIONS[action]
                next_floor_id = drone.floor_id
                x = drone.grid_x + dx
                y = drone.grid_y + dy

            distance_gain = current_distance - self._distance_at(
                world, next_floor_id, x, y
            )
            cluster_id = world.floors[next_floor_id].cluster_grid.get(x, y)
            route_bonus = (
                0.15 if (next_floor_id, cluster_id) in route_ids else 0.0
            )
            wall_penalty = self._wall_proximity_penalty(
                simulation, next_floor_id, x, y
            )
            scores[int(action)] = (
                q_values[int(action)]
                + 0.35 * distance_gain
                + route_bonus
                - wall_penalty
            )

        return Action(int(np.argmax(scores)))

    def _optimize(self):
        if self.target_network is None or self.optimizer is None:
            raise RuntimeError("DQN optimization requires a training controller")

        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones, next_legal_masks = zip(*batch)
        states = torch.from_numpy(np.stack(states)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_legal_masks = torch.from_numpy(np.stack(next_legal_masks)).to(self.device)

        current_q_values = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).masked_fill(
                next_legal_masks == 0, -torch.inf
            ).max(dim=1).values
            targets = rewards + 0.95 * next_q_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        if self.training_steps % 100 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())