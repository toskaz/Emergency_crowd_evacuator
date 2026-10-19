from collections import deque
import random

import numpy as np
import torch
from torch import nn

from controls.action import Action
from map.cell import CellType


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

PATCH_RADIUS = 4
PATCH_SIZE = 2 * PATCH_RADIUS + 1
CHANNELS = 8
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
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy_network = DroneQNetwork().to(self.device)
        self.target_network = DroneQNetwork().to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.0005)
        self.replay_buffer = deque(maxlen=20000)
        self.pending_transitions = {}
        self.route_cursors = {}
        self.distance_map = []
        self.distance_map_version = None
        self.decision_count = 0
        self.training_steps = 0

    def get_actions(self, simulation, routes):
        self._prepare_distance_map(simulation)
        actions = []

        for drone in simulation.drones:
            if simulation.map_grid.get(drone.grid_x, drone.grid_y) == CellType.EXIT:
                self.pending_transitions.pop(drone, None)
                actions.append(Action.STAY)
                continue

            route = routes.get(drone, [])
            state, route_ids = self._observe(simulation, drone, route)
            legal_actions = self._legal_actions(simulation, drone)
            action = self._select_action(state, legal_actions, drone, route_ids, simulation)
            distance = self._distance_at(drone.grid_x, drone.grid_y)
            self.pending_transitions[drone] = (
                state,
                int(action),
                distance,
                route_ids,
                (drone.grid_x, drone.grid_y),
            )
            actions.append(action)

        return actions

    def learn(self, simulation, routes):
        for drone, transition in list(self.pending_transitions.items()):
            state, action, previous_distance, route_ids, previous_position = transition
            current_position = (drone.grid_x, drone.grid_y)
            current_distance = self._distance_at(*current_position)
            reward = -0.04

            if previous_distance < float("inf") and current_distance < float("inf"):
                reward += 0.35 * (previous_distance - current_distance)

            if current_position == previous_position and action != Action.STAY:
                reward -= 0.5

            reward -= self._wall_proximity_penalty(simulation, *current_position)

            cluster_id = simulation.cluster_grid.get(*current_position)
            if cluster_id in route_ids:
                reward += 0.06
            else:
                reward -= 0.05

            done = simulation.map_grid.get(*current_position) == CellType.EXIT
            if done:
                reward += 2.0

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

    def _prepare_distance_map(self, simulation):
        if self.distance_map_version == simulation.map_version:
            return

        width = simulation.map_grid.width
        height = simulation.map_grid.height
        unreachable = width * height + 1
        self.distance_map = [[unreachable for _ in range(width)] for _ in range(height)]
        pending = deque()

        for exit_ in simulation.exits:
            for x, y in exit_.cells:
                if self.distance_map[y][x] != 0:
                    self.distance_map[y][x] = 0
                    pending.append((x, y))

        while pending:
            x, y = pending.popleft()
            for dx, dy in ACTION_DIRECTIONS.values():
                if dx == 0 and dy == 0:
                    continue
                next_x, next_y = x + dx, y + dy
                if not simulation.map_grid.in_bounds(next_x, next_y):
                    continue
                if simulation.map_grid.get(next_x, next_y) == CellType.WALL:
                    continue
                if self.distance_map[next_y][next_x] <= self.distance_map[y][x] + 1:
                    continue
                self.distance_map[next_y][next_x] = self.distance_map[y][x] + 1
                pending.append((next_x, next_y))

        self.distance_map_version = simulation.map_version

    def _distance_at(self, x, y):
        if not self.distance_map or not (0 <= y < len(self.distance_map)):
            return float("inf")
        if not (0 <= x < len(self.distance_map[y])):
            return float("inf")
        return self.distance_map[y][x]

    def _wall_proximity_penalty(self, simulation, x, y):
        closest_wall = WALL_CLEARANCE_RADIUS + 1

        for delta_y in range(-WALL_CLEARANCE_RADIUS, WALL_CLEARANCE_RADIUS + 1):
            max_delta_x = WALL_CLEARANCE_RADIUS - abs(delta_y)
            for delta_x in range(-max_delta_x, max_delta_x + 1):
                distance = abs(delta_x) + abs(delta_y)
                neighbor_x = x + delta_x
                neighbor_y = y + delta_y

                if not simulation.map_grid.in_bounds(neighbor_x, neighbor_y):
                    closest_wall = min(closest_wall, distance)
                elif simulation.map_grid.get(neighbor_x, neighbor_y) == CellType.WALL:
                    closest_wall = min(closest_wall, distance)

        if closest_wall > WALL_CLEARANCE_RADIUS:
            return 0.0

        return WALL_PROXIMITY_PENALTY * (WALL_CLEARANCE_RADIUS + 1 - closest_wall)

    def _observe(self, simulation, drone, route):
        cluster_grid = simulation.cluster_grid
        route = list(route)
        route_cursor = self.route_cursors.get(drone, 0)
        current_cluster = cluster_grid.get(drone.grid_x, drone.grid_y)

        if current_cluster in route:
            matching_indices = [
                index for index, cluster_id in enumerate(route)
                if cluster_id == current_cluster and index >= route_cursor
            ]
            if matching_indices:
                current_index = matching_indices[-1]
                route_cursor = min(current_index + 1, len(route) - 1)
                self.route_cursors[drone] = route_cursor

        remaining_route = route[route_cursor:] if route else []
        route_ids = set(route[max(0, route_cursor - 1):]) if route else set()
        target_cluster = remaining_route[0] if remaining_route else None

        if target_cluster in cluster_grid.exit_clusters:
            target_cells = [
                cell for exit_ in simulation.exits for cell in exit_.cells
                if cluster_grid.get(*cell) == target_cluster
            ]
        elif target_cluster is not None:
            target_cells = cluster_grid.clusters.get(target_cluster, [])
        else:
            target_cells = [cell for exit_ in simulation.exits for cell in exit_.cells]

        if not target_cells:
            target_cells = [cell for exit_ in simulation.exits for cell in exit_.cells]

        state = np.zeros((CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        if target_cells:
            target_x = sum(x for x, _ in target_cells) / len(target_cells)
            target_y = sum(y for _, y in target_cells) / len(target_cells)
            normalization = max(simulation.map_grid.width, simulation.map_grid.height, 1)
            direction_x = (target_x - drone.grid_x) / normalization
            direction_y = (target_y - drone.grid_y) / normalization
        else:
            direction_x = direction_y = 0.0

        for patch_y in range(PATCH_SIZE):
            for patch_x in range(PATCH_SIZE):
                x = drone.grid_x + patch_x - PATCH_RADIUS
                y = drone.grid_y + patch_y - PATCH_RADIUS
                if not simulation.map_grid.in_bounds(x, y):
                    state[0, patch_y, patch_x] = 1.0
                    continue

                if simulation.map_grid.get(x, y) == CellType.WALL:
                    state[0, patch_y, patch_x] = 1.0
                if simulation.occupancy_grid.get(x, y) is not None:
                    state[1, patch_y, patch_x] = 1.0
                if simulation.drone_occupancy_grid.get(x, y) not in (None, drone):
                    state[2, patch_y, patch_x] = 1.0

                cluster_id = cluster_grid.get(x, y)
                if cluster_id in route_ids:
                    state[3, patch_y, patch_x] = 1.0
                if target_cluster is not None and cluster_id == target_cluster:
                    state[4, patch_y, patch_x] = 1.0
                if simulation.map_grid.get(x, y) == CellType.EXIT:
                    state[5, patch_y, patch_x] = 1.0

                state[6, patch_y, patch_x] = direction_x
                state[7, patch_y, patch_x] = direction_y

        return state, route_ids

    def _legal_actions(self, simulation, drone):
        legal_actions = [Action.STAY]
        for action, (dx, dy) in ACTION_DIRECTIONS.items():
            if action == Action.STAY:
                continue
            x = drone.grid_x + dx
            y = drone.grid_y + dy
            if not simulation.can_move_to(x, y):
                continue
            if simulation.occupancy_grid.get(x, y) is not None:
                continue
            if simulation.drone_occupancy_grid.get(x, y) not in (None, drone):
                continue
            legal_actions.append(action)
        return legal_actions

    def _select_action(self, state, legal_actions, drone, route_ids, simulation):
        self.decision_count += 1
        epsilon = max(0.03, 0.15 * (0.9995 ** self.decision_count))
        if random.random() < epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(tensor)[0].cpu().numpy()

        current_distance = self._distance_at(drone.grid_x, drone.grid_y)
        scores = np.full(len(Action), -np.inf, dtype=np.float32)
        for action in legal_actions:
            dx, dy = ACTION_DIRECTIONS[action]
            x, y = drone.grid_x + dx, drone.grid_y + dy
            distance_gain = current_distance - self._distance_at(x, y)
            cluster_id = simulation.cluster_grid.get(x, y)
            route_bonus = 0.15 if cluster_id in route_ids else 0.0
            wall_penalty = self._wall_proximity_penalty(simulation, x, y)
            scores[int(action)] = (
                q_values[int(action)]
                + 0.35 * distance_gain
                + route_bonus
                - wall_penalty
            )

        return Action(int(np.argmax(scores)))

    def _optimize(self):
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