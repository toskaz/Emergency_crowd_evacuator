from collections import deque
from dataclasses import dataclass
import logging
import random
import numpy as np
import torch
from torch import nn
from controls.action import Action
from controls.drone_cnn import ENCODED_SIZE, HISTORY_FRAMES, VIEW_RADIUS, VIEW_SIZE, map_encoder
from map.cell import CellType
logger = logging.getLogger(__name__)
ACTION_DIRECTIONS = {Action.STAY: (0, 0), Action.UP: (0, -1), Action.DOWN: (0, 1), Action.LEFT: (-1, 0), Action.RIGHT: (1, 0)}
PATCH_RADIUS = VIEW_RADIUS
PATCH_SIZE = VIEW_SIZE
CHANNELS = 2
STACKED_CHANNELS = CHANNELS * HISTORY_FRAMES
OBSERVATION_VERSION = 8
NAVIGATION_REWARD_VERSION = 18
TARGET_SYNC_STEPS = 50
SOFT_UPDATE_TAU = 0.005
BATCH_SIZE = 32
MAX_TRAINING_Q = 100.0
DISCOUNT = 0.99
N_STEP = 1

@dataclass
class DroneTransition:
    state: dict
    action: int
    position: tuple
    target_node: tuple | None
    route: tuple
    active: bool

class DroneQNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.map_encoder = map_encoder(STACKED_CHANNELS)
        self.features = nn.Sequential(nn.Linear(ENCODED_SIZE + 2, 128), nn.ReLU(), nn.Linear(128, len(Action)))

    def forward(self, state):
        patch, offsets = state
        return self.features(torch.cat((self.map_encoder(patch), offsets), dim=1))

class DQNDroneController:
    network_type = DroneQNetwork
    observation_version = OBSERVATION_VERSION
    observation_shape = (STACKED_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    feature_count = 2

    def __init__(self, training=True, n_step=N_STEP):
        if not isinstance(n_step, int) or n_step < 1:
            raise ValueError('n_step must be a positive integer')
        self.device = torch.device('cpu')
        self.training = training
        self.epsilon = 1.0
        self.policy_network = self.network_type().to(self.device)
        self.target_network = None
        self.optimizer = None
        if training:
            self.target_network = self.network_type().to(self.device)
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.target_network.eval()
            self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.0001)
        else:
            self.policy_network.eval()
        self.replay_buffer = deque(maxlen=10000)
        self.n_step = n_step
        self.discount = DISCOUNT
        self.return_queues = {}
        self.pending_transitions = {}
        self.route_cursors = {}
        self.route_targets = {}
        self.assigned_routes = {}
        self.decision_count = 0
        self.training_steps = 0
        self.environment_steps = 0
        self.frame_history = {}
        self.last_loss = None
        self.last_q_abs_max = None
        self.last_target_abs_max = None
        self.max_training_q = MAX_TRAINING_Q if training else None
        self.target_sync_steps = TARGET_SYNC_STEPS
        self.path_only = True
        self.reward_distance_fields = {}
        self.reward_world = None
        self.reward_topology = ()
        self.navigation_best = {}
        self.navigation_paid = set()
        self.navigation_exits_paid = set()

    def reset(self):
        self.finish_episode()
        self.frame_history.clear()
        self.pending_transitions.clear()
        self.navigation_best.clear()
        self.navigation_paid.clear()
        self.navigation_exits_paid.clear()
        self.route_cursors.clear()
        self.route_targets.clear()
        self.assigned_routes.clear()
        self.reward_distance_fields.clear()
        self.reward_world = None
        self.reward_topology = ()

    def invalidate(self):
        """Discard cached distance fields after environment edits."""
        self.reward_distance_fields.clear()
        self.reward_world = None
        self.reward_topology = ()

    def get_actions(self, simulation, routes):
        actions = []
        for drone in simulation.drones:
            if getattr(drone, 'in_stair_transit', False):
                actions.append(Action.STAY)
                continue
            world = simulation.grid_world
            floor = world.floors[drone.floor_id]
            if floor.map_grid.get(drone.grid_x, drone.grid_y) == CellType.EXIT:
                self.pending_transitions.pop(drone, None)
                actions.append(Action.STAY)
                continue
            route = routes.get(drone, [])
            state, route_ids = self._observe(simulation, drone, route)
            legal_actions = self._legal_actions(simulation, drone)
            if not legal_actions:
                actions.append(Action.STAY)
                continue
            action = self._select_action(state, legal_actions, drone, route_ids, simulation)
            if self.training:
                self.record_transition(simulation, drone, state, action, route_ids)
            actions.append(action)
        return actions

    def _assigned_group(self, simulation, drone):
        return [evacuee for evacuee in simulation.evacuees if drone.drone_id is not None and evacuee.assigned_drone_id == drone.drone_id]

    def record_transition(self, simulation, drone, state, action, route_ids):
        transition = DroneTransition(state=state, action=int(action), position=(drone.floor_id, drone.grid_x, drone.grid_y), target_node=self.route_targets.get(drone), route=self.assigned_routes.get(drone, ()), active=drone in simulation.drones)
        self._prepare_navigation_transition(simulation, drone, transition)
        self.pending_transitions[drone] = transition
        return transition

    def learn(self, simulation, routes, *, outcomes=None):
        if not self.training:
            return
        for drone, transition in list(self.pending_transitions.items()):
            if outcomes is None:
                reward = self._transition_reward(simulation, drone, transition)
                success = self.mission_finished(simulation, drone)
                mission_done = success or self.mission_failed(simulation, drone)
            else:
                reward, success, mission_done = outcomes[drone]
            self._commit_transition(simulation, drone, transition)
            next_state, _ = self._observe(simulation, drone, routes.get(drone, []))
            done = mission_done
            legal_mask = self.action_mask(simulation, drone)
            experience = (transition.state, transition.action, reward, next_state, done, legal_mask)
            queue = self.return_queues.setdefault(drone, deque())
            queue.append(experience)
            if len(queue) >= self.n_step:
                self._emit_return(drone)
            if done:
                self._flush_returns(drone)
        self.pending_transitions.clear()
        self.environment_steps += 1
        if len(self.replay_buffer) >= BATCH_SIZE:
            self.training_steps += 1
            self._optimize()
        if self.environment_steps % self.target_sync_steps == 0:
            self._update_target_network()

    def _update_target_network(self):
        with torch.no_grad():
            for target, online in zip(self.target_network.parameters(), self.policy_network.parameters()):
                target.lerp_(online, SOFT_UPDATE_TAU)

    def _emit_return(self, drone):
        queue = self.return_queues[drone]
        window = list(queue)[:self.n_step]
        first, last = (window[0], window[-1])
        reward = sum((self.discount ** index * item[2] for index, item in enumerate(window)))
        experience = (first[0], first[1], reward, last[3], last[4], last[5], self.discount ** len(window))
        self.replay_buffer.append(experience)
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

    @property
    def learning_objective(self):
        return 'mission'

    @property
    def reward_version(self):
        return NAVIGATION_REWARD_VERSION

    def _commit_transition(self, simulation, drone, transition):
        self._commit_navigation_transition(simulation, drone, transition)

    def _transition_reward(self, simulation, drone, transition):
        return sum(self.reward_components(simulation, drone, transition).values())

    def reward_components(self, simulation, drone, transition, *, timed_out=False):
        return self._pure_navigation_rewards(simulation, drone, transition, timed_out=timed_out)

    def _prepare_navigation_transition(self, simulation, drone, transition):
        target = self.route_targets.get(drone)
        fields = self._reward_distances(simulation.grid_world, target)
        distance = self._field_distance(fields, transition.position)
        key = (drone, target)
        best = min(self.navigation_best.get(key, distance), distance)
        self.navigation_best[key] = best
        transition.navigation_target = target
        transition.navigation_fields = fields
        transition.navigation_best = best
        transition.navigation_paid = key in self.navigation_paid
        transition.navigation_exit_paid = drone in self.navigation_exits_paid

    def _pure_navigation_rewards(self, simulation, drone, transition, *, timed_out=False):
        position = (drone.floor_id, drone.grid_x, drone.grid_y)
        distance = self._field_distance(transition.navigation_fields, position)
        best = transition.navigation_best
        gain = max(0.0, best - distance) if np.isfinite(best) and np.isfinite(distance) else 0.0
        floor = simulation.grid_world.floors[drone.floor_id]
        node = (drone.floor_id, floor.cluster_grid.get(drone.grid_x, drone.grid_y))
        reached = node == transition.navigation_target
        success = self._at_assigned_exit(simulation.grid_world, drone, transition.route)
        return dict(time=-0.01, progress=0.05 * gain, failure=0.0, waypoint=1.0 if reached and (not success) and (not transition.navigation_paid) else 0.0, exit=10.0 if success and (not transition.navigation_exit_paid) else 0.0, mission_failure=-1.0 if timed_out or self.mission_failed(simulation, drone) else 0.0)

    def _commit_navigation_transition(self, simulation, drone, transition):
        position = (drone.floor_id, drone.grid_x, drone.grid_y)
        distance = self._field_distance(transition.navigation_fields, position)
        key = (drone, transition.navigation_target)
        self.navigation_best[key] = min(self.navigation_best[key], distance)
        floor = simulation.grid_world.floors[drone.floor_id]
        if (drone.floor_id, floor.cluster_grid.get(drone.grid_x, drone.grid_y)) == transition.navigation_target:
            self.navigation_paid.add(key)
        if self._at_assigned_exit(simulation.grid_world, drone, transition.route):
            self.navigation_exits_paid.add(drone)

    def group_finished(self, simulation, drone):
        return self.mission_finished(simulation, drone)

    def mission_finished(self, simulation, drone):
        return self._at_assigned_exit(simulation.grid_world, drone, self.assigned_routes.get(drone, ()))

    def mission_failed(self, simulation, drone):
        return drone not in simulation.drones and (not self._at_assigned_exit(simulation.grid_world, drone, self.assigned_routes.get(drone, ())))

    @staticmethod
    def _at_assigned_exit(world, drone, route):
        if not route or not world.is_exit(drone.grid_x, drone.grid_y, drone.floor_id):
            return False
        return route[-1] == (drone.floor_id, world.floors[drone.floor_id].cluster_grid.get(drone.grid_x, drone.grid_y))

    def save(self, path):
        torch.save({'frame_stack': HISTORY_FRAMES, 'batch_size': BATCH_SIZE, 'replay_capacity': self.replay_buffer.maxlen, 'target_update_steps': self.target_sync_steps, 'soft_update_tau': SOFT_UPDATE_TAU, 'observation_version': self.observation_version, 'reward_version': self.reward_version, 'learning_objective': self.learning_objective, 'n_step': self.n_step, 'discount': self.discount, 'observation_shape': self.observation_shape, 'feature_count': self.feature_count, 'state_dict': self.policy_network.state_dict()}, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            return False
        if checkpoint.get('observation_version') != self.observation_version or tuple(checkpoint.get('observation_shape', ())) != self.observation_shape or checkpoint.get('feature_count', 0) != self.feature_count:
            logger.warning('Drone model at %s uses incompatible observations; retrain with train_drones.py', path)
            return False
        state_dict = checkpoint['state_dict']
        self.policy_network.load_state_dict(state_dict)
        if self.target_network is not None:
            self.target_network.load_state_dict(state_dict)
        return True

    @staticmethod
    def _field_distance(fields, position):
        floor_id, x, y = position
        if not fields or floor_id not in fields:
            return float('inf')
        distances = fields[floor_id]
        if not (0 <= y < distances.shape[0] and 0 <= x < distances.shape[1]):
            return float('inf')
        return float(distances[y, x])

    def _reward_distances(self, world, target_node):
        if target_node is None:
            return None
        topology = tuple(((floor_id, floor.cluster_grid, floor.distance_map) for floor_id, floor in world.floors.items()))
        same_topology = len(topology) == len(self.reward_topology) and all((first[0] == second[0] and first[1] is second[1] and (first[2] is second[2]) for first, second in zip(topology, self.reward_topology)))
        if self.reward_world is not world or not same_topology:
            self.reward_distance_fields.clear()
            self.reward_world, self.reward_topology = (world, topology)
        if target_node in self.reward_distance_fields:
            return self.reward_distance_fields[target_node]
        distances = {floor_id: np.full((floor.map_grid.height, floor.map_grid.width), np.inf, dtype=np.float32) for floor_id, floor in world.floors.items()}
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
                nx, ny = (x + dx, y + dy)
                if (dx or dy) and world.can_move_to(nx, ny, floor_id) and (distances[floor_id][ny, nx] > next_distance):
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

    def _sample_replay(self, batch_size=BATCH_SIZE):
        return random.sample(self.replay_buffer, batch_size)

    def _observe(self, simulation, drone, route):
        world = simulation.grid_world
        floor = world.floors[drone.floor_id]
        cluster_grid = floor.cluster_grid
        route = list(route)
        if self.assigned_routes.get(drone) != tuple(route):
            self._flush_returns(drone)
            self.route_cursors.pop(drone, None)
        self.assigned_routes[drone] = tuple(route)
        route_cursor = self.route_cursors.get(drone, 0)
        current_cluster = cluster_grid.get(drone.grid_x, drone.grid_y)
        current_node = (drone.floor_id, current_cluster)
        if current_node in route:
            matching_indices = [index for index, node in enumerate(route) if node == current_node and index >= route_cursor]
            if matching_indices:
                current_index = matching_indices[-1]
                route_cursor = min(current_index + 1, len(route))
                self.route_cursors[drone] = route_cursor
        remaining_route = route[route_cursor:] if route else []
        route_ids = set(route[max(0, route_cursor - 1):]) if route else set()
        target_node = remaining_route[0] if remaining_route else route[-1] if route else None
        self.route_targets[drone] = target_node
        state = np.zeros((CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for py in range(PATCH_SIZE):
            for px in range(PATCH_SIZE):
                x = drone.grid_x + px - PATCH_RADIUS
                y = drone.grid_y + py - PATCH_RADIUS
                if not floor.map_grid.in_bounds(x, y):
                    state[0, py, px] = 1.0
                    continue
                state[0, py, px] = not world.can_move_to(x, y, drone.floor_id)
                state[1, py, px] = target_node is not None and (drone.floor_id, cluster_grid.get(x, y)) == target_node
        offsets = np.zeros(2, dtype=np.float32)
        if target_node is not None and target_node[0] == drone.floor_id:
            cells = floor.cluster_grid.clusters[target_node[1]].cells
            walkable = [(cx, cy) for cx, cy in cells if world.can_move_to(cx, cy, drone.floor_id)]
            if walkable:
                center_x = sum((cx for cx, _ in cells)) / len(cells)
                center_y = sum((cy for _, cy in cells)) / len(cells)
                target_x, target_y = min(walkable, key=lambda cell: ((cell[0] - center_x) ** 2 + (cell[1] - center_y) ** 2, cell[1], cell[0]))
                offsets[:] = ((target_x - drone.grid_x) / floor.map_grid.width, (target_y - drone.grid_y) / floor.map_grid.height)
        tick = simulation.step_count
        previous = self.frame_history.get(drone)
        if previous is None:
            frames = deque((state.copy() for _ in range(HISTORY_FRAMES)), maxlen=HISTORY_FRAMES)
        else:
            observed_tick, frames = previous
            if tick == observed_tick:
                frames[-1] = state.copy()
            else:
                frames.append(state.copy())
        self.frame_history[drone] = (tick, frames)
        return ({'map': np.concatenate(tuple(frames), axis=0), 'features': offsets}, route_ids)

    def _legal_actions(self, simulation, drone):
        if drone not in simulation.drones or simulation.touching_exit(drone) or drone.in_stair_transit:
            return [Action.STAY]
        world = simulation.grid_world
        legal_actions = []
        for action, (dx, dy) in ACTION_DIRECTIONS.items():
            if action == Action.STAY:
                continue
            x = drone.grid_x + dx
            y = drone.grid_y + dy
            if not world.can_move_to(x, y, drone.floor_id):
                continue
            legal_actions.append(action)
        return legal_actions

    def action_mask(self, simulation, drone):
        """Use the same movement rules for acting and replay bootstrap targets."""
        mask = np.zeros(len(Action), dtype=np.int8)
        mask[[int(action) for action in self._legal_actions(simulation, drone) if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)]] = 1
        return mask

    def _select_action(self, state, legal_actions, drone, route_ids, simulation):
        legal_actions = [action for action in legal_actions if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)]
        if not legal_actions:
            raise RuntimeError('Drone has no legal movement action')
        self.decision_count += 1
        epsilon = self.epsilon if self.training else 0.0
        if random.random() < epsilon:
            return self._explore(legal_actions, drone)
        with torch.no_grad():
            q_values = self.policy_network(self._state_batch((state,)))[0].cpu().numpy()
        legal_values = q_values[[int(action) for action in legal_actions]]
        if not np.isfinite(legal_values).all() or (self.max_training_q is not None and np.max(np.abs(legal_values)) > self.max_training_q):
            raise RuntimeError('Invalid or divergent DQN action values; start from fresh weights')
        scores = np.full(len(Action), -np.inf, dtype=np.float32)
        for action in legal_actions:
            scores[int(action)] = q_values[int(action)]
        return Action(int(np.argmax(scores)))

    def _explore(self, legal_actions, drone):
        return random.choice(legal_actions)

    def _state_batch(self, states):
        return tuple((torch.from_numpy(np.stack([state[key] for state in states])).to(self.device) for key in ('map', 'features')))

    def _optimize(self):
        if self.target_network is None or self.optimizer is None:
            raise RuntimeError('DQN optimization requires a training controller')
        batch = self._sample_replay()
        states, actions, rewards, next_states, dones, next_legal_masks, discounts = zip(*batch)
        states = self._state_batch(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = self._state_batch(next_states)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_legal_masks = torch.from_numpy(np.stack(next_legal_masks)).to(self.device)
        discounts = torch.tensor(discounts, dtype=torch.float32, device=self.device)
        current_q_values = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            targets = rewards.clone()
            if torch.any(discounts > 0):
                next_actions = self.policy_network(next_states).masked_fill(next_legal_masks == 0, -torch.inf).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                targets += discounts * next_q_values * (1.0 - dones) * next_legal_masks.any(dim=1)
        loss = nn.functional.smooth_l1_loss(current_q_values, targets)
        self.last_loss = float(loss.detach())
        self.last_q_abs_max = float(current_q_values.detach().abs().max())
        self.last_target_abs_max = float(targets.abs().max())
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite DQN loss; stopping training')
        if self.max_training_q is not None and max(self.last_q_abs_max, self.last_target_abs_max) > self.max_training_q:
            raise RuntimeError(f'DQN values exceeded {self.max_training_q:g}; stopping divergent training')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
