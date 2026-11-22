"""A* guided Double DQN for escorting groups of evacuees."""

from dataclasses import dataclass
import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn

from controls.action import Action
from controls.drone_astar import AStarDroneController
from controls.drone_dqn import ACTION_DIRECTIONS, PATCH_RADIUS, PATCH_SIZE, DQNDroneController
from controls.evacuee_rl import EXIT_REACH, exit_distance, find_leaders
from map.cell import CellType


CHANNELS = (
    "blocked", "wall_clearance", "other_drones", "assigned_people", "other_people",
    "astar_path", "local_goal_distance", "local_goal", "stairs", "exits",
)
FEATURES = (
    "waypoint_x", "waypoint_y", "next_floor", "visible_people", "recently_seen_people",
    "visible_progress", "ticks_without_movement",
) + tuple(f"previous_{action.name.lower()}" for action in Action) + tuple(
    f"next_astar_{action.name.lower()}" for action in Action
)
FEATURES += ("solo_mode",)
NEXT_ASTAR_OFFSET = len(FEATURES) - len(Action) - 1
HISTORY_TICKS = 12
STRANDED_TICKS = 30
DISTANCE_SCALE = 64.0
DRONE_DISTANCE_WEIGHT = 6.4
LOW_COMPANY_COST = 0.05
RETENTION_DISCOUNT = 0.99
REWARD_VERSION = 16
NAVIGATION_REWARD_VERSION = 17


class EscortQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(len(CHANNELS), 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * PATCH_SIZE * PATCH_SIZE, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + len(FEATURES), 64), nn.ReLU(), nn.Linear(64, len(Action)),
        )

    def forward(self, state):
        patch, features = state
        return self.head(torch.cat((self.map_encoder(patch), features), dim=1))


@dataclass
class EscortTransition:
    state: dict
    action: int
    route: tuple
    position: tuple
    group: tuple
    roster_size: int
    exit_distances: dict | None
    potential: float
    stranded_fraction: float
    move_failures: int
    in_stair_transit: bool


class EscortDroneController(DQNDroneController):
    network_type = EscortQNetwork
    observation_version = 6
    observation_shape = (len(CHANNELS), PATCH_SIZE, PATCH_SIZE)
    feature_count = len(FEATURES)
    learning_objective = "astar_escort"

    def __init__(self, training=True, n_step=1, *, path_only=False):
        super().__init__(training=training, n_step=1)
        self.path_only = path_only
        self.solo_episode = path_only
        self.balanced_replay = False
        self.mode_replay = (deque(maxlen=10000), deque(maxlen=10000))
        # Rehearsal changes solo_episode, not this shared return setting.
        # Standalone solo pretraining keeps its original immediate-reward setup.
        self.discount = 0.0 if path_only else RETENTION_DISCOUNT
        self.guide = AStarDroneController(pace_evacuees=False)
        self.reset()

    @property
    def reward_version(self):
        return NAVIGATION_REWARD_VERSION if self.path_only else REWARD_VERSION

    def reset(self):
        super().reset()
        self.guide.reset()
        self.sightings = {}
        self.motion = {}
        self.previous_actions = {}
        self.stranding = {}
        self.rosters = {}
        self.completion_paid = set()

    def invalidate(self):
        self.guide.invalidate()

    @staticmethod
    def _position(entity):
        return entity.floor_id, entity.grid_x, entity.grid_y

    def _visible_evacuees(self, simulation, drone):
        return [person for person in super()._visible_evacuees(simulation, drone)
                if not person.in_stair_transit and not drone.in_stair_transit]

    def _observe(self, simulation, drone, route):
        world = simulation.grid_world
        floor_id, x, y = self._position(drone)
        floor = world.floors[floor_id]
        self.assigned_routes[drone] = tuple(route)
        path, goal = self.guide.route_guidance(simulation, drone, route)
        clearance = self.guide.wall_distances(floor_id)
        distance = self._reward_distances(world, goal)
        visible = self._visible_evacuees(simulation, drone)
        assigned = [person for person in visible if drone.drone_id is not None
                    and person.assigned_drone_id == drone.drone_id]
        own_cells = {self._position(person) for person in assigned}
        other_cells = {self._position(person) for person in visible} - own_cells
        drones = {self._position(other) for other in simulation.drones
                  if other is not drone and not other.in_stair_transit}
        path_cells = set(path[1:])
        patch = np.zeros(self.observation_shape, dtype=np.float32)
        patch[0].fill(1)
        patch[6].fill(1)
        for py in range(PATCH_SIZE):
            for px in range(PATCH_SIZE):
                cx, cy = x + px - PATCH_RADIUS, y + py - PATCH_RADIUS
                if not floor.map_grid.in_bounds(cx, cy):
                    continue
                cell = (floor_id, cx, cy)
                terrain = floor.wall_and_exit_grid.get(cx, cy)
                steps = self._field_distance(distance, cell)
                patch[:, py, px] = (
                    not world.can_move_to(cx, cy, floor_id),
                    clearance[cy][cx] / max(self.guide.wall_clearance, 1),
                    cell in drones, cell in own_cells, cell in other_cells, cell in path_cells,
                    min(steps / (2 * PATCH_SIZE), 1.0),
                    goal is not None and (floor_id, floor.cluster_grid.get(cx, cy)) == goal,
                    terrain is CellType.STAIRS, terrain is CellType.EXIT,
                )
        features = self._features(simulation, drone, route, path, assigned)
        return {"map": patch, "features": features}, set(route)

    def _features(self, simulation, drone, route, path, assigned):
        position = self._position(drone)
        tick = simulation.step_count
        previous_position, moved_at = self.motion.get(drone, (position, tick))
        if previous_position != position:
            moved_at = tick
        self.motion[drone] = position, moved_at
        features = np.zeros(self.feature_count, dtype=np.float32)
        for cell in path[1:5]:
            if cell[0] != position[0]:
                features[2] = np.sign(cell[0] - position[0])
                break
            features[0] = (cell[1] - position[1]) / PATCH_RADIUS
            features[1] = (cell[2] - position[2]) / PATCH_RADIUS

        # Keep a separate previous snapshot so repeated queries in one tick
        # produce the same history features.
        observed_at, previous, latest = self.sightings.get(drone, (-1, {}, {}))
        if tick != observed_at:
            previous = {person: sighting for person, sighting in latest.items()
                        if tick - sighting[0] <= HISTORY_TICKS}
        latest = dict(previous)
        progress = []
        if assigned and route:
            exit_fields = self._reward_distances(simulation.grid_world, route[-1])
            for person in assigned:
                person_position = self._position(person)
                if person in previous:
                    seen_at, previous_pos = previous[person]
                    gain = self._distance_gain(self._field_distance(exit_fields, previous_pos),
                                               self._field_distance(exit_fields, person_position))
                    progress.append(gain / max(1, tick - seen_at))
                latest[person] = tick, person_position
        self.sightings[drone] = tick, previous, latest
        features[3:7] = (
            min(len(assigned) / 12.0, 1.0), min(len(latest) / 12.0, 1.0),
            np.clip(np.mean(progress), -1, 1) if progress else 0.0,
            min((tick - moved_at) / STRANDED_TICKS, 1.0),
        )
        if drone in self.previous_actions:
            features[7 + int(self.previous_actions[drone])] = 1
        next_action = Action.STAY
        if len(path) > 1 and not drone.in_stair_transit:
            next_cell = path[1]
            if next_cell[0] != position[0]:
                next_action = Action.USE_STAIRS
            else:
                delta = next_cell[1] - position[1], next_cell[2] - position[2]
                next_action = next((action for action, direction in ACTION_DIRECTIONS.items()
                                    if direction == delta), Action.STAY)
        features[NEXT_ASTAR_OFFSET + int(next_action)] = 1
        features[-1] = float(self.path_only or self.solo_episode)
        return features

    def _state_batch(self, states):
        return tuple(torch.from_numpy(np.stack([state[key] for state in states])).to(self.device)
                     for key in ("map", "features"))

    def get_actions(self, simulation, routes):
        actions = super().get_actions(simulation, routes)
        self.previous_actions = dict(zip(simulation.drones, actions))
        return actions

    def record_transition(self, simulation, drone, state, action, route_ids):
        route = self.assigned_routes[drone]
        group = tuple(self._assigned_group(simulation, drone))
        self.initial_groups.setdefault(drone, group)
        roster = self.rosters.setdefault(drone, set())
        roster.update(group)
        distances = self._reward_distances(simulation.grid_world, route[-1] if route else None)
        transition = EscortTransition(
            state, int(action), route, self._position(drone), group, len(roster), distances,
            self._potential(drone, group, distances), self._stranded_fraction(simulation, drone, group),
            self.failed_moves.get(drone, 0), drone.in_stair_transit,
        )
        self.pending_transitions[drone] = transition
        return transition

    def _potential(self, drone, group, distances, evacuated=frozenset()):
        distance = self._field_distance(distances, self._position(drone))
        drone_distance = distance / DISTANCE_SCALE if math.isfinite(distance) else 4.0
        return -DRONE_DISTANCE_WEIGHT * drone_distance

    def _stranded_fraction(self, simulation, drone, group):
        if not group:
            self.stranding.pop(drone, None)
            return 0.0
        leaders = find_leaders(simulation, following_only=True)
        world = simulation.grid_world
        tick = simulation.step_count
        tracked = self.stranding.setdefault(drone, {})
        still_stranded = {}
        stranded = 0
        for person in group:
            remaining = exit_distance(world, person)
            if person in leaders or person.in_stair_transit or remaining <= EXIT_REACH:
                continue
            distance_map = world.floors[person.floor_id].distance_map
            old_map, best, since = tracked.get(person, (None, math.inf, tick))
            if old_map is not distance_map or remaining < best:
                best, since = remaining, tick
            still_stranded[person] = distance_map, best, since
            stranded += tick - since + 1 >= STRANDED_TICKS
        self.stranding[drone] = still_stranded
        return stranded / len(group)

    def _attempt_failed(self, drone, transition):
        return (transition.action != Action.STAY and self._position(drone) == transition.position
                and not drone.in_stair_transit)

    def _commit_transition(self, simulation, drone, transition):
        self.previous_actions[drone] = Action(transition.action)
        self.failed_moves[drone] = transition.move_failures + 1 if self._attempt_failed(drone, transition) else 0
        if transition.group and len(self._evacuated_group(simulation, transition.group)) == len(transition.group):
            self.completion_paid.add(drone)

    def group_finished(self, simulation, drone):
        original = self.initial_groups.get(drone, ())
        return (len(self._evacuated_group(simulation, original)) == len(original)
                and not self._assigned_group(simulation, drone))

    def mission_finished(self, simulation, drone):
        """Reaching the assigned exit is sufficient, even with people inside."""
        return self._at_assigned_exit(
            simulation.grid_world, drone, self.assigned_routes.get(drone, ())
        )

    def _navigation_rewards(self, simulation, drone, transition, *, timed_out):
        failed = timed_out or self.mission_failed(simulation, drone)
        success = self.mission_finished(simulation, drone)
        position = self._position(drone)
        if transition.action == Action.USE_STAIRS and drone.in_stair_transit:
            position = simulation.grid_world.stair_arrival(drone, "drone") or position
        old_distance = self._field_distance(transition.exit_distances, transition.position)
        new_distance = self._field_distance(transition.exit_distances, position)
        progress = 0.0 if transition.in_stair_transit else self._distance_gain(old_distance, new_distance)
        return {
            "time": -0.03,
            "progress": 0.05 * progress,
            "failure": 0.0,
            "mission_failure": -2.0 if failed else 0.0,
            "exit": 2.0 if success and not transition.in_stair_transit else 0.0,
        }

    def reward_components(self, simulation, drone, transition, *, timed_out=False):
        components = self._navigation_rewards(simulation, drone, transition, timed_out=timed_out)
        if self.path_only or self.solo_episode:
            return components
        moved = (self._position(drone) != transition.position
                 or (drone.in_stair_transit and not transition.in_stair_transit))
        components["nearby_people"] = 0.0
        if moved and not transition.in_stair_transit:
            assigned_nearby = sum(
                drone.drone_id is not None and person.assigned_drone_id == drone.drone_id
                for person in self._visible_evacuees(simulation, drone)
            )
            remaining = len(self._assigned_group(simulation, drone))
            target = min(4, remaining)
            if target:
                missing_fraction = max(0, target - assigned_nearby) / target
                components["nearby_people"] = -LOW_COMPANY_COST * missing_fraction
        return components

    def _emit_return(self, drone):
        super()._emit_return(drone)
        experience = self.replay_buffer[-1]
        self.mode_replay[int(experience[0]["features"][-1])].append(experience)

    def _sample_replay(self, batch_size=64):
        if self.balanced_replay and all(len(pool) >= batch_size // 2 for pool in self.mode_replay):
            batch = random.sample(self.mode_replay[0], batch_size // 2)
            batch += random.sample(self.mode_replay[1], batch_size - len(batch))
            random.shuffle(batch)
            return batch
        return super()._sample_replay(batch_size)

    def load(self, path):
        # The added mode input starts with zero weight, preserving the old policy.
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            return False
        if (checkpoint.get("observation_version") == 5
                and checkpoint.get("feature_count") == self.feature_count - 1
                and tuple(checkpoint.get("observation_shape", ())) == self.observation_shape):
            weights = checkpoint["state_dict"]
            old = weights["head.0.weight"]
            weights["head.0.weight"] = torch.cat((old, torch.zeros_like(old[:, :1])), dim=1)
            self.policy_network.load_state_dict(weights)
            if self.target_network is not None:
                self.target_network.load_state_dict(weights)
            return True
        return super().load(path)
