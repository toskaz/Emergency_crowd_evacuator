import math
import random
from collections import deque
from pathlib import Path

import numpy as np

from controls.action import Action
from controls.line_of_sight import can_see
from entities.drone import Drone
from grids.cluster import StairCluster
from grids.grid_world import ACTION_DIRECTIONS
from map.cell import CellType


MOVEMENT_MOVES = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
MOVES = [Action.STAY, *MOVEMENT_MOVES, Action.USE_STAIRS]
DIRECTION_COUNT = 8
DISTANCE_BANDS = 3
BLOCKED_PATTERNS = 16

EXIT_REACH = 10
PATH_SEARCH_LIMIT = 80
PANIC_CHANCE = 0.65

LEARNING_RATE = 0.1
DISCOUNT = 0.9

EXIT_REWARD = 5.0
CLOSER_REWARD = 0.2
STEP_REWARD = -0.02
BUMP_REWARD = -0.2
FAILED_MOVE_GRACE = 2
IDLE_REWARD = -0.05
HOLD_REWARD = 0.02


class QLearningFollowController:
    def __init__(self, epsilon=0.0, learning=False):
        self.q_table = np.zeros(
            (DIRECTION_COUNT, DISTANCE_BANDS, BLOCKED_PATTERNS, len(MOVES)),
            dtype=np.float32,
        )
        self.epsilon = epsilon
        self.learning = learning
        self.last_seen = {}
        self.pending = {}
        self.path_cache = {}
        self.exit_cells = {}
        self.failed_moves = {}
        self.pending_stair_arrivals = {}

    def reset(self):
        self.last_seen.clear()
        self.pending.clear()
        self.path_cache.clear()
        self.exit_cells.clear()
        self.failed_moves.clear()
        self.pending_stair_arrivals.clear()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        if not Path(path).exists():
            return False
        q_table = np.load(path)
        legacy_shape = self.q_table.shape[:-1] + (len(MOVES) - 1,)
        if q_table.shape == legacy_shape:
            migrated = np.zeros_like(self.q_table)
            migrated[..., :len(MOVES) - 1] = q_table
            q_table = migrated
        self.q_table = q_table
        return True

    def get_actions(self, simulation):
        leaders = find_leaders(simulation)
        for evacuee, leader in leaders.items():
            while leader in leaders:
                leader = leaders[leader]
            self.last_seen[evacuee] = (leader.floor_id, leader.grid_x, leader.grid_y)
        return [self.decide(simulation, evacuee, leaders.get(evacuee)) for evacuee in simulation.evacuees]

    def decide(self, simulation, evacuee, leader):
        world = simulation.grid_world
        allow_stairs = not self._is_recent_stair_arrival(world, evacuee)
        if exit_distance(world, evacuee) <= EXIT_REACH:
            action = step_towards_exit(
                world, evacuee, allow_stairs=allow_stairs
            )
        elif leader is None:
            action = self.follow_last_seen(
                world, evacuee, allow_stairs=allow_stairs
            )
        else:
            self.remember_drone_position(world, evacuee, leader)
            if leader.floor_id != evacuee.floor_id:
                action = self.follow_last_seen(
                    world, evacuee, allow_stairs=allow_stairs
                )
            else:
                landing = world.stair_cluster_at(
                    evacuee.floor_id, evacuee.grid_x, evacuee.grid_y
                )
                if (
                    landing is not None
                    and landing
                    == world.stair_cluster_at(
                        leader.floor_id, leader.grid_x, leader.grid_y
                    )
                    and world.stair_arrival(evacuee, "evacuee") is not None
                    and allow_stairs
                ):
                    action = Action.STAY
                else:
                    action = self.follow(
                        world, evacuee, leader, allow_stairs=allow_stairs
                    )

        return self._finish_action(world, evacuee, action)

    def _finish_action(self, world, evacuee, action):
        if action is Action.USE_STAIRS:
            self.last_seen.pop(evacuee, None)
            arrival = world.stair_arrival(evacuee, "evacuee")
            if arrival is not None:
                self.pending_stair_arrivals[evacuee] = arrival
        return action

    def _is_recent_stair_arrival(self, world, evacuee):
        arrival = self.pending_stair_arrivals.get(evacuee)
        if arrival is None:
            return False
        floor_id, x, y = arrival
        if evacuee.floor_id != floor_id:
            self.pending_stair_arrivals.pop(evacuee, None)
            return False
        arrived_cluster = world.stair_cluster_at(floor_id, x, y)
        current_cluster = world.stair_cluster_at(
            evacuee.floor_id, evacuee.grid_x, evacuee.grid_y
        )
        if arrived_cluster is not None and current_cluster == arrived_cluster:
            return True
        self.pending_stair_arrivals.pop(evacuee, None)
        return False

    def remember_drone_position(self, world, evacuee, leader):
        if leader.floor_id != evacuee.floor_id:
            stair_target = linked_stair_target(
                world, evacuee, leader.floor_id
            )
            target = (
                (evacuee.floor_id, *stair_target)
                if stair_target is not None
                else (leader.floor_id, leader.grid_x, leader.grid_y)
            )
            self.last_seen[evacuee] = target
            return

        target = self.last_seen.get(
            leader, (leader.floor_id, leader.grid_x, leader.grid_y)
        )
        if target[0] != evacuee.floor_id:
            stair_target = linked_stair_target(
                world, evacuee, target[0]
            )
            if stair_target is not None:
                target = (evacuee.floor_id, *stair_target)
        self.last_seen[evacuee] = target

    def follow(self, world, evacuee, leader, *, allow_stairs=True):
        state = observe(world, evacuee, leader)
        action = self.choose_action(
            state, movement_options(world, evacuee, allow_stairs=allow_stairs)
        )
        if self.learning:
            target = (leader.floor_id, leader.grid_x, leader.grid_y)
            position = (evacuee.floor_id, evacuee.grid_x, evacuee.grid_y)
            old_distance = manhattan(evacuee, leader)
            can_approach = any(
                abs(leader.grid_x - evacuee.grid_x - ACTION_DIRECTIONS[move][0])
                + abs(leader.grid_y - evacuee.grid_y - ACTION_DIRECTIONS[move][1]) < old_distance
                for move in legal_moves(state) if move != Action.STAY
            )
            self.pending[evacuee] = (state, action, target, position, can_approach)
        return action

    def choose_action(self, state, candidates=MOVES):
        if random.random() < self.epsilon:
            return random.choice(candidates)
        values = self.q_table[state]
        best_value = max(values[int(move)] for move in candidates)
        best_moves = [move for move in candidates if values[int(move)] == best_value]
        return random.choice(best_moves)

    def follow_last_seen(self, world, evacuee, *, allow_stairs=True):
        visible_exit = self.nearest_visible_exit(world, evacuee)
        if visible_exit is not None:
            return self.step_closer(
                world, evacuee, *visible_exit, allow_stairs=allow_stairs
            )

        target = self.last_seen.get(evacuee)
        if target is None:
            return panic(world, evacuee, allow_stairs=allow_stairs)

        target_floor, target_x, target_y = target
        if target_floor != evacuee.floor_id:
            stair_target = linked_stair_target(world, evacuee, target_floor)
            if stair_target is None:
                return panic(world, evacuee, allow_stairs=allow_stairs)
            target_x, target_y = stair_target
            self.last_seen[evacuee] = (evacuee.floor_id, target_x, target_y)

        arrival = world.stair_arrival(evacuee, "evacuee")
        if allow_stairs and arrival is not None and arrival[0] == target_floor:
            return self._finish_action(world, evacuee, Action.USE_STAIRS)

        landing = world.stair_cluster_at(
            evacuee.floor_id, evacuee.grid_x, evacuee.grid_y
        )
        target_landing = world.stair_cluster_at(
            evacuee.floor_id, target_x, target_y
        )
        if landing is not None and landing == target_landing:
            if allow_stairs and arrival is not None:
                return self._finish_action(world, evacuee, Action.USE_STAIRS)
        if (evacuee.grid_x, evacuee.grid_y) == (target_x, target_y):
            self.last_seen.pop(evacuee)
            return step_towards_exit(
                world, evacuee, allow_stairs=allow_stairs
            )

        return self.step_closer(
            world, evacuee, target_x, target_y, allow_stairs=allow_stairs
        )

    def nearest_visible_exit(self, world, evacuee):
        floor_id = evacuee.floor_id
        if floor_id not in self.exit_cells:
            self.exit_cells[floor_id] = find_exit_cells(world, floor_id)

        visible = [
            (x, y)
            for x, y in self.exit_cells[floor_id]
            if can_see(world, floor_id, evacuee.grid_x, evacuee.grid_y, x, y)
        ]
        if not visible:
            return None
        return min(visible, key=lambda cell: abs(cell[0] - evacuee.grid_x) + abs(cell[1] - evacuee.grid_y))

    def step_closer(self, world, evacuee, target_x, target_y, *, allow_stairs=True):
        target = (evacuee.floor_id, target_x, target_y)
        if target not in self.path_cache:
            self.path_cache[target] = distances_from(world, *target)
        return step_down(
            self.path_cache[target],
            evacuee,
            movement_options(world, evacuee, allow_stairs=allow_stairs),
        )

    def learn(self, simulation):
        if not self.learning:
            self.pending.clear()
            return 0.0

        still_inside = set(simulation.evacuees)
        leaders = find_leaders(simulation)
        total_reward = 0.0
        for evacuee in list(self.failed_moves):
            if evacuee not in self.pending:
                self.failed_moves.pop(evacuee)

        for evacuee, (state, action, target, position, can_approach) in self.pending.items():
            if evacuee not in still_inside:
                self.failed_moves.pop(evacuee, None)
                reward, next_state = EXIT_REWARD, None
            else:
                reward, next_state = self.reward_and_next_state(
                    simulation, evacuee, leaders.get(evacuee), action, target, position, can_approach
                )
            next_moves = None if next_state is None else movement_options(simulation.grid_world, evacuee)
            self.update(state, action, reward, next_state, next_moves)
            total_reward += reward

        self.pending.clear()
        return total_reward

    def reward_and_next_state(self, simulation, evacuee, leader, action, target, position, can_approach):
        reward = STEP_REWARD
        reward += self._failed_move_penalty(simulation, evacuee, action, position)

        _, target_x, target_y = target
        old_distance = abs(target_x - position[1]) + abs(target_y - position[2])
        new_distance = abs(target_x - evacuee.grid_x) + abs(target_y - evacuee.grid_y)
        reward += CLOSER_REWARD * (max(0, old_distance - 2) - max(0, new_distance - 2))
        if action == Action.STAY:
            if new_distance <= 2:
                reward += HOLD_REWARD
            elif can_approach:
                reward += IDLE_REWARD

        if leader is None:
            return reward, None
        return reward, observe(simulation.grid_world, evacuee, leader)

    def _failed_move_penalty(self, simulation, evacuee, action, position):
        current = (evacuee.floor_id, evacuee.grid_x, evacuee.grid_y)
        if action == Action.STAY or current != position:
            self.failed_moves.pop(evacuee, None)
            return 0.0
        previous = self.failed_moves.get(evacuee)
        signature = (position, int(action))
        count = (
            previous[2] + 1
            if previous is not None and previous[0] == signature
            and previous[1] == simulation.step_count - 1 else 1
        )
        self.failed_moves[evacuee] = (signature, simulation.step_count, count)
        return BUMP_REWARD if count > FAILED_MOVE_GRACE else 0.0

    def update(self, state, action, reward, next_state, next_moves=MOVES):
        future = 0.0 if next_state is None else max(
            self.q_table[next_state][int(move)]
            for move in next_moves
        )
        target = reward + DISCOUNT * future
        current = self.q_table[state][int(action)]
        self.q_table[state][int(action)] = current + LEARNING_RATE * (target - current)


def find_leaders(simulation):
    world = simulation.grid_world
    leaders = {}
    guides = list(simulation.drones)
    guides_by_group = {}
    for drone in guides:
        guides_by_group.setdefault(drone.drone_id, []).append(drone)

    while True:
        found_new_leader = False
        for evacuee in simulation.evacuees:
            if evacuee in leaders or evacuee.assigned_drone_id is None:
                continue
            same_group = guides_by_group.get(evacuee.assigned_drone_id, [])
            leader = nearest_visible(world, evacuee, same_group)
            if leader is not None:
                leaders[evacuee] = leader
                guides.append(evacuee)
                guides_by_group.setdefault(evacuee.assigned_drone_id, []).append(evacuee)
                found_new_leader = True

        if found_new_leader:
            continue

        for evacuee in simulation.evacuees:
            if evacuee in leaders:
                continue
            leader = nearest_visible(world, evacuee, guides)
            if leader is not None:
                leaders[evacuee] = leader
                guides.append(evacuee)
                guides_by_group.setdefault(evacuee.assigned_drone_id, []).append(evacuee)
                break
        else:
            return leaders


def nearest_visible(world, evacuee, candidates):
    visible = [
        candidate
        for candidate in candidates
        if candidate.floor_id == evacuee.floor_id
        and can_see(world, evacuee.floor_id, evacuee.grid_x, evacuee.grid_y, candidate.grid_x, candidate.grid_y)
    ]
    if not visible:
        return None
    return min(visible, key=lambda candidate: manhattan(evacuee, candidate))


def observe(world, evacuee, leader):
    return (
        direction_sector(evacuee, leader),
        distance_band(manhattan(evacuee, leader)),
        blocked_pattern(world, evacuee),
    )


def manhattan(first, second):
    return abs(first.grid_x - second.grid_x) + abs(first.grid_y - second.grid_y)


def direction_sector(evacuee, leader):
    dx = leader.grid_x - evacuee.grid_x
    dy = leader.grid_y - evacuee.grid_y
    if dx == 0 and dy == 0:
        return 0
    angle = math.atan2(dy, dx)
    return round(angle / (2 * math.pi / DIRECTION_COUNT)) % DIRECTION_COUNT


def distance_band(distance):
    if distance <= 2:
        return 0
    if distance <= 5:
        return 1
    return 2


def blocked_pattern(world, evacuee, include_people=True):
    occupancy = world.floors[evacuee.floor_id].evacuee_occupancy_grid
    pattern = 0
    for bit, move in enumerate(MOVEMENT_MOVES):
        dx, dy = ACTION_DIRECTIONS[move]
        x = evacuee.grid_x + dx
        y = evacuee.grid_y + dy
        if not world.can_move_to(x, y, evacuee.floor_id) or (include_people and occupancy.get(x, y) is not None):
            pattern |= 1 << bit
    return pattern


def legal_moves(state):
    return [Action.STAY] + [
        move for bit, move in enumerate(MOVEMENT_MOVES) if not state[2] & (1 << bit)
    ]


def movement_options(world, evacuee, *, allow_stairs=True):
    options = legal_moves(
        (0, 0, blocked_pattern(world, evacuee, include_people=False))
    )
    if allow_stairs and world.stair_arrival(evacuee, "evacuee") is not None:
        options.append(Action.USE_STAIRS)
    return options


def panic(world, evacuee, *, allow_stairs=True):
    distance = exit_distance(world, evacuee)
    if distance <= EXIT_REACH:
        return step_towards_exit(world, evacuee, allow_stairs=allow_stairs)
    if random.random() < PANIC_CHANCE:
        return random.choice(
            movement_options(world, evacuee, allow_stairs=allow_stairs)
        )
    return step_towards_exit(world, evacuee, allow_stairs=allow_stairs)


def find_exit_cells(world, floor_id):
    walls = world.floors[floor_id].wall_and_exit_grid
    return [
        (x, y)
        for y in range(walls.height)
        for x in range(walls.width)
        if walls.get(x, y) == CellType.EXIT
    ]


def exit_distance(world, evacuee):
    return world.floors[evacuee.floor_id].distance_map[evacuee.grid_y][evacuee.grid_x]


def step_towards_exit(world, entity, *, allow_stairs=True):
    distances = world.floors[entity.floor_id].distance_map
    best_action = Action.STAY
    best_distance = distances[entity.grid_y][entity.grid_x]
    entity_type = "drone" if isinstance(entity, Drone) else "evacuee"
    occupancy = (
        world.floors[entity.floor_id].evacuee_occupancy_grid
        if entity_type == "evacuee"
        else None
    )
    arrival = world.stair_arrival(entity, entity_type)
    if allow_stairs and arrival is not None:
        floor_id, x, y = arrival
        distance = world.floors[floor_id].distance_map[y][x]
        if distance < best_distance:
            best_action = Action.USE_STAIRS
            best_distance = distance

    candidates = []
    for move in MOVEMENT_MOVES:
        dx, dy = ACTION_DIRECTIONS[move]
        x = entity.grid_x + dx
        y = entity.grid_y + dy
        if not world.can_move_to(x, y, entity.floor_id):
            continue
        distance = distances[y][x]
        is_occupied = occupancy is not None and occupancy.get(x, y) not in (
            None,
            entity,
        )
        candidates.append((distance, is_occupied, move))

    improving = [
        candidate for candidate in candidates
        if candidate[0] < best_distance and not candidate[1]
    ]
    if improving:
        return min(improving, key=lambda candidate: candidate[0])[2]

    current_distance = best_distance
    if occupancy is not None and math.isfinite(current_distance):
        occupied_improving = [
            candidate for candidate in candidates if candidate[0] < current_distance
        ]
        preferred = []
        if occupied_improving:
            primary_action = min(
                occupied_improving, key=lambda candidate: candidate[0]
            )[2]
            dx, dy = ACTION_DIRECTIONS[primary_action]
            lateral_directions = ((-dy, dx), (dy, -dx))
            lateral_actions = {
                move
                for move, direction in ACTION_DIRECTIONS.items()
                if direction in lateral_directions
            }
            preferred = [
                candidate for candidate in candidates
                if candidate[2] in lateral_actions
                and not candidate[1]
                and candidate[0] <= current_distance + 1
            ]
        if preferred:
            return min(preferred, key=lambda candidate: candidate[0])[2]

        sidesteps = [
            candidate for candidate in candidates
            if not candidate[1] and candidate[0] <= current_distance + 1
        ]
        if sidesteps:
            return min(sidesteps, key=lambda candidate: candidate[0])[2]

    occupied_improving = [
        candidate for candidate in candidates if candidate[0] < best_distance
    ]
    if occupied_improving:
        best_action = min(occupied_improving, key=lambda candidate: candidate[0])[2]
        best_distance = min(candidate[0] for candidate in occupied_improving)

    return best_action


def step_down(distances, evacuee, candidates=MOVES):
    best_action = Action.STAY
    best_distance = distances.get((evacuee.grid_x, evacuee.grid_y))
    if best_distance is None:
        return Action.STAY

    for move in candidates:
        if move in (Action.STAY, Action.USE_STAIRS):
            continue
        dx, dy = ACTION_DIRECTIONS[move]
        distance = distances.get((evacuee.grid_x + dx, evacuee.grid_y + dy))
        if distance is not None and distance < best_distance:
            best_action = move
            best_distance = distance

    return best_action


def distances_from(world, floor_id, start_x, start_y):
    distances = {(start_x, start_y): 0}
    queue = deque([(start_x, start_y)])

    while queue:
        x, y = queue.popleft()
        if distances[(x, y)] >= PATH_SEARCH_LIMIT:
            continue
        for move in MOVEMENT_MOVES:
            dx, dy = ACTION_DIRECTIONS[move]
            neighbor = (x + dx, y + dy)
            if neighbor in distances or not world.can_move_to(*neighbor, floor_id):
                continue
            distances[neighbor] = distances[(x, y)] + 1
            queue.append(neighbor)

    return distances


def linked_stair_target(world, evacuee, target_floor_id):
    floor = world.floors[evacuee.floor_id]
    reachable = distances_from(
        world, evacuee.floor_id, evacuee.grid_x, evacuee.grid_y
    )
    candidates = []
    for cluster in floor.cluster_grid.clusters:
        if not isinstance(cluster, StairCluster) or cluster.linked_to is None:
            continue
        if cluster.linked_to[0] != target_floor_id:
            continue
        for x, y in cluster.cells:
            distance = reachable.get((x, y))
            if distance is not None:
                candidates.append((distance, y, x))
    if not candidates:
        return None
    _, y, x = min(candidates)
    return x, y
