import math
import random
from collections import deque
from pathlib import Path

import numpy as np

from controls.action import Action
from controls.line_of_sight import can_see
from entities.drone import Drone
from grids.grid_world import ACTION_DIRECTIONS
from map.cell import CellType


MOVES = [Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
DIRECTION_COUNT = 8
DISTANCE_BANDS = 3
BLOCKED_PATTERNS = 16

EXIT_REACH = 10
PATH_SEARCH_LIMIT = 80
PANIC_CHANCE = 0.65
GROUP_SWITCH_TICKS = 30

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
        self._group_follow_streaks = {}

    def reset(self):
        self.last_seen.clear()
        self.pending.clear()
        self.path_cache.clear()
        self.exit_cells.clear()
        self.failed_moves.clear()
        self._group_follow_streaks.clear()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        if not Path(path).exists():
            return False
        self.q_table = np.load(path)
        return True

    def get_actions(self, simulation):
        leaders = find_leaders(simulation, following_only=True)
        tick = simulation.step_count
        active_streaks = {}
        for evacuee, leader in leaders.items():
            if leader is not None:
                evacuee.has_been_guided = True
            while leader in leaders:
                leader = leaders[leader]
            self.last_seen[evacuee] = (leader.floor_id, leader.grid_x, leader.grid_y)
            group_id = leader.drone_id
            if group_id is None or group_id == evacuee.assigned_drone_id:
                continue

            previous_group, first_tick, last_tick = self._group_follow_streaks.get(
                evacuee, (None, tick, tick)
            )
            if previous_group != group_id or last_tick not in (tick - 1, tick):
                first_tick = tick
            if tick - first_tick + 1 >= GROUP_SWITCH_TICKS:
                evacuee.assigned_drone_id = group_id
            else:
                active_streaks[evacuee] = (group_id, first_tick, tick)
        self._group_follow_streaks = active_streaks
        return [self.decide(simulation, evacuee, leaders.get(evacuee)) for evacuee in simulation.evacuees]

    def decide(self, simulation, evacuee, leader):
        if getattr(evacuee, "in_stair_transit", False):
            return Action.STAY
        world = simulation.grid_world
        if exit_distance(world, evacuee) <= EXIT_REACH:
            return step_towards_exit(world, evacuee)
        if leader is None:
            return self.follow_last_seen(world, evacuee)

        self.remember_drone_position(evacuee, leader)
        landing = world.stair_cluster_at(evacuee.floor_id, evacuee.grid_x, evacuee.grid_y)
        if (
            landing is not None
            and landing == world.stair_cluster_at(leader.floor_id, leader.grid_x, leader.grid_y)
            and world.stair_arrival(evacuee, "evacuee") is not None
        ):
            return Action.STAY
        return self.follow(world, evacuee, leader)

    def remember_drone_position(self, evacuee, leader):
        if leader in self.last_seen:
            self.last_seen[evacuee] = self.last_seen[leader]
        else:
            self.last_seen[evacuee] = (leader.floor_id, leader.grid_x, leader.grid_y)

    def follow(self, world, evacuee, leader):
        state = observe(world, evacuee, leader)
        action = self.choose_action(state, movement_options(world, evacuee))
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

    def follow_last_seen(self, world, evacuee):
        visible_exit = self.nearest_visible_exit(world, evacuee)
        if visible_exit is not None:
            return self.step_closer(world, evacuee, *visible_exit)

        target = self.last_seen.get(evacuee)
        if target is None or target[0] != evacuee.floor_id:
            return panic(world, evacuee)

        _, target_x, target_y = target
        landing = world.stair_cluster_at(evacuee.floor_id, evacuee.grid_x, evacuee.grid_y)
        if landing is not None and landing == world.stair_cluster_at(target[0], target_x, target_y):
            if world.stair_arrival(evacuee, "evacuee") is not None:
                return Action.USE_STAIRS
        if (evacuee.grid_x, evacuee.grid_y) == (target_x, target_y):
            self.last_seen.pop(evacuee)
            return panic(world, evacuee)

        return self.step_closer(world, evacuee, target_x, target_y)

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

    def step_closer(self, world, evacuee, target_x, target_y):
        target = (evacuee.floor_id, target_x, target_y)
        if target not in self.path_cache:
            self.path_cache[target] = distances_from(world, *target)
        return step_down(self.path_cache[target], evacuee, movement_options(world, evacuee))

    def learn(self, simulation):
        if not self.learning:
            self.pending.clear()
            return 0.0

        still_inside = set(simulation.evacuees)
        leaders = find_leaders(simulation, following_only=True)
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


def find_leaders(simulation, *, following_only=False):
    """Resolve visible chains rooted at drones without changing assignments."""
    world = simulation.grid_world
    leaders = {}
    guides = [drone for drone in simulation.drones if not drone.in_stair_transit]
    evacuees = [
        person for person in simulation.evacuees
        if not person.in_stair_transit
        and (not following_only or exit_distance(world, person) > EXIT_REACH)
    ]
    guides_by_group = {}
    for drone in guides:
        guides_by_group.setdefault(drone.drone_id, []).append(drone)

    while True:
        found_new_leader = False
        for evacuee in evacuees:
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

        for evacuee in evacuees:
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
    for bit, move in enumerate(MOVES[1:]):
        dx, dy = ACTION_DIRECTIONS[move]
        x = evacuee.grid_x + dx
        y = evacuee.grid_y + dy
        if not world.can_move_to(x, y, evacuee.floor_id) or (include_people and occupancy.get(x, y) is not None):
            pattern |= 1 << bit
    return pattern


def legal_moves(state):
    return [Action.STAY] + [
        move for bit, move in enumerate(MOVES[1:]) if not state[2] & (1 << bit)
    ]


def movement_options(world, evacuee):
    return legal_moves((0, 0, blocked_pattern(world, evacuee, include_people=False)))


def panic(world, evacuee):
    if random.random() < PANIC_CHANCE:
        return random.choice(movement_options(world, evacuee))
    return step_towards_exit(world, evacuee)


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


def step_towards_exit(world, entity):
    distances = world.floors[entity.floor_id].distance_map
    best_action = Action.STAY
    best_distance = distances[entity.grid_y][entity.grid_x]
    entity_type = "drone" if isinstance(entity, Drone) else "evacuee"
    arrival = world.stair_arrival(entity, entity_type)
    if arrival is not None:
        floor_id, x, y = arrival
        distance = world.floors[floor_id].distance_map[y][x]
        if distance < best_distance:
            best_action = Action.USE_STAIRS
            best_distance = distance

    for move in MOVES[1:]:
        dx, dy = ACTION_DIRECTIONS[move]
        x = entity.grid_x + dx
        y = entity.grid_y + dy
        if world.can_move_to(x, y, entity.floor_id) and distances[y][x] < best_distance:
            best_action = move
            best_distance = distances[y][x]

    return best_action


def step_down(distances, evacuee, candidates=MOVES):
    best_action = Action.STAY
    best_distance = distances.get((evacuee.grid_x, evacuee.grid_y))
    if best_distance is None:
        return Action.STAY

    for move in candidates:
        if move == Action.STAY:
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
        for move in MOVES[1:]:
            dx, dy = ACTION_DIRECTIONS[move]
            neighbor = (x + dx, y + dy)
            if neighbor in distances or not world.can_move_to(*neighbor, floor_id):
                continue
            distances[neighbor] = distances[(x, y)] + 1
            queue.append(neighbor)

    return distances
