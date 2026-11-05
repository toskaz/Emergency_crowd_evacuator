import math
import random
from collections import deque
from pathlib import Path

import numpy as np

from controls.action import Action
from controls.line_of_sight import can_see
from grids.grid_world import ACTION_DIRECTIONS
from map.cell import CellType


MOVES = [Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
DIRECTION_COUNT = 8
DISTANCE_BANDS = 3
BLOCKED_PATTERNS = 16

EXIT_REACH = 5
PATH_SEARCH_LIMIT = 80
PANIC_CHANCE = 0.65

LEARNING_RATE = 0.1
DISCOUNT = 0.9

EXIT_REWARD = 5.0
CLOSER_REWARD = 0.2
STEP_REWARD = -0.01
BUMP_REWARD = -0.2
LOST_SIGHT_REWARD = -1.0


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

    def reset(self):
        self.last_seen.clear()
        self.pending.clear()
        self.path_cache.clear()
        self.exit_cells.clear()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        if not Path(path).exists():
            return False
        self.q_table = np.load(path)
        return True

    def get_actions(self, simulation):
        leaders = find_leaders(simulation)
        return [self.decide(simulation, evacuee, leaders.get(evacuee)) for evacuee in simulation.evacuees]

    def decide(self, simulation, evacuee, leader):
        world = simulation.grid_world
        if exit_distance(world, evacuee) <= EXIT_REACH:
            return step_towards_exit(world, evacuee)
        if leader is None:
            return self.follow_last_seen(world, evacuee)

        self.remember_drone_position(evacuee, leader)
        return self.follow(world, evacuee, leader)

    def remember_drone_position(self, evacuee, leader):
        if leader in self.last_seen:
            self.last_seen[evacuee] = self.last_seen[leader]
        else:
            self.last_seen[evacuee] = (leader.floor_id, leader.grid_x, leader.grid_y)

    def follow(self, world, evacuee, leader):
        state = observe(world, evacuee, leader)
        action = self.choose_action(state)
        self.pending[evacuee] = (state, action, leader, evacuee.grid_x, evacuee.grid_y)
        return action

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(MOVES)
        values = self.q_table[state]
        best_moves = [move for move in MOVES if values[int(move)] == values.max()]
        return random.choice(best_moves)

    def follow_last_seen(self, world, evacuee):
        visible_exit = self.nearest_visible_exit(world, evacuee)
        if visible_exit is not None:
            return self.step_closer(world, evacuee, *visible_exit)

        target = self.last_seen.get(evacuee)
        if target is None or target[0] != evacuee.floor_id:
            return panic(world, evacuee)

        _, target_x, target_y = target
        if (evacuee.grid_x, evacuee.grid_y) == (target_x, target_y):
            if world.floors[evacuee.floor_id].map_grid.get(target_x, target_y) == CellType.STAIRS:
                return Action.USE_STAIRS
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
        return step_down(self.path_cache[target], evacuee)

    def learn(self, simulation):
        if not self.learning:
            self.pending.clear()
            return 0.0

        still_inside = set(simulation.evacuees)
        leaders = find_leaders(simulation)
        total_reward = 0.0

        for evacuee, (state, action, old_leader, old_x, old_y) in self.pending.items():
            if evacuee not in still_inside:
                reward, next_state = EXIT_REWARD, None
            else:
                reward, next_state = self.reward_and_next_state(simulation, evacuee, leaders.get(evacuee), action, old_leader, old_x, old_y)
            self.update(state, action, reward, next_state)
            total_reward += reward

        self.pending.clear()
        return total_reward

    def reward_and_next_state(self, simulation, evacuee, leader, action, old_leader, old_x, old_y):
        reward = STEP_REWARD
        if action is not Action.STAY and (evacuee.grid_x, evacuee.grid_y) == (old_x, old_y):
            reward += BUMP_REWARD

        old_distance = abs(old_leader.grid_x - old_x) + abs(old_leader.grid_y - old_y)
        if distance_band(old_distance) > 0:
            reward += CLOSER_REWARD * (old_distance - manhattan(evacuee, old_leader))

        if leader is None:
            return reward + LOST_SIGHT_REWARD, None
        return reward, observe(simulation.grid_world, evacuee, leader)

    def update(self, state, action, reward, next_state):
        future = 0.0 if next_state is None else self.q_table[next_state].max()
        target = reward + DISCOUNT * future
        current = self.q_table[state][int(action)]
        self.q_table[state][int(action)] = current + LEARNING_RATE * (target - current)


def find_leaders(simulation):
    world = simulation.grid_world
    leaders = {}
    guides = list(simulation.drones)
    found_new_leader = True

    while found_new_leader:
        found_new_leader = False
        for evacuee in simulation.evacuees:
            if evacuee in leaders:
                continue
            leader = nearest_visible(world, evacuee, guides)
            if leader is not None:
                leaders[evacuee] = leader
                guides.append(evacuee)
                found_new_leader = True

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


def blocked_pattern(world, evacuee):
    occupancy = world.floors[evacuee.floor_id].evacuee_occupancy_grid
    pattern = 0
    for bit, move in enumerate(MOVES[1:]):
        dx, dy = ACTION_DIRECTIONS[move]
        x = evacuee.grid_x + dx
        y = evacuee.grid_y + dy
        if not world.can_move_to(x, y, evacuee.floor_id) or occupancy.get(x, y) is not None:
            pattern |= 1 << bit
    return pattern


def panic(world, evacuee):
    if random.random() < PANIC_CHANCE:
        return random.choice(MOVES)
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

    for move in MOVES[1:]:
        dx, dy = ACTION_DIRECTIONS[move]
        x = entity.grid_x + dx
        y = entity.grid_y + dy
        if world.can_move_to(x, y, entity.floor_id) and distances[y][x] < best_distance:
            best_action = move
            best_distance = distances[y][x]

    return best_action


def step_down(distances, evacuee):
    best_action = Action.STAY
    best_distance = distances.get((evacuee.grid_x, evacuee.grid_y))
    if best_distance is None:
        return Action.STAY

    for move in MOVES[1:]:
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
