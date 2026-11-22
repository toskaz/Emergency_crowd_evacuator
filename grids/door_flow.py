import math
from collections import defaultdict

from grids.cluster import ClusterType
from utils import config


class Doorway:
    def __init__(self, floor_id, doorway_id, kind, cluster_ids, cells, width, rate_per_cell):
        self.floor_id = floor_id
        self.doorway_id = doorway_id
        self.kind = kind
        self.cluster_ids = set(cluster_ids)
        self.cells = set(cells)
        self.width = width
        self.capacity = width * rate_per_cell
        self.measured_capacity = self.capacity
        self.approach_cluster_ids = set()
        self.queue = 0
        self.queue_ticks = 0.0
        self.streak = 0
        self.congested = False
        self.passed = 0
        self.passed_this_tick = 0
        self.max_queue = 0
        self.total_wait = 0

    @property
    def key(self):
        return self.floor_id, self.doorway_id

    @property
    def nodes(self):
        return {(self.floor_id, cluster_id) for cluster_id in self.cluster_ids}

    def allowed(self, tick):
        return (
            math.floor(self.capacity * (tick + 1) + 1e-9)
            - math.floor(self.capacity * tick + 1e-9)
        )

    def effective_capacity(self):
        return max(self.measured_capacity, 1e-6)


def build_doorways(floor_id, cluster_grid, rate_per_cell):
    grouped = defaultdict(list)
    for cluster_id, cluster in enumerate(cluster_grid.clusters):
        if cluster.doorway_id is None or cluster.type not in (ClusterType.DOOR, ClusterType.EXIT):
            continue
        grouped[cluster.doorway_id].append(cluster_id)

    doorways = []
    for doorway_id, cluster_ids in sorted(grouped.items()):
        clusters = [cluster_grid.clusters[cluster_id] for cluster_id in cluster_ids]
        kind = "exit" if clusters[0].type is ClusterType.EXIT else "door"
        cells = [cell for cluster in clusters for cell in cluster.cells]
        width = min(cluster.width or 1 for cluster in clusters)
        doorways.append(
            Doorway(floor_id, doorway_id, kind, cluster_ids, cells, width, rate_per_cell)
        )

    by_cluster = {
        cluster_id: doorway for doorway in doorways for cluster_id in doorway.cluster_ids
    }
    for first, second in cluster_grid.adjacent_cluster_pairs:
        first_doorway = by_cluster.get(first)
        second_doorway = by_cluster.get(second)
        if first_doorway is not None and first_doorway is not second_doorway:
            first_doorway.approach_cluster_ids.add(second)
        if second_doorway is not None and second_doorway is not first_doorway:
            second_doorway.approach_cluster_ids.add(first)
    return doorways


class DoorFlow:
    def __init__(self, world, rate_per_cell=None):
        self.world = world
        self.rate_per_cell = config.door_flow_per_cell if rate_per_cell is None else rate_per_cell
        self.tick = 0
        self.waiting = {}
        self.moved = set()
        self.newly_congested = []
        self.doorways = []
        self.exit_budgets = None
        self._topology = None
        self.build()

    def build(self):
        self.doorways = []
        self.by_cell = {}
        self.by_node = {}
        self.approach_of = defaultdict(list)
        for floor_id, floor in self.world.floors.items():
            for doorway in build_doorways(floor_id, floor.cluster_grid, self.rate_per_cell):
                self.doorways.append(doorway)
                for x, y in doorway.cells:
                    self.by_cell[(floor_id, x, y)] = doorway
                for node in doorway.nodes:
                    self.by_node[node] = doorway
                for cluster_id in doorway.approach_cluster_ids:
                    self.approach_of[(floor_id, cluster_id)].append(doorway)
        self._topology = self._current_topology()

    def _current_topology(self):
        return tuple(
            (floor_id, id(floor.cluster_grid)) for floor_id, floor in self.world.floors.items()
        )

    def sync(self):
        if self._current_topology() != self._topology:
            self.build()

    def doorway_at(self, floor_id, x, y):
        return self.by_cell.get((floor_id, x, y))

    def doorway_of_node(self, node):
        return self.by_node.get(node)

    def get(self, key):
        for doorway in self.doorways:
            if doorway.key == key:
                return doorway
        return None

    def _entered_doorway(self, entity, destination):
        target = self.doorway_at(*destination)
        if target is None or target.kind != "door":
            return None
        if target is self.doorway_at(entity.floor_id, entity.grid_x, entity.grid_y):
            return None
        return target

    def limit(self, entities, desired_moves, resolve):
        self.sync()
        budgets = {doorway.key: doorway.allowed(self.tick) for doorway in self.doorways}
        granted = {}
        candidates = []
        for entity in entities:
            if entity not in desired_moves:
                continue
            if self.world.is_exit(entity.grid_x, entity.grid_y, entity.floor_id):
                continue
            target = self._entered_doorway(entity, desired_moves[entity])
            if target is None:
                granted[entity] = desired_moves[entity]
            else:
                candidates.append((entity, target))
        candidates.sort(key=lambda item: -self.waiting.get(item[0], 0))

        refused = set()
        pending = list(candidates)
        moves = resolve(granted)
        while pending:
            admitted = []
            remaining = []
            for entity, target in pending:
                if budgets[target.key] > 0:
                    budgets[target.key] -= 1
                    granted[entity] = desired_moves[entity]
                    admitted.append((entity, target))
                else:
                    remaining.append((entity, target))
            if not admitted:
                refused.update(entity for entity, _ in remaining)
                break
            moves = resolve(granted)
            failed = [(entity, target) for entity, target in admitted if entity not in moves]
            for entity, target in failed:
                budgets[target.key] += 1
                del granted[entity]
                refused.add(entity)
            if failed:
                moves = resolve(granted)
            pending = remaining
            if not failed:
                refused.update(entity for entity, _ in remaining)
                break

        for entity, target in candidates:
            if entity in moves:
                target.passed_this_tick += 1
                self.waiting.pop(entity, None)
            elif entity in refused:
                self.waiting[entity] = self.waiting.get(entity, 0) + 1
        return moves

    def record_moves(self, moves):
        self.moved.update(moves)

    def exit_departures(self, candidates):
        self.sync()
        if self.exit_budgets is None:
            self.exit_budgets = {
                doorway.key: doorway.allowed(self.tick) for doorway in self.doorways
            }
        budgets = self.exit_budgets
        ordered = sorted(candidates, key=lambda entity: -self.waiting.get(entity, 0))
        leaving = []
        for entity in ordered:
            doorway = self.doorway_at(entity.floor_id, entity.grid_x, entity.grid_y)
            if doorway is None:
                leaving.append(entity)
                continue
            if budgets[doorway.key] > 0:
                budgets[doorway.key] -= 1
                doorway.passed_this_tick += 1
                self.waiting.pop(entity, None)
                leaving.append(entity)
            else:
                self.waiting[entity] = self.waiting.get(entity, 0) + 1
        return leaving

    def end_tick(self, evacuees):
        queues = defaultdict(int)
        for evacuee in evacuees:
            if evacuee in self.moved:
                continue
            floor = self.world.floors[evacuee.floor_id]
            cluster_id = floor.cluster_grid.get(evacuee.grid_x, evacuee.grid_y)
            if cluster_id is None or cluster_id < 0:
                continue
            node = (evacuee.floor_id, cluster_id)
            own = self.by_node.get(node)
            touched = set()
            if own is not None:
                touched.add(own.key)
            for doorway in self.approach_of.get(node, ()):
                touched.add(doorway.key)
            for key in touched:
                queues[key] += 1

        self.newly_congested = []
        smoothing = config.door_flow_smoothing
        for doorway in self.doorways:
            doorway.queue = queues.get(doorway.key, 0)
            doorway.max_queue = max(doorway.max_queue, doorway.queue)
            doorway.total_wait += doorway.queue
            doorway.passed += doorway.passed_this_tick
            if doorway.queue > 0:
                observed = min(doorway.passed_this_tick, doorway.capacity)
                doorway.measured_capacity += smoothing * (observed - doorway.measured_capacity)
                doorway.measured_capacity = max(doorway.measured_capacity, 0.25 * doorway.capacity)
            doorway.queue_ticks = doorway.queue / doorway.effective_capacity()
            if (
                doorway.queue >= config.door_congestion_min_people
                and doorway.queue_ticks > config.door_congestion_ticks
            ):
                doorway.streak += 1
            else:
                doorway.streak = 0
            was_congested = doorway.congested
            doorway.congested = doorway.streak >= config.door_congestion_streak
            if doorway.congested and not was_congested:
                self.newly_congested.append(doorway)
            doorway.passed_this_tick = 0

        active = set(evacuees)
        for entity in list(self.waiting):
            if entity not in active:
                del self.waiting[entity]
        self.moved = set()
        self.exit_budgets = None
        self.tick += 1

    def summary(self):
        return [
            {
                "floor": doorway.floor_id,
                "doorway": doorway.doorway_id,
                "kind": doorway.kind,
                "width": doorway.width,
                "capacity": round(doorway.capacity, 2),
                "measured": round(doorway.measured_capacity, 2),
                "passed": doorway.passed,
                "max_queue": doorway.max_queue,
                "total_wait": doorway.total_wait,
            }
            for doorway in self.doorways
        ]
