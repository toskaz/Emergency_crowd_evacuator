"""Cell navigation along coordinator routes, with a rolling cluster horizon."""

from collections import deque
from heapq import heappop, heappush
from itertools import count
import math

from controls.action import Action
from controls.evacuee_rl import EXIT_REACH
from controls.global_coordinator import GlobalCoordinator
from grids.grid_world import ACTION_DIRECTIONS
from utils.config import (
    drone_lead_distance,
    drone_lookahead_clusters,
    drone_max_pacing_wait,
    drone_wall_clearance,
    drone_wall_cost,
)


MOVES = tuple((action, delta) for action, delta in ACTION_DIRECTIONS.items()
              if action != Action.STAY)


class AStarDroneController:
    def __init__(self, *, lookahead=drone_lookahead_clusters,
                 wall_clearance=drone_wall_clearance, wall_cost=drone_wall_cost,
                 pace_evacuees=True):
        if not isinstance(lookahead, int) or lookahead < 1:
            raise ValueError("lookahead must be a positive number of clusters")
        if not isinstance(wall_clearance, int) or wall_clearance < 0:
            raise ValueError("wall_clearance must be a nonnegative cell distance")
        if not math.isfinite(wall_cost) or wall_cost < 0:
            raise ValueError("wall_cost must be finite and nonnegative")
        self.lookahead = lookahead
        self.wall_clearance = wall_clearance
        self.wall_cost = wall_cost
        self.pace_evacuees = pace_evacuees
        self.reset()

    def reset(self):
        self.paths = {}
        self.planned_windows = {}
        self.route_cursors = {}
        self._routes = {}
        self._last_moves = {}
        self._blocked_moves = {}
        self._pacing = {}
        self._world = None
        self._topology = None
        self._clearances = {}
        self._exit_distances = {}
        self.coordinator = None

    def invalidate(self, drone=None):
        """Discard paths after a route change or an edit to the environment."""
        if drone is None:
            self.paths.clear()
            self.planned_windows.clear()
            self.route_cursors.clear()
            self._routes.clear()
            self._last_moves.clear()
            self._blocked_moves.clear()
            self._pacing.clear()
        else:
            for cache in (self.paths, self.planned_windows, self.route_cursors,
                          self._routes, self._last_moves, self._blocked_moves,
                          self._pacing):
                cache.pop(drone, None)

    def _sync_world(self, world):
        topology = tuple((floor_id, id(floor.cluster_grid), id(floor.distance_map))
                         for floor_id, floor in world.floors.items())
        if world is not self._world:
            self.invalidate()
            self.coordinator = GlobalCoordinator(world)
        if world is not self._world or topology != self._topology:
            self.invalidate()
            self._clearances.clear()
            self._exit_distances.clear()
        self._world, self._topology = world, topology

    @staticmethod
    def _position(entity):
        return entity.floor_id, entity.grid_x, entity.grid_y

    def _node(self, position):
        floor_id, x, y = position
        return floor_id, self._world.floors[floor_id].cluster_grid.get(x, y)

    def get_actions(self, simulation, routes=None):
        self._sync_world(simulation.grid_world)
        if routes is None:
            routes = self.coordinator.drone_routes
        else:
            self.coordinator.drone_routes = routes
        active = set(simulation.drones)
        for drone in set(self._routes) - active:
            self.invalidate(drone)

        actions = []
        reserved = set()
        for drone in simulation.drones:
            action, destination = self._choose_action(simulation, drone, routes)
            if destination is not None and destination in reserved:
                action, destination = Action.STAY, None
            if destination is not None:
                reserved.add(destination)
                self._last_moves[drone] = (
                    self._position(drone), simulation.step_count,
                )
            actions.append(action)
        return actions

    def _choose_action(self, simulation, drone, routes):
        if getattr(drone, "in_stair_transit", False):
            return Action.STAY, None
        world = self._world
        position = self._position(drone)
        if world.is_exit(drone.grid_x, drone.grid_y, drone.floor_id):
            return Action.STAY, None

        previous = self._last_moves.pop(drone, None)
        if previous is not None and simulation.step_count > previous[1]:
            self._blocked_moves[drone] = (
                self._blocked_moves.get(drone, 0) + 1
                if previous[0] == position else 0
            )
        blocked = self._blocked_moves.get(drone, 0) >= 2
        if blocked:
            self.paths.pop(drone, None)

        for attempt in range(2):
            route = tuple(routes.get(drone, ()))
            if not route:
                route = tuple(self._request_route(drone))
                if not route:
                    return Action.STAY, None
            if route != self._routes.get(drone):
                self.invalidate(drone)
                self._routes[drone] = route
            cursor = self.route_cursors.get(drone, 0)
            node = self._node(position)
            matching = [index for index in range(cursor, len(route))
                        if route[index] == node]
            if matching:
                if matching[0] != cursor:
                    self.paths.pop(drone, None)
                cursor = matching[0]
                self.route_cursors[drone] = cursor
            elif drone not in self.paths:
                self._request_route(drone)
                continue

            path = self.paths.get(drone)
            if path and len(path) > 1 and path[1] == position:
                path.popleft()
            if not path or path[0] != position or len(path) < 2:
                occupied = {
                    self._position(other) for other in simulation.drones
                    if other is not drone
                } if blocked else set()
                path = self._plan(drone, route, cursor, occupied)
            if path and len(path) > 1:
                if self._should_wait(simulation, drone, route):
                    return Action.STAY, None
                destination = path[1]
                if destination[0] != drone.floor_id:
                    arrival = world.stair_arrival(drone, "drone")
                    if (arrival is not None and world.can_move_to(arrival[1], arrival[2], arrival[0])
                            and self._node(arrival) == self._node(destination)):
                        return Action.USE_STAIRS, arrival
                else:
                    delta = destination[1] - drone.grid_x, destination[2] - drone.grid_y
                    for action, direction in MOVES:
                        if direction == delta and world.can_move_to(
                                destination[1], destination[2], destination[0]):
                            return action, destination
                self.paths.pop(drone, None)
            if attempt == 0 and not blocked:
                self._request_route(drone)
            else:
                break
        return Action.STAY, None

    def _request_route(self, drone):
        self.coordinator.update_graph_passability()
        self.invalidate(drone)
        return self.coordinator.request_route(drone)

    def _cluster_cells(self, node):
        floor_id, cluster_id = node
        floor = self._world.floors.get(floor_id)
        if floor is None or cluster_id is None or not 0 <= cluster_id < len(floor.cluster_grid.clusters):
            return set()
        return {(floor_id, x, y) for x, y in floor.cluster_grid.clusters[cluster_id].cells
                if self._world.can_move_to(x, y, floor_id)}

    def _plan(self, drone, route, cursor, occupied):
        start = self._position(drone)
        end = min(cursor + self.lookahead + 1, len(route))
        initial_goals = self._cluster_cells(route[end - 1])
        limits = [end]
        while limits[-1] < len(route):
            limits.append(min(len(route), cursor + 2 * (limits[-1] - cursor)))
        for limit in limits:
            window = route[cursor:limit]
            allowed = set().union(*(self._cluster_cells(node) for node in window))
            links = {frozenset(pair) for pair in zip(window, window[1:])}
            path = self._astar(start, self._cluster_cells(window[-1]), allowed, links, occupied)
            if path:
                self.paths[drone] = path
                self.planned_windows[drone] = window
                return path

        path = self._astar(start, initial_goals, None, None, occupied)
        if path:
            self.paths[drone] = path
            self.planned_windows[drone] = route[cursor:end]
        return path

    def _neighbors(self, position):
        floor_id, x, y = position
        for _, (dx, dy) in MOVES:
            if self._world.can_move_to(x + dx, y + dy, floor_id):
                yield floor_id, x + dx, y + dy
        node = self._world.stair_cluster_at(floor_id, x, y)
        if node is not None:
            stair = self._world._get_stair_cluster(node)
            if stair.linked_to is not None:
                yield from sorted(self._cluster_cells(stair.linked_to))

    def _astar(self, start, goals, allowed, links, occupied):
        goals = goals - occupied
        if not goals or (allowed is not None and start not in allowed):
            return None
        single_floor = (len(self._world.floors) == 1
                        or (allowed is not None
                            and all(cell[0] == start[0] for cell in allowed)))

        def heuristic(position):
            if not single_floor:
                return 0
            return min(abs(position[1] - goal[1]) + abs(position[2] - goal[2])
                       for goal in goals)

        serial = count()
        pending = [(heuristic(start), next(serial), 0.0, start)]
        costs = {start: 0.0}
        parents = {}
        while pending:
            _, _, cost, current = heappop(pending)
            if cost != costs[current]:
                continue
            if current in goals:
                path = deque([current])
                while current in parents:
                    current = parents[current]
                    path.appendleft(current)
                return path
            current_node = self._node(current)
            for neighbor in self._neighbors(current):
                if neighbor in occupied or (allowed is not None and neighbor not in allowed):
                    continue
                neighbor_node = self._node(neighbor)
                if (links is not None and current_node != neighbor_node
                        and frozenset((current_node, neighbor_node)) not in links):
                    continue
                new_cost = cost + self._cell_cost(neighbor)
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    parents[neighbor] = current
                    heappush(pending, (new_cost + heuristic(neighbor),
                                      next(serial), new_cost, neighbor))
        return None

    def _cell_cost(self, position):
        floor_id, x, y = position
        if floor_id not in self._clearances:
            self._clearances[floor_id] = self._wall_distances(floor_id)
        clearance = self._clearances[floor_id][y][x]
        return 1.0 + self.wall_cost * max(0, self.wall_clearance - clearance) ** 2

    def _wall_distances(self, floor_id):
        floor = self._world.floors[floor_id]
        width, height = floor.map_grid.width, floor.map_grid.height
        distances = [[self.wall_clearance] * width for _ in range(height)]
        pending = deque()
        for y in range(height):
            for x in range(width):
                distance = min(self.wall_clearance, x + 1, y + 1, width - x, height - y)
                if not self._world.can_move_to(x, y, floor_id):
                    distance = 0
                distances[y][x] = distance
                if distance < self.wall_clearance:
                    pending.append((x, y))
        while pending:
            x, y = pending.popleft()
            distance = distances[y][x] + 1
            for _, (dx, dy) in MOVES:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and distance < distances[ny][nx]:
                    distances[ny][nx] = distance
                    pending.append((nx, ny))
        return distances

    def _distances_to_exit(self, exit_node):
        if exit_node not in self._exit_distances:
            goals = self._cluster_cells(exit_node)
            distances = {position: 0 for position in goals}
            pending = deque(sorted(goals))
            expanded_stairs = set()
            while pending:
                current = pending.popleft()
                floor_id, x, y = current
                candidates = [(floor_id, x + dx, y + dy) for _, (dx, dy) in MOVES
                              if self._world.can_move_to(x + dx, y + dy, floor_id)]
                stair_node = self._world.stair_cluster_at(floor_id, x, y)
                if stair_node is not None and stair_node not in expanded_stairs:
                    expanded_stairs.add(stair_node)
                    stair = self._world._get_stair_cluster(stair_node)
                    if stair.linked_to is not None:
                        candidates.extend(sorted(self._cluster_cells(stair.linked_to)))
                for neighbor in candidates:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        pending.append(neighbor)
            self._exit_distances[exit_node] = distances
        return self._exit_distances[exit_node]

    def _should_wait(self, simulation, drone, route):
        if not self.pace_evacuees or drone.drone_id is None:
            return False
        group = [person for person in simulation.evacuees
                 if person.assigned_drone_id == drone.drone_id
                 and self._world.floors[person.floor_id].distance_map[person.grid_y][person.grid_x] > EXIT_REACH]
        if not group:
            self._pacing.pop(drone, None)
            return False
        distances = self._distances_to_exit(route[-1])
        drone_distance = distances.get(self._position(drone), math.inf)
        front_distance = min(distances.get(self._position(person), math.inf) for person in group)
        if not math.isfinite(drone_distance) or not math.isfinite(front_distance):
            return False
        gap = front_distance - drone_distance
        waiting, started = self._pacing.get(drone, (False, simulation.step_count))
        if waiting and gap <= drone_lead_distance:
            waiting = False
        elif not waiting and gap > drone_lead_distance + 2:
            waiting, started = True, simulation.step_count
        if waiting and simulation.step_count - started >= drone_max_pacing_wait:
            waiting = False
        self._pacing[drone] = waiting, started
        return waiting
