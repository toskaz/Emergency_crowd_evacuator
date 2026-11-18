from collections import deque
import heapq
import itertools
import logging
import math
from types import SimpleNamespace

from controls.action import Action
from grids.cluster import ClusterType
from grids.grid_world import ACTION_DIRECTIONS
from map.cell import CellType

logger = logging.getLogger(__name__)


class GlobalCoordinator:
    def __init__(self, grid_world):
        self.world = grid_world
        self.drone_routes = {}
        self.cell_routes = {}

    def get_exit_nodes(self):
        return [
            (floor_id, cluster_id)
            for floor_id, floor in self.world.floors.items()
            for cluster_id, cluster in enumerate(floor.cluster_grid.clusters)
            if cluster.type is ClusterType.EXIT
        ]

    def get_drone_cluster(self, drone):
        floor_id = getattr(drone, "floor_id", 0)
        floor = self.world.floors[floor_id]
        cluster_grid = floor.cluster_grid

        cluster_id = cluster_grid.get(drone.grid_x, drone.grid_y)
        if cluster_id is not None and cluster_id >= 0:
            return (floor_id, cluster_id)

        for radius in (1, 2):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx_pos, ny_pos = drone.grid_x + dx, drone.grid_y + dy
                    if not floor.map_grid.in_bounds(nx_pos, ny_pos):
                        continue
                    cid = cluster_grid.get(nx_pos, ny_pos)
                    if cid is not None and cid >= 0:
                        return (floor_id, cid)

        return None

    def request_route(self, drone, current_cluster_node=None):
        if current_cluster_node is None:
            current_cluster_node = self.get_drone_cluster(drone)

        if current_cluster_node is None:
            logger.warning(
                "Coordinator: Drone at floor %d (%d, %d) is outside any valid cluster!",
                getattr(drone, "floor_id", 0) + 1,
                drone.grid_x,
                drone.grid_y,
            )
            self.drone_routes[drone] = []
            self.cell_routes[drone] = []
            return []

        # TODO: przygotowanie pod implementację Flow Rate w grafie
        self.update_flow_weights()

        exit_nodes = self.get_exit_nodes()
        distance, route = self.world.navigation_graph.distance_to_closest_source(
            exit_nodes, current_cluster_node
        )

        if math.isinf(distance) or not route:
            logger.warning(
                "Coordinator: No passable route found for drone from cluster %s",
                current_cluster_node,
            )
            self.drone_routes[drone] = []
            self.cell_routes[drone] = []
            return []

        self.drone_routes[drone] = route
        self.cell_routes[drone] = self.find_cell_route(drone, route)
        logger.info(
            "Coordinator assigned route for drone from %s: %s (%d clusters)",
            current_cluster_node,
            route,
            len(route),
        )
        return route

    def replan_route(self, drone, *, refresh_graph=True):
        if refresh_graph:
            self.update_graph_passability()
        return self.request_route(drone, self.get_drone_cluster(drone))

    def plan_initial_routes(self, drones):
        self.drone_routes.clear()
        self.update_graph_passability()

        for drone in drones:
            self.replan_route(drone, refresh_graph=False)

        return self.drone_routes

    def on_environment_changed(self, drones):
        replanned_drones = []
        self.update_graph_passability()

        for drone in drones:
            current_route = self.drone_routes.get(drone, [])
            drone_cluster = self.get_drone_cluster(drone)

            if not self._is_remaining_route_passable(drone_cluster, current_route):
                logger.info(
                    "Coordinator: Remaining route for drone at %s is blocked! Calculating new path...",
                    drone_cluster,
                )
                self.replan_route(drone, refresh_graph=False)
                replanned_drones.append(drone)
            else:
                cell_route = self.cell_routes.get(drone, [])
                if not self._is_cell_route_passable(cell_route):
                    self.replan_route(drone, refresh_graph=False)
                    replanned_drones.append(drone)
                else:
                    self.cell_routes[drone] = self.find_cell_route(
                        drone, current_route
                    )
                logger.debug(
                    "Coordinator: Drone at %s can continue on its current route, hazard is behind or off-route.",
                    drone_cluster,
                )

        return self.drone_routes, replanned_drones

    def find_cell_route(self, drone, cluster_route):
        """Find a walkable cell path from a drone to its assigned exit cluster."""
        if not cluster_route:
            return []
        target_floor_id, target_cluster_id = cluster_route[-1]
        target_floor = self.world.floors.get(target_floor_id)
        if target_floor is None or not (
            0 <= target_cluster_id < len(target_floor.cluster_grid.clusters)
        ):
            return []
        targets = {
            (target_floor_id, x, y)
            for x, y in target_floor.cluster_grid.clusters[target_cluster_id].cells
            if self.world.can_move_to(x, y, target_floor_id)
        }
        start = (drone.floor_id, drone.grid_x, drone.grid_y)
        if not targets or not self.world.can_move_to(start[1], start[2], start[0]):
            return []

        has_linked_stairs = any(
            cluster.type is ClusterType.STAIRS and cluster.linked_to is not None
            for floor in self.world.floors.values()
            for cluster in floor.cluster_grid.clusters
        )
        heuristic = self._cell_heuristic if not has_linked_stairs else lambda _p, _t: 0
        order = itertools.count()
        frontier = [(heuristic(start, targets), next(order), start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        reached = None

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current in targets:
                reached = current
                break

            for neighbor in self._cell_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if new_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + heuristic(neighbor, targets)
                heapq.heappush(frontier, (priority, next(order), neighbor))

        if reached is None:
            return []

        path = []
        current = reached
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    @staticmethod
    def _cell_heuristic(position, targets):
        floor_id, x, y = position
        same_floor = [
            target for target in targets if target[0] == floor_id
        ]
        if not same_floor:
            return 0
        return min(abs(x - tx) + abs(y - ty) for _, tx, ty in same_floor)

    def _cell_neighbors(self, position):
        floor_id, x, y = position
        for dx, dy in ACTION_DIRECTIONS.values():
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if self.world.can_move_to(nx, ny, floor_id):
                yield floor_id, nx, ny

        floor = self.world.floors[floor_id]
        if floor.map_grid.get(x, y) != CellType.STAIRS:
            return
        entity = SimpleNamespace(floor_id=floor_id, grid_x=x, grid_y=y)
        arrival = self.world.stair_arrival(entity, "drone")
        if arrival is not None and self.world.can_move_to(arrival[1], arrival[2], arrival[0]):
            yield arrival

    def _is_cell_route_passable(self, route):
        if not route:
            return False
        for floor_id, x, y in route:
            if not self.world.can_move_to(x, y, floor_id):
                return False
        for first, second in zip(route, route[1:]):
            if first[0] == second[0]:
                if abs(first[1] - second[1]) + abs(first[2] - second[2]) != 1:
                    return False
            elif second not in self._cell_neighbors(first):
                return False
        return True

    def refresh_cell_routes(self, drones):
        """Refresh paths from current positions and replan stale cluster routes."""
        for drone in drones:
            existing_route = self.cell_routes.get(drone, [])
            if existing_route and not self._is_cell_route_passable(existing_route):
                self.replan_route(drone)
                continue
            route = self.drone_routes.get(drone, [])
            if not route:
                self.cell_routes[drone] = []
                continue
            cell_route = self.find_cell_route(drone, route)
            if not cell_route:
                self.replan_route(drone)
            else:
                self.cell_routes[drone] = cell_route
        return self.cell_routes

    @staticmethod
    def action_for_cell_route(drone, route):
        if not route or route[0] != (drone.floor_id, drone.grid_x, drone.grid_y):
            return None
        if len(route) == 1:
            return Action.STAY
        next_floor, next_x, next_y = route[1]
        if next_floor != drone.floor_id:
            return Action.USE_STAIRS
        delta = (next_x - drone.grid_x, next_y - drone.grid_y)
        return next(
            (action for action, direction in ACTION_DIRECTIONS.items() if direction == delta),
            None,
        )

    def _is_remaining_route_passable(self, drone_cluster, route):
        if not route or drone_cluster is None:
            return False

        if drone_cluster not in route:
            return False

        current_index = route.index(drone_cluster)
        remaining_route = route[current_index:]

        G = self.world.navigation_graph.G
        for i in range(len(remaining_route) - 1):
            if not G.has_edge(remaining_route[i], remaining_route[i + 1]):
                return False

        return True

    def update_graph_passability(self):
        if self.world.navigation_graph is None:
            self.world.rebuild_navigation()

        self.world.navigation_graph.rebuild_base_graph()
        G = self.world.navigation_graph.G

        for node_u, node_v in list(G.edges()):
            if not self.is_cluster_transition_passable(node_u, node_v):
                G.remove_edge(node_u, node_v)
                logger.debug(
                    "Coordinator: Transition between %s and %s is blocked and removed.",
                    node_u,
                    node_v,
                )

    def is_cluster_transition_passable(self, node_u, node_v):
        floor_u, cluster_u_id = node_u
        floor_v, cluster_v_id = node_v

        if floor_u != floor_v:
            cells_u = self.world.floors[floor_u].cluster_grid.clusters[cluster_u_id].cells
            cells_v = self.world.floors[floor_v].cluster_grid.clusters[cluster_v_id].cells
            free_u = any(self.world.can_move_to(x, y, floor_u) for x, y in cells_u)
            free_v = any(self.world.can_move_to(x, y, floor_v) for x, y in cells_v)
            return free_u and free_v

        floor = self.world.floors[floor_u]
        cluster_u = floor.cluster_grid.clusters[cluster_u_id]
        cluster_v = floor.cluster_grid.clusters[cluster_v_id]

        walkable_u = {
            (x, y) for x, y in cluster_u.cells if self.world.can_move_to(x, y, floor_u)
        }
        walkable_v = {
            (x, y) for x, y in cluster_v.cells if self.world.can_move_to(x, y, floor_u)
        }

        if not walkable_u or not walkable_v:
            return False

        has_open_border = any(
            (x + dx, y + dy) in walkable_v
            for x, y in walkable_u
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        if not has_open_border:
            return False

        cx_u, cy_u = cluster_u.get_centroid()
        start_cell = min(walkable_u, key=lambda c: abs(c[0] - cx_u) + abs(c[1] - cy_u))

        allowed_cells = walkable_u | walkable_v
        queue = deque([start_cell])
        visited = {start_cell}

        while queue:
            x, y = queue.popleft()
            if (x, y) in walkable_v:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if neighbor in allowed_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    # TODO: jak zaimplementujemy flow rate, to tutaj bedziemy aktualizowac wagi krawedzi przed wyznaczeniem trasy
    # predykcja w kroku np 50 itp
    def update_flow_weights(self):
        """
        Miejsce przygotowane pod przyszłą implementację Flow Rate (np. predykcja w kroku +50).
        Kiedy zespół zaimplementuje obliczanie przepływu w klastrach, tutaj będą
        aktualizowane wagi krawędzi G[u][v]['weight'] przed wyznaczeniem trasy.
        """
        pass

