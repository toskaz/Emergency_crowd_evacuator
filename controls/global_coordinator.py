from collections import deque
import logging
import math

import networkx as nx

from grids.cluster import ClusterType
from utils import config

logger = logging.getLogger(__name__)


class GlobalCoordinator:
    def __init__(self, grid_world, routing_algorithm="Dijkstra"):
        self.world = grid_world
        self.drone_routes = {}
        self.routing_algorithm = routing_algorithm
        if self.world and hasattr(self.world, "navigation_graph") and self.world.navigation_graph:
            self.world.navigation_graph.algorithm = routing_algorithm

    def set_routing_algorithm(self, algorithm_name):
        self.routing_algorithm = algorithm_name
        if self.world and hasattr(self.world, "navigation_graph") and self.world.navigation_graph:
            self.world.navigation_graph.algorithm = algorithm_name

    def get_exit_nodes(self):
        return [
            (floor_id, cluster_id)
            for floor_id, floor in self.world.floors.items()
            for cluster_id, cluster in enumerate(floor.cluster_grid.clusters)
            if cluster.type is ClusterType.EXIT
        ]

    def get_drone_cluster(self, drone):
        floor_id = getattr(drone, "floor_id", 0)
        floor = self.world.floors.get(floor_id)
        if floor is None:
            return None
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
            return []

        self.update_flow_weights(drone, current_cluster_node)

        exit_nodes = self.get_exit_nodes()
        distance, route = self.world.navigation_graph.distance_to_closest_source(
            exit_nodes, current_cluster_node, algorithm=self.routing_algorithm
        )

        if math.isinf(distance) or not route:
            logger.warning(
                "Coordinator: No passable route found for drone from cluster %s",
                current_cluster_node,
            )
            self.drone_routes[drone] = []
            return []

        self.drone_routes[drone] = route
        logger.info(
            "Coordinator assigned route for drone from %s: %s (%d clusters)",
            current_cluster_node,
            route,
            len(route),
        )
        return route

    def plan_initial_routes(self, drones):
        self.drone_routes.clear()
        self.update_graph_passability()

        for drone in drones:
            drone_cluster = self.get_drone_cluster(drone)
            self.request_route(drone, drone_cluster)

        return self.drone_routes

    def on_environment_changed(self, drones):
        replanned_drones = []
        self.update_graph_passability()

        for drone in drones:
            if getattr(drone, "in_stair_transit", False):
                logger.debug("Coordinator: Drone %s is in stair transit; deferring replan until arrival.", drone)
                continue
            drone_cluster = self.get_drone_cluster(drone)
            if drone_cluster is None:
                continue

            old_route = self.drone_routes.get(drone, [])
            new_route = self.request_route(drone, drone_cluster)
            if new_route != old_route:
                replanned_drones.append(drone)

        return self.drone_routes, replanned_drones

    def _is_remaining_route_passable(self, drone_cluster, route):
        if not route or drone_cluster is None:
            return False

        if drone_cluster not in route:
            return False

        current_index = route.index(drone_cluster)
        remaining_route = route[current_index:]

        G = self.world.navigation_graph.G
        for i in range(len(remaining_route) - 1):
            u, v = remaining_route[i], remaining_route[i + 1]
            if not G.has_edge(u, v):
                return False
            if G[u][v].get("weight", 1.0) >= 1000.0:
                return False

        for node in remaining_route:
            floor_id, cluster_id = node
            floor = self.world.floors.get(floor_id)
            if floor and hasattr(floor, "temporary_wall_grid") and floor.temporary_wall_grid:
                cluster = floor.cluster_grid.clusters[cluster_id]
                if any(floor.temporary_wall_grid.get(x, y) for x, y in cluster.cells):
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
            else:
                floor_u, cid_u = node_u
                floor_v, cid_v = node_v
                has_hazard_u = False
                has_hazard_v = False
                fl_u = self.world.floors.get(floor_u)
                fl_v = self.world.floors.get(floor_v)
                if fl_u and hasattr(fl_u, "temporary_wall_grid") and fl_u.temporary_wall_grid:
                    cl_u = fl_u.cluster_grid.clusters[cid_u]
                    has_hazard_u = any(fl_u.temporary_wall_grid.get(x, y) for x, y in cl_u.cells)
                if fl_v and hasattr(fl_v, "temporary_wall_grid") and fl_v.temporary_wall_grid:
                    cl_v = fl_v.cluster_grid.clusters[cid_v]
                    has_hazard_v = any(fl_v.temporary_wall_grid.get(x, y) for x, y in cl_v.cells)

                if has_hazard_u or has_hazard_v:
                    G[node_u][node_v]["weight"] = 1000.0
                else:
                    G[node_u][node_v]["weight"] = 1.0

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

    def update_flow_weights(self, drone=None, start_node=None):
        forecast = getattr(self.world, "flow_forecast", None)
        if not config.door_reroute_enabled or forecast is None:
            return
        forecast.apply_edge_weights(self.world.navigation_graph.G, drone, start_node)

    def flow_routes(self, drone, start_node):
        self.update_flow_weights(drone, start_node)
        graph = self.world.navigation_graph.G
        routes = []
        if start_node not in graph:
            return routes
        for exit_node in self.get_exit_nodes():
            try:
                route = nx.dijkstra_path(graph, start_node, exit_node, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not self._is_remaining_route_passable(start_node, route):
                continue
            routes.append(route)
        return routes
