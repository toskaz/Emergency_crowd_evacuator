from collections import deque
import logging

from controls.action import Action
from controls.drone_assignment import calculate_drone_assignments
from graph_approach.navigation_graphx import NavigationGraphx
from grids.cluster import ClusterType, StairCluster
from grids.cluster_grid import CLUSTERING_VERSION, ClusterGrid
from grids.floor import Floor
from grids.grid import Grid
from grids.map_grid import Map_Grid
from map.cell import CellType


logger = logging.getLogger(__name__)


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


class Grid_World:

    def __init__(self):
        self.floors = {}
        self.active_floor_id = 0
        self.same_height_for_all = True
        self.pending_stair_endpoint = None
        self.stair_link_error = None
        self.stair_link_notice = None
        self.stair_link_notice_time = 0.0
        self.changed_floor_ids = set()
        self.navigation_graph = None
        self.drone_ids = {}
        self.door_flow = None
        self.flow_forecast = None
        if self.floors:
            self.rebuild_map_data()

    def add_floor_from_map(self, map_grid, floor_height_m=3.0):
        floor_id = len(self.floors)
        self.floors[floor_id] = Floor(floor_id, map_grid, floor_height_m=floor_height_m)
        self.active_floor_id = floor_id if floor_id == 0 else self.active_floor_id
        self.rebuild_map_data()

    def to_dict(self):
        floors = []
        for floor_id, floor in self.floors.items():
            cells = [
                [int(floor.map_grid.get(x, y)) for x in range(floor.map_grid.width)]
                for y in range(floor.map_grid.height)
            ]
            floors.append({
                "floor_id": floor_id,
                "width": floor.map_grid.width,
                "height": floor.map_grid.height,
                "cell_size_m": getattr(floor, "cell_size_m", 0.6),
                "floor_height_m": getattr(floor, "floor_height_m", 3.0),
                "cells": cells,
            })

        stair_links = []
        seen_links = set()
        for floor_id, floor in self.floors.items():
            for cluster_id, cluster in enumerate(floor.cluster_grid.clusters):
                if not isinstance(cluster, StairCluster) or cluster.linked_to is None:
                    continue
                source_node = (floor_id, cluster_id)
                target_node = cluster.linked_to
                pair = (min(source_node, target_node), max(source_node, target_node))
                if pair in seen_links:
                    continue
                seen_links.add(pair)
                stair_links.append([list(source_node), list(target_node)])

        return {
            "cluster_version": CLUSTERING_VERSION,
            "same_height_for_all": getattr(self, "same_height_for_all", True),
            "floors": floors,
            "stair_links": stair_links,
        }

    def from_dict(self, data):
        self.floors = {}
        self.same_height_for_all = data.get("same_height_for_all", True)
        self.pending_stair_endpoint = None
        self.stair_link_error = None
        self.stair_link_notice = None
        self.stair_link_notice_time = 0.0
        self.changed_floor_ids = set()
        self.navigation_graph = None
        self.drone_ids = {}

        for floor_data in data["floors"]:
            floor_id = floor_data["floor_id"]
            width = floor_data["width"]
            height = floor_data["height"]
            cell_size_m = floor_data.get("cell_size_m", 0.6)
            floor_height_m = floor_data.get("floor_height_m", 3.0)
            map_grid = Map_Grid(width, height, CellType.EMPTY)
            map_grid.cell_size_m = cell_size_m
            for y, row in enumerate(floor_data["cells"]):
                for x, cell_value in enumerate(row):
                    map_grid.set(x, y, CellType(cell_value))
            floor = Floor(floor_id, map_grid, floor_height_m=floor_height_m)
            floor.cell_size_m = cell_size_m
            self.floors[floor_id] = floor

        if self.floors:
            self.active_floor_id = 0

        self.rebuild_map_data()

        legacy_stair_nodes = None
        if data.get("cluster_version", 1) < CLUSTERING_VERSION and data.get("stair_links"):
            legacy_stair_nodes = {}
            for floor_id, floor in self.floors.items():
                legacy = ClusterGrid(floor.map_grid, include_doors=False)
                for cluster_id, cluster in enumerate(legacy.clusters):
                    if cluster.type is ClusterType.STAIRS:
                        legacy_stair_nodes[(floor_id, cluster_id)] = self.stair_cluster_at(
                            floor_id, *cluster.cells[0]
                        )

        for source_list, target_list in data.get("stair_links", []):
            source_node = tuple(source_list)
            target_node = tuple(target_list)
            if legacy_stair_nodes is not None:
                source_node = legacy_stair_nodes.get(source_node)
                target_node = legacy_stair_nodes.get(target_node)
                if source_node is None or target_node is None:
                    continue
            source_cluster = self._get_stair_cluster(source_node)
            target_cluster = self._get_stair_cluster(target_node)
            if source_cluster is not None and target_cluster is not None:
                source_cluster.linked_to = target_node
                target_cluster.linked_to = source_node

        self.rebuild_navigation()

    @property
    def active_floor(self):
        return self.floors[self.active_floor_id]

    @property
    def map_grid(self):
        return self.active_floor.map_grid

    @property
    def wall_and_exit_grid(self):
        return self.active_floor.wall_and_exit_grid

    @property
    def evacuee_occupancy_grid(self):
        return self.active_floor.evacuee_occupancy_grid

    @property
    def drone_occupancy_grid(self):
        return self.active_floor.drone_occupancy_grid

    @property
    def simulation_grid(self):
        return self.active_floor.simulation_grid

    @property
    def temporary_wall_grid(self):
        return self.active_floor.temporary_wall_grid

    @property
    def cluster_grid(self):
        return self.active_floor.cluster_grid

    @property
    def distance_map(self):
        return self.active_floor.distance_map

    @property
    def exits(self):
        return self.map_grid.exits

    def set_active_floor(self, floor_id):
        if floor_id not in self.floors:
            return False
        previous_floor_id = self.active_floor_id
        self.active_floor_id = floor_id
        if floor_id != previous_floor_id:
            logger.debug(
                "Active floor switched: %d -> %d",
                previous_floor_id + 1,
                floor_id + 1,
            )
        return True

    def rebuild_map_data(self, floor_ids=None):
        floors_to_rebuild = set(self.floors if floor_ids is None else floor_ids)
        stair_links = self._snapshot_stair_links()
        old_stair_clusters = self._snapshot_stair_clusters(floors_to_rebuild)
        pending_stair_endpoint = self.pending_stair_endpoint

        for floor_id in floors_to_rebuild:
            self.floors[floor_id].rebuild_map_data()

        stair_successors = self._restore_stair_links(
            stair_links, old_stair_clusters, floors_to_rebuild
        )
        if pending_stair_endpoint is not None:
            if pending_stair_endpoint[0] in floors_to_rebuild:
                self.pending_stair_endpoint = stair_successors.get(
                    pending_stair_endpoint
                )
            else:
                self.pending_stair_endpoint = pending_stair_endpoint
        self.navigation_graph = NavigationGraphx(self.floors)
        self.build_distance_maps()
        self.build_drone_assignments()
        self.changed_floor_ids.difference_update(floors_to_rebuild)

    def _snapshot_stair_clusters(self, floor_ids):
        return {
            floor_id: [
                (cluster_id, set(cluster.cells))
                for cluster_id, cluster in enumerate(
                    self.floors[floor_id].cluster_grid.clusters
                )
                if isinstance(cluster, StairCluster)
            ]
            for floor_id in floor_ids
        }

    def _snapshot_stair_links(self):
        links = []
        for floor_id, floor in self.floors.items():
            for cluster_id, cluster in enumerate(floor.cluster_grid.clusters):
                if not isinstance(cluster, StairCluster) or cluster.linked_to is None:
                    continue

                source_node = (floor_id, cluster_id)
                if source_node >= cluster.linked_to:
                    continue

                target_floor_id, target_cluster_id = cluster.linked_to
                target_floor = self.floors.get(target_floor_id)
                if target_floor is None or not (
                    0 <= target_cluster_id < len(target_floor.cluster_grid.clusters)
                ):
                    continue

                target_cluster = target_floor.cluster_grid.clusters[target_cluster_id]
                if not isinstance(target_cluster, StairCluster):
                    continue

                links.append((source_node, cluster.linked_to))
        return links

    def _restore_stair_links(self, old_links, old_stair_clusters, floors_to_rebuild):
        stair_successors = {}
        for floor_id, old_clusters in old_stair_clusters.items():
            candidate_lists = {}
            new_clusters = self.floors[floor_id].cluster_grid.clusters
            for old_cluster_id, old_cells in old_clusters:
                candidate_lists[old_cluster_id] = [
                    new_cluster_id
                    for new_cluster_id, new_cluster in enumerate(new_clusters)
                    if isinstance(new_cluster, StairCluster)
                    and old_cells.intersection(new_cluster.cells)
                ]

            assignments = self._assign_stair_successors(candidate_lists)
            stair_successors.update(
                {
                    (floor_id, old_cluster_id): (floor_id, new_cluster_id)
                    for old_cluster_id, new_cluster_id in assignments.items()
                }
            )

        for old_first_node, old_second_node in old_links:
            first_node = self._restored_stair_node(
                old_first_node, stair_successors, floors_to_rebuild
            )
            second_node = self._restored_stair_node(
                old_second_node, stair_successors, floors_to_rebuild
            )
            first_cluster = (
                self._get_stair_cluster(first_node) if first_node is not None else None
            )
            second_cluster = (
                self._get_stair_cluster(second_node) if second_node is not None else None
            )

            if first_cluster is None or second_cluster is None:
                if (
                    first_cluster is not None
                    and first_cluster.linked_to == old_second_node
                ):
                    first_cluster.linked_to = None
                if (
                    second_cluster is not None
                    and second_cluster.linked_to == old_first_node
                ):
                    second_cluster.linked_to = None
                continue
            if first_node[0] == second_node[0]:
                continue
            link_changed = (
                first_cluster.linked_to != second_node
                or second_cluster.linked_to != first_node
            )
            first_cluster.linked_to = second_node
            second_cluster.linked_to = first_node
            if link_changed:
                self._log_stair_link(first_node, second_node)
        return stair_successors

    def _restored_stair_node(self, old_node, stair_successors, floors_to_rebuild):
        if old_node[0] in floors_to_rebuild:
            return stair_successors.get(old_node)
        return old_node

    def _assign_stair_successors(self, candidate_lists):
        assignments = {}
        claimants = {}

        for old_cluster_id, candidates in candidate_lists.items():
            if len(candidates) != 1:
                continue
            self._claim_stair_successor(
                old_cluster_id,
                candidate_lists,
                assignments,
                claimants,
                set(),
                set(),
            )

        for old_cluster_id, candidates in candidate_lists.items():
            if len(candidates) <= 1 or old_cluster_id in assignments:
                continue
            self._claim_stair_successor(
                old_cluster_id,
                candidate_lists,
                assignments,
                claimants,
                set(),
                set(),
            )

        return assignments

    def _claim_stair_successor(
        self,
        old_cluster_id,
        candidate_lists,
        assignments,
        claimants,
        visited_clusters,
        visited_candidates,
    ):
        if old_cluster_id in visited_clusters:
            return False
        visited_clusters.add(old_cluster_id)
        candidates = candidate_lists[old_cluster_id]
        current_candidate = assignments.get(old_cluster_id)

        for candidate_id in candidates:
            if candidate_id == current_candidate or candidate_id in visited_candidates:
                continue
            if candidate_id not in claimants:
                self._set_stair_successor(
                    old_cluster_id, candidate_id, assignments, claimants
                )
                return True

        candidate_index = 0
        while candidate_index < len(candidates):
            candidate_id = candidates[candidate_index]
            if candidate_id == current_candidate or candidate_id in visited_candidates:
                candidate_index += 1
                continue

            visited_candidates.add(candidate_id)
            claimant_id = claimants[candidate_id]
            if self._claim_stair_successor(
                claimant_id,
                candidate_lists,
                assignments,
                claimants,
                visited_clusters,
                visited_candidates,
            ):
                self._set_stair_successor(
                    old_cluster_id, candidate_id, assignments, claimants
                )
                return True
            candidates.pop(candidate_index)
        return False

    @staticmethod
    def _set_stair_successor(old_cluster_id, candidate_id, assignments, claimants):
        previous_candidate_id = assignments.get(old_cluster_id)
        if previous_candidate_id is not None:
            del claimants[previous_candidate_id]
        assignments[old_cluster_id] = candidate_id
        claimants[candidate_id] = old_cluster_id

    def _get_stair_cluster(self, node):
        floor_id, cluster_id = node
        floor = self.floors.get(floor_id)
        if floor is None or not (0 <= cluster_id < len(floor.cluster_grid.clusters)):
            return None
        cluster = floor.cluster_grid.clusters[cluster_id]
        return cluster if isinstance(cluster, StairCluster) else None

    @staticmethod
    def _log_stair_link(first_node, second_node):
        logger.info(
            "Stair clusters linked: floor %d cluster %d <-> floor %d cluster %d",
            first_node[0],
            first_node[1],
            second_node[0],
            second_node[1],
        )

    def stair_cluster_at(self, floor_id, x, y):
        floor = self.floors[floor_id]
        if not floor.cluster_grid.in_bounds(x, y):
            return None
        cluster_id = floor.cluster_grid.get(x, y)
        if cluster_id is None or cluster_id < 0:
            return None
        cluster = floor.cluster_grid.clusters[cluster_id]
        if not isinstance(cluster, StairCluster):
            return None
        return floor_id, cluster_id

    def stair_arrival(self, entity, entity_type, reserved=()):
        floor_id = getattr(entity, "floor_id", 0)
        source_node = self.stair_cluster_at(floor_id, entity.grid_x, entity.grid_y)
        if source_node is None:
            return None

        source_cluster = self._get_stair_cluster(source_node)
        if source_cluster.linked_to is None:
            return None

        target_node = source_cluster.linked_to
        target_cluster = self._get_stair_cluster(target_node)
        if target_cluster is None:
            return None

        target_floor_id = target_node[0]
        occupancy = self._occupancy_grid(target_floor_id, entity_type)
        distances = self.floors[target_floor_id].distance_map
        reserved = set(reserved)

        candidates = sorted(
            target_cluster.cells,
            key=lambda cell: (
                occupancy.get(*cell) is not None,
                distances[cell[1]][cell[0]] if distances is not None else float("inf"),
                abs(cell[0] - entity.grid_x) + abs(cell[1] - entity.grid_y),
                cell[1],
                cell[0],
            ),
        )
        for x, y in candidates:
            if (target_floor_id, x, y) in reserved:
                continue
            return target_floor_id, x, y
        return None

    def connect_stair_clusters(self, first_node, second_node):
        if first_node[0] == second_node[0]:
            raise ValueError("Stair clusters must be on different floors")

        first_cluster = self._get_stair_cluster(first_node)
        second_cluster = self._get_stair_cluster(second_node)
        if first_cluster is None or second_cluster is None:
            raise ValueError("Both endpoints must be stair clusters")
        if first_cluster.linked_to not in (None, second_node):
            raise ValueError("The first stair cluster is already linked")
        if second_cluster.linked_to not in (None, first_node):
            raise ValueError("The second stair cluster is already linked")

        link_changed = (
            first_cluster.linked_to != second_node
            or second_cluster.linked_to != first_node
        )
        first_cluster.linked_to = second_node
        second_cluster.linked_to = first_node
        if link_changed:
            self._log_stair_link(first_node, second_node)
        self.rebuild_navigation()
        return True

    def disconnect_stair_cluster(self, node):
        cluster = self._get_stair_cluster(node)
        if cluster is None or cluster.linked_to is None:
            return False
        partner_node = cluster.linked_to
        partner_cluster = self._get_stair_cluster(partner_node)
        cluster.linked_to = None
        if partner_cluster is not None and partner_cluster.linked_to == node:
            partner_cluster.linked_to = None
        self.rebuild_navigation()
        return True

    def get_stair_clusters(self, floor_id):
        floor = self.floors.get(floor_id)
        if not floor or not hasattr(floor, "cluster_grid") or not floor.cluster_grid:
            return []
        return [
            (cid, c) for cid, c in enumerate(floor.cluster_grid.clusters)
            if c.type is ClusterType.STAIRS and c.cells
        ]

    def get_stair_name(self, floor_id, cluster_id):
        clusters = self.get_stair_clusters(floor_id)
        for idx, (cid, _) in enumerate(clusters, start=1):
            if cid == cluster_id:
                return f"Stairs {idx}"
        return f"Stairs C{cluster_id}"

    def get_stair_badge(self, floor_id, cluster_id):
        clusters = self.get_stair_clusters(floor_id)
        for idx, (cid, _) in enumerate(clusters, start=1):
            if cid == cluster_id:
                return f"S{idx}"
        return f"S{cluster_id}"

    def get_exit_clusters(self, floor_id):
        floor = self.floors.get(floor_id)
        if not floor or not hasattr(floor, "cluster_grid") or not floor.cluster_grid:
            return []
        return [
            (cid, c) for cid, c in enumerate(floor.cluster_grid.clusters)
            if c.type is ClusterType.EXIT and c.cells
        ]

    def get_exit_name(self, floor_id, cluster_id):
        clusters = self.get_exit_clusters(floor_id)
        for idx, (cid, _) in enumerate(clusters, start=1):
            if cid == cluster_id:
                return f"Exit {idx} (F{floor_id + 1})"
        return f"Exit (F{floor_id + 1})"

    def get_exit_badge(self, floor_id, cluster_id):
        clusters = self.get_exit_clusters(floor_id)
        for idx, (cid, _) in enumerate(clusters, start=1):
            if cid == cluster_id:
                return f"E{idx}"
        return "E"

    def rebuild_navigation(self):
        self.navigation_graph = NavigationGraphx(self.floors)
        self.build_distance_maps()
        self.build_drone_assignments()

    def route_to_closest_exit(self, start_node):
        exit_nodes = [
            (floor_id, cluster_id)
            for floor_id, floor in self.floors.items()
            for cluster_id, cluster in enumerate(floor.cluster_grid.clusters)
            if cluster.type is ClusterType.EXIT
        ]
        _, route = self.navigation_graph.distance_to_closest_source(
            exit_nodes, start_node
        )
        return route

    def plan_drone_routes(self, drones):
        routes = {}
        for drone_index, drone in enumerate(drones, start=1):
            floor_id = getattr(drone, "floor_id", 0)
            cluster_id = self.floors[floor_id].cluster_grid.get(
                drone.grid_x, drone.grid_y
            )
            route = self.route_to_closest_exit((floor_id, cluster_id))
            routes[drone] = route
            if route:
                logger.info(
                    "Found route for drone %d at floor %d, position (%d, %d): "
                    "%s (%d clusters)",
                    drone_index,
                    floor_id + 1,
                    drone.grid_x,
                    drone.grid_y,
                    route,
                    len(route),
                )
            else:
                logger.warning(
                    "No route found for drone %d at floor %d, position (%d, %d)",
                    drone_index,
                    floor_id + 1,
                    drone.grid_x,
                    drone.grid_y,
                )
        return routes

    def build_drone_assignments(self):
        drone_positions = [
            (floor_id, x, y)
            for floor_id, floor in self.floors.items()
            for y in range(floor.map_grid.height)
            for x in range(floor.map_grid.width)
            if floor.map_grid.get(x, y) == CellType.DRONE
        ]
        self.drone_ids, assignment_grids = calculate_drone_assignments(
            self, drone_positions
        )
        for floor_id, floor in self.floors.items():
            floor.drone_assignment_grid = assignment_grids[floor_id]

    def build_distance_maps(self):
        distances = {}
        pending = deque()

        for floor_id, floor in self.floors.items():
            distances[floor_id] = [
                [float("inf") for _ in range(floor.map_grid.width)]
                for _ in range(floor.map_grid.height)
            ]
            for exit_cells in floor.map_grid.exits:
                for x, y in exit_cells:
                    distances[floor_id][y][x] = 0
                    pending.append((floor_id, x, y))

        while pending:
            floor_id, x, y = pending.popleft()
            current_distance = distances[floor_id][y][x]
            floor = self.floors[floor_id]

            for delta_x, delta_y in ACTION_DIRECTIONS.values():
                if delta_x == 0 and delta_y == 0:
                    continue
                next_x = x + delta_x
                next_y = y + delta_y
                if not self.can_move_to(next_x, next_y, floor_id):
                    continue
                if distances[floor_id][next_y][next_x] <= current_distance + 1:
                    continue
                distances[floor_id][next_y][next_x] = current_distance + 1
                pending.append((floor_id, next_x, next_y))

            cluster_id = floor.cluster_grid.get(x, y)
            if cluster_id is None or cluster_id < 0:
                continue
            cluster = floor.cluster_grid.clusters[cluster_id]
            if not isinstance(cluster, StairCluster) or cluster.linked_to is None:
                continue

            target_floor_id, target_cluster_id = cluster.linked_to
            target_cluster = self._get_stair_cluster((target_floor_id, target_cluster_id))
            if target_cluster is None:
                continue
            for target_x, target_y in target_cluster.cells:
                if distances[target_floor_id][target_y][target_x] <= current_distance + 1:
                    continue
                distances[target_floor_id][target_y][target_x] = current_distance + 1
                pending.append((target_floor_id, target_x, target_y))

        for floor_id, floor in self.floors.items():
            floor.distance_map = distances[floor_id]

    def is_exit(self, x, y, floor_id=None):
        floor = self._get_floor(floor_id)
        return floor.wall_and_exit_grid.get(x, y) == CellType.EXIT

    def can_move_to(self, x, y, floor_id=None):
        floor = self._get_floor(floor_id)
        return (
            floor.map_grid.in_bounds(x, y)
            and floor.wall_and_exit_grid.get(x, y) != CellType.WALL
            and not floor.temporary_wall_grid.get(x, y)
        )

    def set_temporary_wall(self, floor_id, x, y, enabled):
        floor = self.floors[floor_id]
        if not floor.map_grid.in_bounds(x, y):
            return False
        if enabled and (
            floor.map_grid.get(x, y) != CellType.EMPTY
            or floor.evacuee_occupancy_grid.get(x, y) is not None
            or floor.drone_occupancy_grid.get(x, y) is not None
        ):
            return False
        if floor.temporary_wall_grid.get(x, y) == enabled:
            return False

        floor.temporary_wall_grid.set(x, y, enabled)
        self.build_distance_maps()
        return True

    def clear_temporary_walls(self):
        changed = False
        for floor in self.floors.values():
            changed = changed or any(
                any(row) for row in floor.temporary_wall_grid.cells
            )
            floor.temporary_wall_grid = Grid(
                floor.map_grid.width, floor.map_grid.height, False
            )
        if changed:
            self.build_distance_maps()

    def _get_floor(self, floor_id=None):
        return self.floors[self.active_floor_id if floor_id is None else floor_id]

    def set_map_cell(self, floor_id, x, y, cell_type):
        floor = self.floors[floor_id]
        if floor.map_grid.get(x, y) == cell_type:
            return False
        floor.map_grid.set(x, y, cell_type)
        self.changed_floor_ids.add(floor_id)
        return True

    def rebuild_changed_map_data(self):
        if not self.changed_floor_ids:
            return False
        self.rebuild_map_data(self.changed_floor_ids)
        return True

    def set_floor_height(self, floor_id, height_m, same_for_all=None):
        val = max(1.0, min(20.0, float(height_m)))
        if same_for_all is None:
            same_for_all = getattr(self, "same_height_for_all", True)
        self.same_height_for_all = bool(same_for_all)
        if same_for_all:
            for f in self.floors.values():
                f.floor_height_m = val
        elif floor_id in self.floors:
            self.floors[floor_id].floor_height_m = val

    def clear_people_and_drones(self, floor_id, simulation=None):
        if floor_id not in self.floors:
            return
        floor = self.floors[floor_id]
        for y in range(floor.map_grid.height):
            for x in range(floor.map_grid.width):
                if floor.map_grid.get(x, y) in (CellType.EVACUEE, CellType.DRONE):
                    floor.map_grid.set(x, y, CellType.EMPTY)
        floor.evacuee_occupancy_grid = Grid(floor.map_grid.width, floor.map_grid.height, None)
        floor.drone_occupancy_grid = Grid(floor.map_grid.width, floor.map_grid.height, None)
        if simulation is not None:
            simulation.evacuees[:] = [e for e in simulation.evacuees if getattr(e, "floor_id", 0) != floor_id]
            simulation.drones[:] = [d for d in simulation.drones if getattr(d, "floor_id", 0) != floor_id]
        self.changed_floor_ids.add(floor_id)
        self.rebuild_changed_map_data()
        if simulation is not None:
            self.rebuild_simulation_grid(simulation.evacuees, simulation.drones)

    def rebuild_simulation_grid(self, evacuees=(), drones=()):
        for floor in self.floors.values():
            floor.simulation_grid = floor.wall_and_exit_grid.copy()

        for evacuee in evacuees:
            if getattr(evacuee, "in_stair_transit", False):
                continue
            floor = self.floors[getattr(evacuee, "floor_id", 0)]
            floor.simulation_grid.set(evacuee.grid_x, evacuee.grid_y, CellType.EVACUEE)

        for drone in drones:
            if getattr(drone, "in_stair_transit", False):
                continue
            floor = self.floors[getattr(drone, "floor_id", 0)]
            floor.simulation_grid.set(drone.grid_x, drone.grid_y, CellType.DRONE)

    def add_evacuee(self, evacuee):
        floor = self.floors[getattr(evacuee, "floor_id", 0)]
        floor.evacuee_occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, evacuee)

    def remove_evacuee(self, evacuee):
        floor = self.floors[getattr(evacuee, "floor_id", 0)]
        floor.evacuee_occupancy_grid.set(evacuee.grid_x, evacuee.grid_y, None)

    def add_drone(self, drone):
        floor = self.floors[getattr(drone, "floor_id", 0)]
        floor.drone_occupancy_grid.set(drone.grid_x, drone.grid_y, drone)

    def remove_drone(self, drone):
        floor = self.floors[getattr(drone, "floor_id", 0)]
        floor.drone_occupancy_grid.set(drone.grid_x, drone.grid_y, None)

    def get_desired_moves(self, entities, actions, entity_type):
        desired_moves = {}
        destinations = set()

        for entity, action in zip(entities, actions):
            if getattr(entity, "in_stair_transit", False):
                continue

            if action is Action.USE_STAIRS:
                floor_id = getattr(entity, "floor_id", 0)
                source_node = self.stair_cluster_at(floor_id, entity.grid_x, entity.grid_y)
                if source_node is not None:
                    source_cluster = self._get_stair_cluster(source_node)
                    if source_cluster is not None and source_cluster.linked_to is not None:
                        target_floor_id = source_cluster.linked_to[0]
                        low_f = min(floor_id, target_floor_id)
                        high_f = max(floor_id, target_floor_id)
                        if high_f > low_f:
                            h = sum(getattr(self.floors.get(f), "floor_height_m", 3.0) for f in range(low_f + 1, high_f + 1))
                        else:
                            h = 0.0
                        delay = max(2, int(round(h * 1.5))) if entity_type == "drone" else max(3, int(round(h * 3.0)))
                        entity.in_stair_transit = True
                        entity.stair_steps_left = delay
                        entity.stair_target_node = source_cluster.linked_to
                        self._occupancy_grid(floor_id, entity_type).set(entity.grid_x, entity.grid_y, None)
                continue

            delta_x, delta_y = ACTION_DIRECTIONS[action]
            floor_id = getattr(entity, "floor_id", 0)
            x = entity.grid_x + delta_x
            y = entity.grid_y + delta_y

            if not self.can_move_to(x, y, floor_id):
                continue

            destination = (floor_id, x, y)
            if destination in destinations:
                continue

            destinations.add(destination)
            desired_moves[entity] = destination

        return desired_moves

    def _occupancy_grid(self, floor_id, entity_type):
        floor = self.floors[floor_id]
        if entity_type == "drone":
            return floor.drone_occupancy_grid
        return floor.evacuee_occupancy_grid

    def can_complete_move(self, entity, root, desired_moves, visited, entity_type):
        destination = desired_moves.get(entity)
        if destination is None:
            return False

        floor_id, x, y = destination
        occupant = self._occupancy_grid(floor_id, entity_type).get(x, y)
        if occupant is None or occupant is root:
            return True

        if occupant in visited:
            return False

        visited.add(entity)
        return self.can_complete_move(
            occupant, root, desired_moves, visited, entity_type
        )

    def resolve_moves(self, entities, desired_moves, entity_type):
        moves = {}
        for entity in entities:
            if entity in desired_moves and self.can_complete_move(
                entity, entity, desired_moves, set(), entity_type
            ):
                moves[entity] = desired_moves[entity]
        return moves

    def apply_moves(self, moves, entity_type):
        for entity in moves:
            self._occupancy_grid(
                getattr(entity, "floor_id", 0), entity_type
            ).set(entity.grid_x, entity.grid_y, None)

        for entity, (floor_id, x, y) in moves.items():
            entity.floor_id = floor_id
            entity.grid_x = x
            entity.grid_y = y
            self._occupancy_grid(floor_id, entity_type).set(x, y, entity)

    def step_evacuees(self, evacuees, actions):
        return self._step_entities(evacuees, actions, "evacuee")

    def step_drones(self, drones, actions):
        return self._step_entities(drones, actions, "drone")

    def _step_entities(self, entities, actions, entity_type):
        for entity in entities:
            if getattr(entity, "in_stair_transit", False):
                entity.stair_steps_left -= 1
                if entity.stair_steps_left <= 0:
                    target_node = getattr(entity, "stair_target_node", None)
                    target_cluster = self._get_stair_cluster(target_node) if target_node else None
                    if target_cluster is not None:
                        target_floor_id = target_node[0]
                        occupancy = self._occupancy_grid(target_floor_id, entity_type)
                        distances = self.floors[target_floor_id].distance_map
                        candidates = sorted(
                            target_cluster.cells,
                            key=lambda c: (
                                distances[c[1]][c[0]] if distances is not None else float("inf"),
                                abs(c[0] - entity.grid_x) + abs(c[1] - entity.grid_y),
                            )
                        )
                        landed = False
                        for cx, cy in candidates:
                            if occupancy.get(cx, cy) is None:
                                entity.floor_id = target_floor_id
                                entity.grid_x = cx
                                entity.grid_y = cy
                                entity.in_stair_transit = False
                                entity.stair_steps_left = 0
                                entity.stair_target_node = None
                                occupancy.set(cx, cy, entity)
                                landed = True
                                break
                        if not landed:
                            entity.stair_steps_left = 1
                    else:
                        entity.in_stair_transit = False

        desired_moves = self.get_desired_moves(entities, actions, entity_type)
        if entity_type == "evacuee" and self.door_flow is not None:
            moves = self.door_flow.limit(
                entities,
                desired_moves,
                lambda granted: self.resolve_moves(entities, granted, entity_type),
            )
        else:
            moves = self.resolve_moves(entities, desired_moves, entity_type)
        self.apply_moves(moves, entity_type)
        if entity_type == "evacuee" and self.door_flow is not None:
            self.door_flow.record_moves(moves)
        return moves