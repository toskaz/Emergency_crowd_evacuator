from collections import deque
import logging

from controls.action import Action
from graph_approach.navigation_graphx import NavigationGraphx
from grids.cluster import ClusterType, StairCluster
from grids.floor import Floor
from grids.grid import Grid
from grids.map_grid import Map_Grid
from map.cell import CellType
from utils.config import image_path


logger = logging.getLogger(__name__)


ACTION_DIRECTIONS = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


class Grid_World:
    FLOOR_COUNT = 2

    def __init__(self):
        source_grid = Map_Grid.from_image(image_path)
        self.floors = {}
        for floor_id in range(self.FLOOR_COUNT):
            map_grid = Map_Grid(source_grid.width, source_grid.height)
            map_grid.cells = [row.copy() for row in source_grid.cells]
            self.floors[floor_id] = Floor(floor_id, map_grid)

        self.active_floor_id = 0
        self.pending_stair_endpoint = None
        self.stair_link_error = None
        self.map_changed = False
        self.changed_floor_id = None
        self.navigation_graph = None
        self.rebuild_map_data()

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
        reserved = set(reserved) #occupancy grid is not enough here
        candidates = sorted(
            target_cluster.cells,
            key=lambda cell: (
                occupancy.get(*cell) is not None,
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

    def rebuild_navigation(self):
        self.navigation_graph = NavigationGraphx(self.floors)
        self.build_distance_maps()

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
        )

    def _get_floor(self, floor_id=None):
        return self.floors[self.active_floor_id if floor_id is None else floor_id]

    def set_map_cell(self, floor_id, x, y, cell_type):
        floor = self.floors[floor_id]
        if floor.map_grid.get(x, y) == cell_type:
            return False
        floor.map_grid.set(x, y, cell_type)
        self.map_changed = True
        self.changed_floor_id = floor_id
        return True

    def rebuild_changed_map_data(self):
        if not self.map_changed:
            return False
        self.rebuild_map_data({self.changed_floor_id})
        self.changed_floor_id = None
        self.map_changed = False
        return True

    def rebuild_simulation_grid(self, evacuees=(), drones=()):
        for floor in self.floors.values():
            floor.simulation_grid = floor.wall_and_exit_grid.copy()

        for evacuee in evacuees:
            floor = self.floors[getattr(evacuee, "floor_id", 0)]
            floor.simulation_grid.set(evacuee.grid_x, evacuee.grid_y, CellType.EVACUEE)

        for drone in drones:
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
            if action is Action.USE_STAIRS:
                destination = self.stair_arrival(entity, entity_type, destinations)
                if destination is not None:
                    destinations.add(destination)
                    desired_moves[entity] = destination
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
        desired_moves = self.get_desired_moves(entities, actions, entity_type)
        moves = self.resolve_moves(entities, desired_moves, entity_type)
        self.apply_moves(moves, entity_type)
        return moves