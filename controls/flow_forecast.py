import math
from collections import Counter

import networkx as nx

from grids.cluster import ClusterType
from utils import config


HAZARD_WEIGHT = 1000.0
MIN_MOVE_RATE = 0.25
PACE_SMOOTHING = 0.1


def door_timeline(arrivals, capacity, free_at=0.0):
    timeline = []
    for group_id, arrival, size in sorted(arrivals, key=lambda item: item[1]):
        start = max(arrival, free_at)
        free_at = start + size / capacity
        timeline.append((group_id, arrival, start, free_at))
    return timeline


class FlowForecast:
    def __init__(self, world):
        self.world = world
        self.tick = 0
        self.move_rates = {}
        self.positions = {}
        self.group_sizes = Counter()
        self.reservations = {}

    @property
    def door_flow(self):
        return self.world.door_flow

    def observe(self, simulation):
        self.tick = simulation.step_count
        active = set(simulation.drones)
        for drone in simulation.drones:
            position = (drone.floor_id, drone.grid_x, drone.grid_y)
            previous = self.positions.get(drone)
            if previous is not None:
                moved = 1.0 if previous != position else 0.0
                rate = self.move_rates.get(drone, 1.0)
                self.move_rates[drone] = rate + PACE_SMOOTHING * (moved - rate)
            self.positions[drone] = position
        for drone in list(self.positions):
            if drone not in active:
                self.positions.pop(drone, None)
                self.move_rates.pop(drone, None)
        self.group_sizes = Counter(
            evacuee.assigned_drone_id
            for evacuee in simulation.evacuees
            if evacuee.assigned_drone_id is not None
        )

    def group_size(self, drone):
        return self.group_sizes.get(drone.drone_id, 0)

    def ticks_per_cell(self, drone):
        return 1.0 / max(MIN_MOVE_RATE, self.move_rates.get(drone, 1.0))

    def travel_cells(self, node_u, node_v):
        if node_u[0] != node_v[0]:
            return 1.0
        first = self.world.floors[node_u[0]].cluster_grid.clusters[node_u[1]]
        second = self.world.floors[node_v[0]].cluster_grid.clusters[node_v[1]]
        (ux, uy), (vx, vy) = first.get_centroid(), second.get_centroid()
        return max(1.0, abs(ux - vx) + abs(uy - vy))

    def doorway_arrivals(self, drone, route):
        if self.door_flow is None or not route:
            return []
        pace = self.ticks_per_cell(drone)
        elapsed = 0.0
        arrivals = []
        seen = set()
        for index, node in enumerate(route):
            if index > 0:
                elapsed += self.travel_cells(route[index - 1], node) * pace
            doorway = self.door_flow.doorway_of_node(node)
            if doorway is None or doorway.key in seen:
                continue
            seen.add(doorway.key)
            arrivals.append((doorway, self.tick + elapsed))
        return arrivals

    def rebuild(self, simulation, routes, starts=None):
        self.observe(simulation)
        self.reservations = {}
        if self.door_flow is None:
            return
        for drone in simulation.drones:
            size = self.group_size(drone)
            route = self.remaining_route(drone, routes.get(drone), starts)
            if size == 0 or not route:
                continue
            for doorway, arrival in self.doorway_arrivals(drone, route):
                self.reservations.setdefault(doorway.key, []).append((drone, arrival, size))

    def remaining_route(self, drone, route, starts=None):
        if not route:
            return []
        start = None if starts is None else starts.get(drone)
        if start is not None and start in route:
            return list(route[route.index(start):])
        return list(route)

    def free_at(self, doorway, arrival, exclude=None):
        capacity = doorway.effective_capacity()
        free = self.tick + doorway.queue_ticks if doorway.queue > 0 else self.tick
        others = [
            (other, other_arrival, size)
            for other, other_arrival, size in self.reservations.get(doorway.key, ())
            if other is not exclude and other_arrival <= arrival
        ]
        timeline = door_timeline(others, capacity, free)
        return timeline[-1][3] if timeline else free

    def wait_at(self, doorway, arrival, exclude=None):
        return max(0.0, self.free_at(doorway, arrival, exclude) - arrival)

    def pass_ticks(self, doorway, size):
        return size / doorway.effective_capacity()

    def finish_time(self, drone, route):
        if not route:
            return math.inf
        size = max(1, self.group_size(drone))
        pace = self.ticks_per_cell(drone)
        elapsed = 0.0
        bottleneck = 0.0
        seen = set()
        for index, node in enumerate(route):
            if index > 0:
                elapsed += self.travel_cells(route[index - 1], node) * pace
            doorway = None if self.door_flow is None else self.door_flow.doorway_of_node(node)
            if doorway is None or doorway.key in seen:
                continue
            seen.add(doorway.key)
            elapsed += self.wait_at(doorway, self.tick + elapsed, drone)
            bottleneck = max(bottleneck, self.pass_ticks(doorway, size))
        return self.tick + elapsed + bottleneck

    def apply_edge_weights(self, graph, drone, start_node):
        pace = 1.0 if drone is None else self.ticks_per_cell(drone)
        for node_u, node_v, data in graph.edges(data=True):
            if data.get("weight", 1.0) >= HAZARD_WEIGHT:
                continue
            data["weight"] = self.travel_cells(node_u, node_v) * pace
        if self.door_flow is None or drone is None or start_node not in graph:
            return
        size = max(1, self.group_size(drone))
        distances = nx.single_source_dijkstra_path_length(graph, start_node, weight="weight")
        costs = {}
        for doorway in self.door_flow.doorways:
            reach = [distances[node] for node in doorway.nodes if node in distances]
            if not reach:
                continue
            arrival = self.tick + min(reach)
            costs[doorway.key] = (
                self.wait_at(doorway, arrival, drone) + self.pass_ticks(doorway, size)
            )
        for node_u, node_v, data in graph.edges(data=True):
            if data["weight"] >= HAZARD_WEIGHT:
                continue
            first = self.door_flow.doorway_of_node(node_u)
            second = self.door_flow.doorway_of_node(node_v)
            if first is second:
                continue
            for doorway in (first, second):
                if doorway is None or doorway.key not in costs:
                    continue
                share = 1.0 if doorway.kind == "exit" else 0.5
                data["weight"] += share * costs[doorway.key]
            data["weight"] = min(data["weight"], HAZARD_WEIGHT - 1.0)


class FlowRerouter:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.last_reroute = {}
        self.reroutes = []

    @property
    def world(self):
        return self.coordinator.world

    def step(self, simulation, routes, drone_controller=None):
        world = self.world
        if not config.door_reroute_enabled or world.door_flow is None or not simulation.drones:
            return []
        if world.flow_forecast is None:
            world.flow_forecast = FlowForecast(world)
        forecast = world.flow_forecast
        forecast.observe(simulation)
        congested = [doorway.key for doorway in world.door_flow.newly_congested]
        periodic = simulation.step_count % config.flow_forecast_interval == 0
        if not periodic and not congested:
            return []

        self.coordinator.update_graph_passability()
        starts = {drone: self.coordinator.get_drone_cluster(drone) for drone in simulation.drones}
        forecast.rebuild(simulation, routes, starts)
        order = sorted(
            simulation.drones,
            key=lambda drone: min(
                (arrival for _, arrival in forecast.doorway_arrivals(
                    drone, forecast.remaining_route(drone, routes.get(drone), starts))),
                default=math.inf,
            ),
        )
        changed = []
        for drone in order:
            new_route = self.consider(simulation, drone, routes, starts.get(drone), congested, periodic)
            if new_route is None:
                continue
            routes[drone] = new_route
            self.coordinator.drone_routes[drone] = new_route
            self.last_reroute[drone] = simulation.step_count
            if drone_controller is not None:
                drone_controller.invalidate(drone)
            changed.append(drone)
            forecast.rebuild(simulation, routes, starts)
        return changed

    def consider(self, simulation, drone, routes, start, congested, periodic):
        forecast = self.world.flow_forecast
        if forecast.group_size(drone) == 0 or start is None:
            return None
        if self.world.door_flow.doorway_of_node(start) is not None:
            return None
        last = self.last_reroute.get(drone)
        if last is not None and simulation.step_count - last < config.min_ticks_between_reroutes:
            return None
        current = forecast.remaining_route(drone, routes.get(drone), {drone: start})
        if not current or current[0] != start:
            return None
        if not periodic and not self._passes_congested(current, congested):
            return None

        current_finish = forecast.finish_time(drone, current)
        best_route, best_finish = None, current_finish
        for candidate in self.coordinator.flow_routes(drone, start):
            if candidate == current:
                continue
            finish = forecast.finish_time(drone, candidate)
            if finish < best_finish:
                best_route, best_finish = candidate, finish
        if best_route is None or best_finish + config.reroute_min_gain >= current_finish:
            return None
        self.reroutes.append({
            "step": simulation.step_count,
            "drone": drone.drone_id,
            "from": current_finish,
            "to": best_finish,
            "exit": best_route[-1],
        })
        return best_route

    def _passes_congested(self, route, congested):
        door_flow = self.world.door_flow
        for node in route[2:]:
            doorway = door_flow.doorway_of_node(node)
            if doorway is not None and doorway.key in congested:
                return True
        return False


def exit_nodes(world):
    return [
        (floor_id, cluster_id)
        for floor_id, floor in world.floors.items()
        for cluster_id, cluster in enumerate(floor.cluster_grid.clusters)
        if cluster.type is ClusterType.EXIT
    ]
