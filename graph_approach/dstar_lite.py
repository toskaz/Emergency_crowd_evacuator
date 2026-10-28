import math
import heapq
from utils.config import cluster_size


def scaled_euclidean_heuristic(u, v, cluster_grid):
    if cluster_grid is not None:
        pos_u = cluster_grid.clusters[u].get_centroid()
        pos_v = cluster_grid.clusters[v].get_centroid()
        raw_dist = math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])
        cluster_step = max(2 * cluster_size + 1, 1)
        return raw_dist / cluster_step
    return 0.0


class DStarLite:
    """
    Implementacja algorytmu D* Lite wedlug pseudokodu Koeniga i Likhacheva
    """

    def __init__(self, G, start, goal, cluster_grid=None):
        self.G = G
        self.s_start = start
        self.s_goal = goal
        self.cluster_grid = cluster_grid

        self.c = {}
        for u, v in self.G.edges():
            self.c[(u, v)] = 1.0
            self.c[(v, u)] = 1.0

        self.s_last = self.s_start
        self.initialize()
        self.compute_shortest_path()

    def calculate_key(self, s):
        min_g_rhs = min(self.g[s], self.rhs[s])
        k1 = min_g_rhs + scaled_euclidean_heuristic(self.s_start, s, self.cluster_grid) + self.km
        k2 = min_g_rhs
        return (k1, k2)

    def initialize(self):
        self.U = []
        self.km = 0.0
        self.g = {}
        self.rhs = {}
        for node in self.G.nodes():
            self.g[node] = math.inf
            self.rhs[node] = math.inf

        self.rhs[self.s_goal] = 0.0
        initial_key = self.calculate_key(self.s_goal)
        heapq.heappush(self.U, (initial_key, self.s_goal))

    def get_edge_cost(self, u, v):
        return self.c.get((u, v), math.inf)

    def update_vertex(self, u):
        if u != self.s_goal:
            neighbors = list(self.G.neighbors(u))
            if neighbors:
                self.rhs[u] = min(self.get_edge_cost(u, sprime) + self.g[sprime] for sprime in neighbors)
            else:
                self.rhs[u] = math.inf

        self.U = [entry for entry in self.U if entry[1] != u]
        heapq.heapify(self.U)


        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.U and (self.U[0][0] < self.calculate_key(self.s_start) or self.rhs[self.s_start] != self.g[self.s_start]):
            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u)

            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for pred in self.G.neighbors(u):
                    self.update_vertex(pred)
            else:
                self.g[u] = math.inf
                self.update_vertex(u)
                for pred in self.G.neighbors(u):
                    self.update_vertex(pred)

    def notify_hazard_changed(self, u, v, new_cost=math.inf):
        if self.s_last != self.s_start:
            self.km += scaled_euclidean_heuristic(self.s_last, self.s_start, self.cluster_grid)
            self.s_last = self.s_start

        self.c[(u, v)] = new_cost
        self.c[(v, u)] = new_cost

        self.update_vertex(u)
        self.update_vertex(v)

        self.compute_shortest_path()

    def update_drone_position(self, new_start):
        if new_start != self.s_start:
            self.s_start = new_start

    def get_path(self):
        if self.s_start == self.s_goal:
            return 0.0, [self.s_start]

        if self.g[self.s_start] == math.inf:
            return math.inf, []

        path = [self.s_start]
        curr = self.s_start
        visited = {curr}

        while curr != self.s_goal:
            neighbors = list(self.G.neighbors(curr))
            if not neighbors:
                return math.inf, []

            best_neighbor = min(neighbors, key=lambda n: self.get_edge_cost(curr, n) + self.g[n])
            
            if best_neighbor in visited or self.get_edge_cost(curr, best_neighbor) == math.inf:
                return math.inf, []

            curr = best_neighbor
            visited.add(curr)
            path.append(curr)

        distance = float(self.g[self.s_start])
        return distance, path
