import math
import networkx as nx

from graph_approach.multi_target_dijkstra import find_path_multi_target_dijkstra
from graph_approach.multi_target_bfs import find_path_multi_target_bfs
from graph_approach.multi_target_astar import find_path_multi_target_astar
from graph_approach.multi_target_bidirectional_dijkstra import find_path_multi_target_bidirectional_dijkstra
from grids.cluster import ClusterType


class NavigationGraphx:
    def __init__(self, floors):
        self.floors = floors
        self.G = nx.Graph()
        self.algorithm = "Dijkstra"
        self.rebuild_base_graph()

    def rebuild_base_graph(self):
        self.G.clear()
        for floor_id, floor in self.floors.items():
            cluster_grid = floor.cluster_grid
            self.G.add_nodes_from(
                (floor_id, cluster_id)
                for cluster_id in range(len(cluster_grid.clusters))
            )
            self.G.add_edges_from(
                ((floor_id, first), (floor_id, second))
                for first, second in cluster_grid.adjacent_cluster_pairs
            )
            for cluster_id, cluster in enumerate(cluster_grid.clusters):
                if cluster.type is ClusterType.STAIRS and cluster.linked_to is not None:
                    self.G.add_edge(
                        (floor_id, cluster_id), cluster.linked_to, weight=1
                    )

    def distance_to_closest_source(self, source_nodes, target_node, algorithm=None):
        if target_node not in self.G:
            return math.inf, []

        source_nodes = {node for node in source_nodes if node in self.G}
        if not source_nodes:
            return math.inf, []

        algo = algorithm or getattr(self, "algorithm", "Dijkstra")
        if algo in ("A*", "A* (A-Star)"):
            return find_path_multi_target_astar(self.G, source_nodes, target_node, self.floors)
        elif algo == "Bidirectional Dijkstra":
            return find_path_multi_target_bidirectional_dijkstra(self.G, source_nodes, target_node)
        elif algo == "BFS":
            return find_path_multi_target_bfs(self.G, source_nodes, target_node)
        return find_path_multi_target_dijkstra(self.G, source_nodes, target_node)
