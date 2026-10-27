import math
import networkx as nx

from graph_approach.multi_target_dijkstra import find_path_multi_target_dijkstra
from graph_approach.multi_target_bfs import find_path_multi_target_bfs
from graph_approach.multi_target_astar import find_path_multi_target_astar
from graph_approach.multi_target_bidirectional_dijkstra import find_path_multi_target_bidirectional_dijkstra


class NavigationGraphx:
    def __init__(self, cluster_grid):
        self.cluster_grid = cluster_grid

        self.G = nx.Graph()
        self.G.add_nodes_from(range(len(self.cluster_grid.clusters)))
        self.G.add_edges_from(self.cluster_grid.adjacent_cluster_pairs)

    def distance_to_closest_source(self, source_nodes, target_node):
        if target_node not in self.G:
            return math.inf, []

        source_nodes = {node for node in source_nodes if node in self.G}
        if not source_nodes:
            return math.inf, []


        #return find_path_multi_target_dijkstra(self.G, source_nodes, target_node)
        #return find_path_multi_target_bfs(self.G, source_nodes, target_node)
        return find_path_multi_target_astar(self.G, source_nodes, target_node, self.cluster_grid)
        #return find_path_multi_target_bidirectional_dijkstra(self.G, source_nodes, target_node)
