from graph_approach.multi_target_dijkstra import get_shortest_path_to_exits
import math
import networkx as nx


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
        return get_shortest_path_to_exits(self.G, source_nodes, target_node)
