from graph_approach.multi_target_dijkstra import get_shortest_path_to_exits
from grids.cluster import ClusterType
import math
import networkx as nx


class NavigationGraphx:
    def __init__(self, floors):
        self.floors = floors
        self.G = nx.Graph()

        for floor_id, floor in floors.items():
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

    def distance_to_closest_source(self, source_nodes, target_node):
        if target_node not in self.G:
            return math.inf, []
        source_nodes = {node for node in source_nodes if node in self.G}
        return get_shortest_path_to_exits(self.G, source_nodes, target_node)
