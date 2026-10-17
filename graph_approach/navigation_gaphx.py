from graph_approach.multi_target_dijkstra import get_shortest_path_to_exits
import networkx as nx


class NavigationGraphx:
    def __init__(self, cluster_grid):

        self.cluster_grid = cluster_grid

        self.G = nx.Graph()
        self.G.add_nodes_from(self.cluster_grid.clusters)
        self.G.add_edges_from(self.cluster_grid.adjacent_cluster_pairs)
        
    def node_at(self, cell):
        x, y = cell

        if not self.cluster_grid.grid.in_bounds(x, y):
            return None

        node = self.cluster_grid.get(x, y)
        return node if node in self.G else None

    def distance_to_closest_source(self, source_cells, target_cell):
        target_node = self.node_at(target_cell)

        source_nodes = set()
        for cell in source_cells:
            node = self.node_at(cell)

            if node is not None:
                source_nodes.add(node)
        
        return get_shortest_path_to_exits(self.G, source_nodes, target_node)
