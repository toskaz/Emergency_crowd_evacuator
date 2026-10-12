from multi_target_dijkstra import get_shortest_path_to_exits
import networkx as nx
from grid import Grid


class NavigationGraphx:
    def __init__(self, map_grid, block_size):
        self.height = map_grid.height
        self.width = map_grid.width
        self.block_size = block_size

        self.node_of = Grid(self.width, self.height, None)
        self.node_cells = []
        self.representatives = []

        self.edges = []

        #pojnar to zrobi
        #self.build_nodes(map_grid)
        #self.build_edges()
        
        self.G = nx.Graph()
        # u - poczatek drogi, v - koniec drogi
        #potem dodamy weight #TODO dodaj wagę
        for u, v in self.edges:
            self.G.add_edge(u, v)
        

    #funkcja node_at 
        
    def distance_to_closest_source(self, source_cells, target_cell):
        target_node = self.node_at(target_cell)

        source_nodes = set()
        for cell in source_cells:
            node = self.node_at(cell)

            if node is not None:
                source_nodes.add(node)
        
        return get_shortest_path_to_exits(self.G, source_nodes, target_node)
