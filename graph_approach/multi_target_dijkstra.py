import networkx as nx
import math

# dla uproszczenia implementacji 
# source_nodes to sa wyjscia
#target_node to "dron"
def get_shortest_path_to_exits(G, source_nodes, target_node):
    
    if not source_nodes or target_node is None:
        return math.inf, []
    
    try:
        distance, path = nx.multi_source_dijkstra(
            G, 
            source=source_nodes, 
            target=target_node, 
            #weight='weight' #TODO : implementacja wagi
        )
        
        return float(distance), path[::-1]
    except nx.NetworkXNoPath:
        return math.inf, []
