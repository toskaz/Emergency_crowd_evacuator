import networkx as nx
import math

# dla uproszczenia implementacji 
# source_nodes to sa wyjscia
#target_node to "dron"
def find_path_multi_target_dijkstra(G, source_nodes, target_node):
    
    if not source_nodes or target_node is None:
        return math.inf, []
    
    try:
        distance, path = nx.multi_source_dijkstra(
            G, 
            sources=source_nodes, 
            target=target_node, 
            #weight='weight' #TODO : implementacja wagi
        )
        
        return float(distance), path[::-1]
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf, []
