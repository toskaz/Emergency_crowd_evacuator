import math
import networkx as nx

def find_path_multi_target_bidirectional_dijkstra(G, source_nodes, target_node):

    if not source_nodes or target_node is None or target_node not in G:
        return math.inf, []

    best_path = []
    best_distance = math.inf

    for exit_node in source_nodes:
        if exit_node not in G:
            continue
        try:
            distance, path = nx.bidirectional_dijkstra(
                G,
                source=target_node,
                target=exit_node,
                weight='weight', # TODO: trzeba dodac wagi jak bedzie Flow Rate
             )

            distance = float(distance)

            if distance < best_distance:
                best_distance, best_path = distance, path

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    return best_distance, best_path
