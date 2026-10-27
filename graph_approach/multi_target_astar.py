import math
import networkx as nx
from utils.config import cluster_size


def find_path_multi_target_astar(G, source_nodes, target_node, cluster_grid=None):
    if not source_nodes or target_node is None:
        return math.inf, []

    best_path = []
    best_distance = math.inf

    for exit_node in source_nodes:
        if exit_node not in G:
            continue
        try:
            path = nx.astar_path(
                G, 
                source=target_node, 
                target=exit_node,
                heuristic=lambda u, v: scaled_euclidean_heuristic(u, v, cluster_grid),
                # weight='weight' # TODO: dodac wagi krawedzi (przepustowosc i zagrozenia)
            )

            distance = float(len(path) - 1)
            if distance < best_distance:
                best_distance, best_path = distance, path

        except nx.NetworkXNoPath:
            continue

    print("Multitarget astar")
    return best_distance, best_path


def scaled_euclidean_heuristic(u, v, cluster_grid):
    if cluster_grid is not None:
        pos_u = cluster_grid.clusters[u].get_centroid()
        pos_v = cluster_grid.clusters[v].get_centroid()
        raw_dist = math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])

        cluster_step = max(2 * cluster_size + 1, 1)
        return raw_dist / cluster_step

    return 0.0


    

    

    