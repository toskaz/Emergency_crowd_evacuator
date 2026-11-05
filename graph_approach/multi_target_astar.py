import math
import networkx as nx
from grids.cluster import StairCluster
from utils.config import cluster_size


def find_path_multi_target_astar(G, source_nodes, target_node, floors=None):
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
                heuristic=lambda u, v: scaled_euclidean_heuristic(u, v, floors),
                # weight='weight' # TODO: dodac wagi krawedzi (przepustowosc i zagrozenia)
            )

            distance = float(len(path) - 1)
            if distance < best_distance:
                best_distance, best_path = distance, path

        except nx.NetworkXNoPath:
            continue

    return best_distance, best_path


def scaled_euclidean_heuristic(u, v, floors):
    if floors is None:
        return 0.0

    cluster_step = max(2 * cluster_size + 1, 1)
    floor_u, cluster_u = u
    floor_v, cluster_v = v

    pos_u = floors[floor_u].cluster_grid.clusters[cluster_u].get_centroid()
    pos_v = floors[floor_v].cluster_grid.clusters[cluster_v].get_centroid()

    if floor_u == floor_v:
        raw_dist = math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])
        return raw_dist / cluster_step

    best_estimate = math.inf
    clusters_u = floors[floor_u].cluster_grid.clusters

    for cluster_id, cluster in enumerate(clusters_u):
        if isinstance(cluster, StairCluster) and cluster.linked_to is not None:
            pos_stair = cluster.get_centroid()
            dist_to_stair = math.hypot(pos_u[0] - pos_stair[0], pos_u[1] - pos_stair[1]) / cluster_step

            target_floor, target_cluster = cluster.linked_to
            if target_floor == floor_v:
                pos_arrival = floors[target_floor].cluster_grid.clusters[target_cluster].get_centroid()
                dist_from_stair = math.hypot(pos_arrival[0] - pos_v[0], pos_arrival[1] - pos_v[1]) / cluster_step
            else:
                dist_from_stair = abs(target_floor - floor_v)

            estimate = dist_to_stair + 1.0 + dist_from_stair
            if estimate < best_estimate:
                best_estimate = estimate

    return best_estimate