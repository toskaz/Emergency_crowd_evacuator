import numpy as np


# Shortest paths between every pair of nodes.
# weights[i][j] is the cost of going from i to j (np.inf if there's no edge, 0 on the diagonal).
# Apart from the distances we also remember next_node[i][j], the first node to go to on the way
# from i to j, so we can rebuild the whole path later (-1 means you can't get from i to j at all).
def floyd_warshall(weights):
    dist = np.array(weights, dtype=float)
    node_count = len(dist)
    next_node = np.where(np.isfinite(dist), np.arange(node_count), -1)

    for k in range(node_count):
        # Same as the textbook triple loop, numpy just does the two inner loops in one go.
        # That's ok because row k and column k never change while we're on k (going from k to k
        # costs 0), so it doesn't matter in which order the pairs get updated.
        through_k = dist[:, k:k + 1] + dist[k:k + 1, :]
        shorter = through_k < dist

        np.copyto(dist, through_k, where=shorter)
        np.copyto(next_node, next_node[:, k:k + 1].copy(), where=shorter)

    return dist, next_node


def reconstruct_path(next_node, start, goal):
    if next_node[start, goal] < 0:
        return []

    path = [start]

    while start != goal:
        start = int(next_node[start, goal])
        path.append(start)

    return path
