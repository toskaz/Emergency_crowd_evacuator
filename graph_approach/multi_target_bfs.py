import math
from collections import deque


def find_path_multi_target_bfs(G, source_nodes, target_node):
    if not source_nodes or target_node is None:
        return math.inf, []

    if target_node in source_nodes:
        return 0.0, [target_node]

    queue = deque(source_nodes)
    visited = {node: None for node in source_nodes if node in G}

    found = False
    while queue:
        current = queue.popleft()
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = current
                if neighbor == target_node:
                    found = True
                    break
                queue.append(neighbor)
        if found:
            break

    if not found or target_node not in visited:
        return math.inf, []

    path = []
    curr = target_node
    while curr is not None:
        path.append(curr)
        curr = visited[curr]

    distance = float(len(path) - 1)

    return distance, path
