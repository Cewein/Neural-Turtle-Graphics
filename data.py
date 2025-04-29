import math
import random
import networkx as nx

def generate_synthetic_graphs(num_graphs=5, grid_size=5, spacing=20, random_extra_edges=2):
    """
    Generate a list of synthetic planar graphs to simulate road layouts.
    Each graph is a grid (grid_size x grid_size) with additional random short connections.
    Nodes are labeled by grid coordinates (i,j) with positions (i*spacing, j*spacing).
    """
    graphs = []
    for _ in range(num_graphs):
        G = nx.Graph()
        # Create grid nodes and edges (4-neighbor connectivity)
        for i in range(grid_size):
            for j in range(grid_size):
                G.add_node((i, j), pos=(i * spacing, j * spacing))
                if i < grid_size - 1:  # horizontal edge
                    G.add_edge((i, j), (i+1, j))
                if j < grid_size - 1:  # vertical edge
                    G.add_edge((i, j), (i, j+1))
        # Add random extra edges to introduce cycles/diagonals
        nodes = list(G.nodes())
        added = 0
        while added < random_extra_edges:
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u != v and not G.has_edge(u, v):
                # Only add if nodes are close (to keep roads local)
                ux, uy = G.nodes[u]['pos']; vx, vy = G.nodes[v]['pos']
                if math.hypot(vx - ux, vy - uy) < 2 * spacing:
                    G.add_edge(u, v)
                    added += 1
        graphs.append(G)
    return graphs

def sample_paths_for_node(G, node, K=5, L=10):
    """
    Sample up to K acyclic incoming paths (length ≤ L) that terminate at 'node'.
    Uses breadth-first search outwards from 'node' (reverse direction of travel).
    Returns a list of paths (each path is a list of node positions from some ancestor to the node).
    """
    paths = []
    queue = [(node, [node])]
    while queue:
        curr, path = queue.pop(0)
        if len(path) - 1 >= L:
            continue  # reached maximum allowed path length
        for nbr in G.neighbors(curr):
            if nbr in path:
                continue  # avoid cycles in the path
            new_path = [nbr] + path  # prepend neighbor (so path still ends at original node)
            paths.append([G.nodes[p]['pos'] for p in new_path])  # store as list of coordinates
            queue.append((nbr, new_path))
    # Deduplicate and limit to K paths
    unique_paths = []
    seen = set()
    for p in paths:
        tup = tuple(p)
        if tup not in seen:
            seen.add(tup)
            unique_paths.append(p)
    random.shuffle(unique_paths)
    return unique_paths[:K]

def prepare_training_data(graphs, K=5, L=10):
    """
    Prepare training samples from each graph.
    Returns a list of (incoming_paths, outgoing_deltas) for each node with neighbors.
      - incoming_paths: list of coordinate paths (each an acyclic path ending at the node)
      - outgoing_deltas: list of (Δx, Δy) displacements for each outgoing edge from the node, sorted CCW.
    """
    data = []
    for G in graphs:
        pos = nx.get_node_attributes(G, 'pos')
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue  # skip isolated nodes (no outgoing edges)
            # Sample incoming paths ending at this node
            inc_paths = sample_paths_for_node(G, node, K=K, L=L)
            # Determine outgoing edges (deltas) sorted by angle around the node
            deltas = []
            base_x, base_y = pos[node]
            angles = []
            for nbr in neighbors:
                dx = pos[nbr][0] - base_x
                dy = pos[nbr][1] - base_y
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                angles.append((angle, dx, dy))
            angles.sort(key=lambda x: x[0])  # sort by angle
            for _, dx, dy in angles:
                # Clamp displacements to [-100,100] range as per paper
                dx = int(round(dx)); dy = int(round(dy))
                dx = max(-100, min(100, dx))
                dy = max(-100, min(100, dy))
                deltas.append((dx, dy))
            data.append((inc_paths, deltas))
    return data

# Example: prepare data from synthetic graphs
# graphs = generate_synthetic_graphs(num_graphs=1, grid_size=3, spacing=20)
# data_samples = prepare_training_data(graphs, K=3, L=5)
# print(f"Prepared {len(data_samples)} training samples from {len(graphs)} graph(s).")
# # Each sample: (incoming_paths, outgoing_deltas)
# print("Sample data format:", data_samples[0][0], "=>", data_samples[0][1])
