import math
import random
import networkx as nx
from osm_parser import graph_from_osm # Import the new OSM parser

# --- Constants from Model (for clamping) ---
# It's slightly better practice to define these in a config or pass them,
# but let's keep it simple for now and mirror model.py
MAX_DISPLACEMENT = 100
# --- End Constants ---

def sample_paths_for_node(G, node, K=5, L=10):
    """
    Sample up to K acyclic incoming paths (length <= L) that terminate at 'node'.
    Uses breadth-first search outwards from 'node' (reverse direction of travel).
    Returns a list of paths (each path is a list of node positions from some ancestor to the node).

    Args:
        G (nx.Graph): The input graph with 'pos' attributes for nodes.
        node: The target node to find incoming paths for.
        K (int): Maximum number of paths to sample.
        L (int): Maximum length of each path (number of edges).

    Returns:
        list[list[tuple[float, float]]]: A list of paths, where each path is a list of (x, y) coordinates.
    """
    if node not in G:
        print(f"Warning: Node {node} not found in graph during path sampling.")
        return []
    if 'pos' not in G.nodes[node]:
         print(f"Warning: Node {node} missing 'pos' attribute.")
         return []

    paths_nodes = [] # Store paths as node sequences first
    queue = [(node, [node])] # (current_node_in_bfs, path_so_far_reversed)
    visited_edges = set() # To avoid trivial back-and-forth cycles in BFS

    pos = nx.get_node_attributes(G, 'pos')

    while queue:
        curr, path_nodes_rev = queue.pop(0)

        # Path length is number of edges, which is len(nodes) - 1
        if len(path_nodes_rev) - 1 >= L:
            continue

        # Explore neighbors of the current node in the reversed path search
        for neighbor in G.neighbors(curr):
            # Avoid immediate loops back along the edge we just traversed in BFS
            # Avoid adding nodes already in the current path sequence to ensure acyclic
            edge = tuple(sorted((curr, neighbor)))
            if neighbor in path_nodes_rev or edge in visited_edges:
                continue

            new_path_nodes_rev = [neighbor] + path_nodes_rev
            visited_edges.add(edge) # Mark edge as visited for this BFS expansion direction

            # Store the path in the correct order (ancestor -> ... -> node)
            paths_nodes.append(list(reversed(new_path_nodes_rev)))

            # Add neighbor to the queue for further exploration
            queue.append((neighbor, new_path_nodes_rev))

            # Optimization: If we have enough paths already, maybe stop early?
            # The paper doesn't specify, let BFS complete for potentially diverse paths.
            # if len(paths_nodes) >= K * 5: # Heuristic limit if needed
            #     break
        # if len(paths_nodes) >= K * 5: break # Heuristic limit if needed


    # Convert node paths to coordinate paths and ensure uniqueness
    unique_paths_coords = []
    seen_coord_tuples = set()
    for p_nodes in paths_nodes:
        path_coords = []
        valid_path = True
        for n in p_nodes:
            if n in pos:
                path_coords.append(pos[n])
            else:
                # This should not happen if graph preprocessing is correct
                print(f"Warning: Node {n} in path lacks 'pos' attribute. Skipping path.")
                valid_path = False
                break
        if not valid_path or len(path_coords) < 2: # Need at least 2 points for a delta
            continue

        # Use tuple of tuples for hashing coordinates to check uniqueness
        path_coord_tuple = tuple(path_coords)
        if path_coord_tuple not in seen_coord_tuples:
            seen_coord_tuples.add(path_coord_tuple)
            unique_paths_coords.append(path_coords) # Store as list of tuples

    # Shuffle and limit to K paths
    random.shuffle(unique_paths_coords)
    return unique_paths_coords[:K]

def prepare_training_data_from_graphs(graphs, K=5, L=10):
    """
    Prepare training samples from a list of NetworkX graphs.
    Assumes graphs have nodes with 'pos' attribute containing (x, y) coordinates in meters.

    Args:
        graphs (list[nx.Graph]): List of graphs loaded (e.g., from OSM).
        K (int): Max number of incoming paths to sample per node.
        L (int): Max length of sampled incoming paths.

    Returns:
        list[tuple]: A list of training samples, where each sample is:
                     (incoming_paths, outgoing_deltas)
                     - incoming_paths: list[list[tuple[float, float]]]
                     - outgoing_deltas: list[tuple[int, int]] (clamped dx, dy sorted CCW)
    """
    data = []
    total_nodes_processed = 0
    nodes_with_neighbors = 0

    for G_idx, G in enumerate(graphs):
        print(f"\nProcessing graph {G_idx+1}/{len(graphs)} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        if not G or G.number_of_nodes() == 0:
             print("Skipping empty graph.")
             continue

        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            print("Warning: Graph has no 'pos' attributes. Cannot process.")
            continue

        processed_nodes_in_graph = 0
        for node in G.nodes():
            total_nodes_processed += 1
            processed_nodes_in_graph += 1
            if processed_nodes_in_graph % 500 == 0:
                 print(f"  Processed {processed_nodes_in_graph}/{G.number_of_nodes()} nodes in graph {G_idx+1}...")

            # Check if node has position data
            if node not in pos:
                 # print(f"Skipping node {node} without position data.") # Can be verbose
                 continue

            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue # Skip isolated nodes (no outgoing edges to predict)

            nodes_with_neighbors += 1

            # Sample incoming paths ending at this node
            # Ensure the node itself has 'pos' before sampling
            inc_paths = sample_paths_for_node(G, node, K=K, L=L)

            # Determine outgoing edges (deltas) sorted by angle around the node
            deltas = []
            base_x, base_y = pos[node]
            angles = []
            for nbr in neighbors:
                 # Ensure neighbor has position data
                 if nbr not in pos:
                     # print(f"Skipping neighbor {nbr} of node {node} without position data.") # Can be verbose
                     continue

                 dx = pos[nbr][0] - base_x
                 dy = pos[nbr][1] - base_y

                 # Calculate angle (counter-clockwise from positive x-axis)
                 angle = math.atan2(dy, dx)
                 # Map angle to [0, 2*pi) range
                 if angle < 0:
                     angle += 2 * math.pi
                 angles.append((angle, dx, dy))

            # Sort neighbors by angle (counter-clockwise) - crucial for decoder target sequence
            angles.sort(key=lambda x: x[0])

            # Clamp displacements to [-MAX_DISPLACEMENT, MAX_DISPLACEMENT] range as per paper
            # And convert to integers
            for _, dx, dy in angles:
                dx_clamped = int(round(dx))
                dy_clamped = int(round(dy))
                dx_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dx_clamped))
                dy_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dy_clamped))
                deltas.append((dx_clamped, dy_clamped))

            # Add the sample (only if there are outgoing deltas)
            if deltas:
                 # We need both incoming paths (can be empty if node is a source in sample)
                 # and outgoing deltas for a valid training sample.
                 data.append((inc_paths, deltas))

        print(f"Finished processing graph {G_idx+1}. Found {len(data)} samples so far.")

    print(f"\n--- Data Preparation Summary ---")
    print(f"Total nodes processed across all graphs: {total_nodes_processed}")
    print(f"Nodes with neighbors (potential samples): {nodes_with_neighbors}")
    print(f"Total training samples generated: {len(data)}")
    if not data:
        print("Warning: No training data was generated. Check graph structures and parameters (K, L).")
    else:
        # Print an example sample
        sample_idx = min(5, len(data)-1) # Print one of the first few samples
        print("\nExample Sample:")
        print(f"  Incoming Paths (K={K}, L={L}, {len(data[sample_idx][0])} sampled):")
        for i, p in enumerate(data[sample_idx][0]):
            print(f"    Path {i+1} (len {len(p)}): {p[:2]}...{p[-2:]}" if len(p) > 4 else f"    Path {i+1} (len {len(p)}): {p}")
            if i >= 2: # Print max 3 paths for brevity
                print("    ...")
                break
        print(f"  Outgoing Deltas (Sorted CCW, Clamped [-{MAX_DISPLACEMENT},{MAX_DISPLACEMENT}]): {data[sample_idx][1]}")
    print("---------------------------------")

    return data


# --- ynthetic graph generator for testing/comparison if needed ---
def generate_synthetic_graphs(num_graphs=5, grid_size=5, spacing=20, random_extra_edges=2):
    """
    Generate a list of synthetic planar graphs to simulate road layouts.
    Each graph is a grid (grid_size x grid_size) with additional random short connections.
    Nodes are labeled by grid coordinates (i,j) with positions (i*spacing, j*spacing).
    """
    print(f"\nGenerating {num_graphs} synthetic grid graphs ({grid_size}x{grid_size})...")
    graphs = []
    for _ in range(num_graphs):
        G = nx.Graph()
        pos_dict = {}
        # Create grid nodes and edges (4-neighbor connectivity)
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = (i, j)
                node_pos = (i * spacing, j * spacing)
                G.add_node(node_id)
                pos_dict[node_id] = node_pos
                if i < grid_size - 1:  # horizontal edge
                    G.add_edge(node_id, (i+1, j))
                if j < grid_size - 1:  # vertical edge
                    G.add_edge(node_id, (i, j+1))

        nx.set_node_attributes(G, pos_dict, 'pos') # Set positions after creating all nodes

        # Add random extra edges to introduce cycles/diagonals
        nodes = list(G.nodes())
        added = 0
        attempts = 0
        max_attempts = random_extra_edges * 10 # Avoid infinite loops if graph is small

        while added < random_extra_edges and attempts < max_attempts:
            attempts += 1
            if len(nodes) < 2: break # Need at least 2 nodes
            u, v = random.sample(nodes, 2) # Use random.sample to ensure u != v

            if not G.has_edge(u, v):
                # Only add if nodes are close (to keep roads local)
                ux, uy = G.nodes[u]['pos']; vx, vy = G.nodes[v]['pos']
                # Add edge if distance is less than sqrt(2)*spacing (diagonal) + epsilon
                if math.hypot(vx - ux, vy - uy) < 1.5 * spacing:
                    G.add_edge(u, v)
                    added += 1
        graphs.append(G)
    print(f"Generated {len(graphs)} synthetic graphs.")
    return graphs

