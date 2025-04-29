import math
import random
import networkx as nx
import numpy as np # Import numpy for float types if needed for printing
from osm_parser import graph_from_osm # Import the new OSM parser

# --- Constants from Model (for clamping) ---
# Ensure this matches the value in model.py
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
        # print(f"Warning: Node {node} not found in graph during path sampling.") # Can be verbose
        return []
    if 'pos' not in G.nodes[node]:
         # print(f"Warning: Node {node} missing 'pos' attribute.") # Can be verbose
         return []

    paths_nodes = [] # Store paths as node sequences first
    queue = [(node, [node])] # (current_node_in_bfs, path_so_far_reversed)
    visited_edges = set() # To avoid trivial back-and-forth cycles in BFS

    pos = nx.get_node_attributes(G, 'pos')

    # Limit BFS depth implicitly by path length check later
    # Limit queue size or iterations to prevent excessive search on dense graphs if needed
    bfs_steps = 0
    max_bfs_steps = G.number_of_nodes() * 2 # Heuristic limit

    while queue and bfs_steps < max_bfs_steps:
        bfs_steps += 1
        if not queue: break # Should not happen with the check above, but safety
        curr, path_nodes_rev = queue.pop(0)

        # Path length is number of edges, which is len(nodes) - 1
        if len(path_nodes_rev) - 1 >= L:
            continue

        # Explore neighbors of the current node in the reversed path search
        neighbors = list(G.neighbors(curr))
        random.shuffle(neighbors) # Add randomness to BFS exploration order

        for neighbor in neighbors:
            # Avoid immediate loops back along the edge we just traversed in BFS
            # Avoid adding nodes already in the current path sequence to ensure acyclic
            edge = tuple(sorted((curr, neighbor)))
            if neighbor in path_nodes_rev or edge in visited_edges:
                continue

            # Check if neighbor exists and has position
            if neighbor not in G or 'pos' not in G.nodes[neighbor]:
                continue

            new_path_nodes_rev = [neighbor] + path_nodes_rev
            visited_edges.add(edge) # Mark edge as visited for this BFS expansion direction

            # Store the path in the correct order (ancestor -> ... -> node)
            paths_nodes.append(list(reversed(new_path_nodes_rev)))

            # Add neighbor to the queue for further exploration
            # Only add if path length allows further steps
            if len(new_path_nodes_rev) - 1 < L:
                 queue.append((neighbor, new_path_nodes_rev))

            # Optimization: Stop adding to queue if we already have many candidate paths?
            if len(paths_nodes) > K * 10: # Stop exploring if we have 10x K paths found
                 break
        if len(paths_nodes) > K * 10: break


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
                # print(f"Warning: Node {n} in path lacks 'pos' attribute. Skipping path.")
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
    # --- Statistics ---
    total_deltas_processed = 0
    deltas_clamped_count = 0
    # --- End Statistics ---

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
                 continue

            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue # Skip isolated nodes (no outgoing edges to predict)

            nodes_with_neighbors += 1

            # Sample incoming paths ending at this node
            inc_paths = sample_paths_for_node(G, node, K=K, L=L)

            # Determine outgoing edges (deltas) sorted by angle around the node
            deltas = []
            base_x, base_y = pos[node]
            angles = []
            for nbr in neighbors:
                 if nbr not in pos:
                     continue

                 dx_float = pos[nbr][0] - base_x
                 dy_float = pos[nbr][1] - base_y

                 # Calculate angle (counter-clockwise from positive x-axis)
                 angle = math.atan2(dy_float, dx_float)
                 if angle < 0:
                     angle += 2 * math.pi
                 angles.append((angle, dx_float, dy_float)) # Keep float deltas for now

            # Sort neighbors by angle (counter-clockwise)
            angles.sort(key=lambda x: x[0])

            # Clamp displacements and convert to integers
            for _, dx_float, dy_float in angles:
                total_deltas_processed += 2 # Counting dx and dy separately

                dx_clamped = int(round(dx_float))
                dy_clamped = int(round(dy_float))

                # Check for clamping before applying it
                if dx_clamped > MAX_DISPLACEMENT or dx_clamped < -MAX_DISPLACEMENT:
                    deltas_clamped_count += 1
                if dy_clamped > MAX_DISPLACEMENT or dy_clamped < -MAX_DISPLACEMENT:
                    deltas_clamped_count += 1

                # Apply clamping
                dx_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dx_clamped))
                dy_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dy_clamped))

                deltas.append((dx_clamped, dy_clamped))

            # Add the sample (only if there are outgoing deltas)
            if deltas:
                 data.append((inc_paths, deltas))

        print(f"Finished processing graph {G_idx+1}. Found {len(data)} samples so far.")

    print(f"\n--- Data Preparation Summary ---")
    print(f"Total nodes processed across all graphs: {total_nodes_processed}")
    print(f"Nodes with neighbors (potential samples): {nodes_with_neighbors}")
    print(f"Total training samples generated: {len(data)}")

    # --- Print Statistics ---
    if total_deltas_processed > 0:
        clamping_percentage = (deltas_clamped_count / total_deltas_processed) * 100
        print(f"\nDelta Clamping Statistics (Range [-{MAX_DISPLACEMENT}, {MAX_DISPLACEMENT}]):")
        print(f"  Total delta values processed (dx + dy): {total_deltas_processed}")
        print(f"  Number of delta values clamped: {deltas_clamped_count}")
        print(f"  Clamping Percentage: {clamping_percentage:.2f}%")
    else:
        print("\nNo deltas processed, cannot calculate clamping statistics.")
    # --- End Print Statistics ---


    if not data:
        print("Warning: No training data was generated. Check graph structures and parameters (K, L).")
    else:
        # Print an example sample
        sample_idx = min(5, len(data)-1) # Print one of the first few samples
        if sample_idx >= 0: # Ensure data is not empty
            print("\nExample Sample:")
            print(f"  Incoming Paths (K={K}, L={L}, {len(data[sample_idx][0])} sampled):")
            # Use numpy float type for printing if needed, otherwise standard float is fine
            np_float_type = getattr(np, "float64", float) # Get numpy float64 if available, else use standard float
            for i, p in enumerate(data[sample_idx][0]):
                # Format coordinates for printing to avoid excessive precision
                path_str = ", ".join([f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in p])
                print(f"    Path {i+1} (len {len(p)}): [{path_str[:100]}...]" if len(path_str) > 100 else f"    Path {i+1} (len {len(p)}): [{path_str}]")
                if i >= 2: # Print max 3 paths for brevity
                    print("    ...")
                    break
            print(f"  Outgoing Deltas (Sorted CCW, Clamped [-{MAX_DISPLACEMENT},{MAX_DISPLACEMENT}]): {data[sample_idx][1]}")
    print("---------------------------------")

    return data


# --- Synthetic graph generator for testing/comparison if needed ---
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
                # Add edges only if the neighbor node exists (avoids index errors at boundary)
                if i < grid_size - 1:  # horizontal edge to the right
                     neighbor_id = (i+1, j)
                     if neighbor_id in G: G.add_edge(node_id, neighbor_id)
                if j < grid_size - 1:  # vertical edge upwards
                     neighbor_id = (i, j+1)
                     if neighbor_id in G: G.add_edge(node_id, neighbor_id)
                # Also add edges to left and down for completeness if desired,
                # but the above covers all edges once.

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
                # Ensure nodes have 'pos' before accessing
                if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
                    ux, uy = G.nodes[u]['pos']
                    vx, vy = G.nodes[v]['pos']
                    # Add edge if distance is less than sqrt(2)*spacing (diagonal) + epsilon
                    if math.hypot(vx - ux, vy - uy) < 1.5 * spacing:
                        G.add_edge(u, v)
                        added += 1
        graphs.append(G)
    print(f"Generated {len(graphs)} synthetic graphs.")
    return graphs

