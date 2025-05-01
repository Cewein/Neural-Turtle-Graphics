import math
import random
import networkx as nx
import numpy as np # Import numpy for float types if needed for printing
from osm_parser import graph_from_osm # Import the new OSM parser

# --- Constants from Model (for clamping) ---
# Ensure this matches the value in model.py
MAX_DISPLACEMENT = 300
# --- End Constants ---

def _perform_random_walk(G, start_node, max_length_L):
    """
    Performs a single random walk backwards from start_node up to max_length_L.

    Args:
        G (nx.Graph): The input graph.
        start_node: The node where the incoming path terminates (walk starts here).
        max_length_L (int): Maximum number of edges in the path.

    Returns:
        list or None: A list of node IDs representing the path (from ancestor to start_node),
                      or None if a valid path cannot be formed (e.g., start_node has no neighbors).
                      Returns path even if shorter than max_length_L if walk gets stuck.
    """
    if start_node not in G:
        return None

    path_nodes_rev = [start_node] # Path stored in reverse (start_node -> ancestor)
    current_node = start_node

    for _ in range(max_length_L): # Iterate up to L steps (edges)
        neighbors = list(G.neighbors(current_node))

        # Filter neighbors: exclude the node we just came from (if path has >1 node)
        # and exclude nodes already in the current path to ensure acyclicity.
        valid_neighbors = []
        previous_node = path_nodes_rev[-2] if len(path_nodes_rev) > 1 else None
        for n in neighbors:
             # Ensure neighbor exists and has pos (redundant check if graph is clean, but safe)
            if n not in G or 'pos' not in G.nodes[n]:
                continue
            # Avoid immediate backtracking and cycles within the walk
            if n != previous_node and n not in path_nodes_rev:
                 valid_neighbors.append(n)


        if not valid_neighbors:
            break # Walk is stuck (dead end or only leads back/into cycle)

        # Choose the next node randomly from valid neighbors
        next_node = random.choice(valid_neighbors)
        path_nodes_rev.append(next_node)
        current_node = next_node

    if len(path_nodes_rev) < 2: # Path needs at least two nodes (one edge)
        return None

    # Return the path in the correct order (ancestor -> ... -> start_node)
    return list(reversed(path_nodes_rev))


def sample_paths_for_node(G, node, K=5, L=10, max_attempts_factor=5):
    """
    Sample up to K unique, acyclic incoming paths (length <= L) that terminate at 'node'
    using random walks starting from 'node' and moving backwards.

    Args:
        G (nx.Graph): The input graph with 'pos' attributes for nodes.
        node: The target node to find incoming paths for.
        K (int): Maximum number of unique paths to sample.
        L (int): Maximum length (number of edges) of each path.
        max_attempts_factor (int): Factor to determine max attempts (K * factor) to find unique paths.

    Returns:
        list[list[tuple[float, float]]]: A list of unique paths, where each path is a list of (x, y) coordinates.
    """
    if node not in G or 'pos' not in G.nodes[node]:
        return []

    pos = nx.get_node_attributes(G, 'pos')
    unique_paths_nodes = set() # Store tuples of node IDs to ensure uniqueness
    max_attempts = K * max_attempts_factor
    attempts = 0

    while len(unique_paths_nodes) < K and attempts < max_attempts:
        attempts += 1
        path_nodes = _perform_random_walk(G, node, L)

        if path_nodes:
            # Add the tuple representation of the node path to the set
            unique_paths_nodes.add(tuple(path_nodes))

    # Convert unique node paths to coordinate paths
    final_paths_coords = []
    for p_nodes_tuple in unique_paths_nodes:
        path_coords = []
        valid_path = True
        for n in p_nodes_tuple:
            # Check if node still exists and has position (should be true if walk succeeded)
            if n in pos:
                path_coords.append(pos[n])
            else:
                # print(f"Warning: Node {n} in sampled path lacks 'pos'. Skipping path.")
                valid_path = False
                break
        if valid_path and len(path_coords) >= 2: # Ensure path has at least one edge
            final_paths_coords.append(path_coords)

    # Note: We might return fewer than K paths if walks fail or duplicates are frequent.
    return final_paths_coords


def prepare_training_data_from_graphs(graphs, K=5, L=10):
    """
    Prepare training samples from a list of NetworkX graphs.
    Assumes graphs have nodes with 'pos' attribute containing (x, y) coordinates in meters.
    Uses random walk sampling for incoming paths.

    Args:
        graphs (list[nx.Graph]): List of graphs loaded (e.g., from OSM).
        K (int): Max number of unique incoming paths to sample per node via random walk.
        L (int): Max length of sampled incoming paths.

    Returns:
        list[tuple]: A list of training samples, where each sample is:
                     (incoming_paths, outgoing_deltas)
                     - incoming_paths: list[list[tuple[float, float]]] (from random walks)
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

            # --- MODIFIED PART: Use random walk sampling ---
            inc_paths = sample_paths_for_node(G, node, K=K, L=L)
            # --- END MODIFICATION ---

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
            # Note: We might still add a sample even if inc_paths is empty,
            # allowing the model to learn generation from minimal history.
            if deltas:
                 data.append((inc_paths, deltas))

        print(f"Finished processing graph {G_idx+1}. Found {len(data)} samples so far.")

    print(f"\n--- Data Preparation Summary (using Random Walks) ---")
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
            print("\nExample Sample (using Random Walks):")
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

# --- NEW: Statistics Calculation ---

def calculate_vector_angle(v1, v2):
    """Calculates the angle between two 2D vectors in radians (0 to pi)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0] # Needed for atan2 for full range
    angle_rad = math.atan2(det, dot) # Gives angle from v1 to v2 (-pi to pi)

    # We want the minimum angle between them, so use absolute value
    # But need to handle the range carefully. Let's use atan2 for each vector
    # relative to x-axis and find the difference.
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    angle_diff = abs(angle1 - angle2)

    # Normalize angle difference to be between 0 and pi
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    return angle_diff


def calculate_graph_statistics(graphs, degree_percentile=99, angle_percentile=1):
    """
    Calculates statistics (max degree, min angle) from training graphs for generation constraints.

    Args:
        graphs (list[nx.Graph]): List of training graphs.
        degree_percentile (float): Percentile for maximum allowed node degree.
        angle_percentile (float): Percentile for minimum allowed angle between edges (in degrees).

    Returns:
        dict: A dictionary containing constraint thresholds, e.g.,
              {'max_degree': int, 'min_angle_rad': float}
              Returns None if graphs are empty or stats cannot be computed.
    """
    print("\n--- Calculating Statistics from Training Graphs ---")
    all_degrees = []
    min_angles_at_nodes = [] # Store the minimum angle found at each node with degree >= 2

    if not graphs:
        print("Warning: No graphs provided for statistics calculation.")
        return None

    pos_missing_count = 0
    nodes_processed = 0
    nodes_with_angles = 0

    for G_idx, G in enumerate(graphs):
        if not G: continue
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            print(f"Warning: Graph {G_idx} missing 'pos' attributes. Skipping.")
            continue

        # print(f"Processing graph {G_idx+1}/{len(graphs)} for stats...")
        for node in G.nodes():
            nodes_processed += 1
            degree = G.degree(node)
            all_degrees.append(degree)

            if degree >= 2:
                neighbors = list(G.neighbors(node))
                if node not in pos:
                     pos_missing_count += 1
                     continue

                node_pos = pos[node]
                neighbor_vectors = []
                valid_neighbors = 0
                for nbr in neighbors:
                    if nbr in pos:
                        nbr_pos = pos[nbr]
                        # Calculate vector from node to neighbor
                        vec = (nbr_pos[0] - node_pos[0], nbr_pos[1] - node_pos[1])
                        # Ensure vector is not zero length (shouldn't happen in simplified graph)
                        if vec[0] != 0 or vec[1] != 0:
                           neighbor_vectors.append(vec)
                           valid_neighbors += 1
                    else:
                         pos_missing_count += 1


                if valid_neighbors >= 2: # Need at least two vectors to calculate an angle
                    nodes_with_angles += 1
                    min_angle_for_node = math.pi # Initialize with max possible angle (180 deg)
                    # Calculate angle between all pairs of neighbor vectors
                    for i in range(len(neighbor_vectors)):
                        for j in range(i + 1, len(neighbor_vectors)):
                            angle = calculate_vector_angle(neighbor_vectors[i], neighbor_vectors[j])
                            min_angle_for_node = min(min_angle_for_node, angle)

                    # Only add valid angles (avoiding potential issues if min_angle_for_node remains pi)
                    if min_angle_for_node < math.pi:
                         min_angles_at_nodes.append(min_angle_for_node)


    if pos_missing_count > 0:
         print(f"Warning: Encountered {pos_missing_count} missing node positions during angle calculation.")

    if not all_degrees:
        print("Warning: Could not calculate degree statistics.")
        return None

    # Calculate thresholds using percentiles
    max_degree_threshold = int(np.percentile(all_degrees, degree_percentile)) if all_degrees else 5 # Default fallback
    min_angle_threshold_rad = np.percentile(min_angles_at_nodes, angle_percentile) if min_angles_at_nodes else 0.1 # Default fallback (radians)

    # Convert min angle to degrees for printing
    min_angle_threshold_deg = math.degrees(min_angle_threshold_rad)

    print(f"Statistics calculated from {len(graphs)} graphs ({nodes_processed} nodes):")
    print(f"  Node Degrees: Min={np.min(all_degrees)}, Max={np.max(all_degrees)}, Mean={np.mean(all_degrees):.2f}")
    print(f"  Degree Threshold ({degree_percentile}th percentile): {max_degree_threshold}")
    if min_angles_at_nodes:
        print(f"  Min Angles (at {nodes_with_angles} nodes): Min={math.degrees(np.min(min_angles_at_nodes)):.2f}°, Max={math.degrees(np.max(min_angles_at_nodes)):.2f}°, Mean={math.degrees(np.mean(min_angles_at_nodes)):.2f}°")
        print(f"  Min Angle Threshold ({angle_percentile}th percentile): {min_angle_threshold_deg:.2f}° ({min_angle_threshold_rad:.3f} rad)")
    else:
        print("  Warning: Could not calculate angle statistics (no nodes with degree >= 2 or issues found). Using default.")


    constraints = {
        'max_degree': max_degree_threshold,
        'min_angle_rad': min_angle_threshold_rad
    }
    print(f"--- Statistics Calculation Finished. Constraints: {constraints} ---")
    return constraints

# --- End NEW ---
