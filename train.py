import random
import torch
import torch.optim as optim
import torch.nn as nn
import networkx as nx
import math
import time
import numpy as np # Needed for vector calculations
from model import NTGModel # Assuming model.py is in the same directory
# Import the angle calculation helper if it's in data.py or utils.py
try:
    from data import calculate_vector_angle
except ImportError:
    # Define it locally if not importable (ensure consistency)
    def calculate_vector_angle(v1, v2):
        """Calculates the angle between two 2D vectors in radians (0 to pi)."""
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        angle_diff = abs(angle1 - angle2)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        return angle_diff

# Define the node merging distance threshold from the paper (Sec 3.5 Inference)
NODE_MERGE_DISTANCE = 5.0 # meters

# --- NEW: Helper functions for Line Segment Intersection ---

def on_segment(p, q, r):
    """Given three collinear points p, q, r, the function checks if
    point q lies on line segment 'pr'"""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    """To find orientation of ordered triplet (p, q, r).
    Returns:
        0 --> p, q and r are collinear
        1 --> Clockwise
        2 --> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def do_intersect(p1, q1, p2, q2):
    """Main function to check if line segment 'p1q1' and 'p2q2' intersect."""
    # Find the four orientations needed for general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1): return True
    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1): return True
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2): return True
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False # Doesn't intersect

# --- End Intersection Helpers ---


def train_ntg(model, train_data, epochs=10, batch_size=32, lr=1e-3, weight_decay=1e-4, grad_clip=1.0, device='cpu'):
    """
    Trains the NTG model.
    (Function content remains the same as previous version)
    """
    if not train_data:
        print("Error: Training data is empty. Cannot train.")
        return

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n--- Starting Training ---")
    print(f"Dataset size: {len(train_data)} samples")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")
    print(f"Device: {device}")
    print(f"-------------------------")

    for epoch in range(1, epochs + 1):
        model.train() # Set model to training mode
        random.shuffle(train_data) # Shuffle data at the beginning of each epoch
        epoch_start_time = time.time()
        total_loss = 0.0
        processed_samples = 0
        skipped_batches = 0

        # Simple batching
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            if not batch: continue

            optimizer.zero_grad()
            batch_loss = 0.0
            actual_batch_size = len(batch)
            accumulated_loss = 0.0
            valid_samples_in_batch = 0

            for incoming_paths, target_deltas in batch:
                 if not target_deltas:
                     continue

                 # Forward pass - model handles teacher forcing internally
                 loss = model(incoming_paths, target_deltas)

                 if loss is None or not torch.isfinite(loss):
                      # print(f"Warning: Invalid loss encountered ({loss}). Skipping sample.") # Can be verbose
                      continue

                 accumulated_loss += loss.item() # Track total loss for reporting
                 valid_samples_in_batch += 1
                 (loss / actual_batch_size).backward() # Normalize loss for grad accumulation


            # Only step and clip if valid samples were processed
            if valid_samples_in_batch > 0:
                processed_samples += valid_samples_in_batch
                total_loss += accumulated_loss # Add the batch's total loss

                # Gradient Clipping (applied after accumulating gradients for the batch)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # Optimizer Step (updates weights based on accumulated gradients)
                optimizer.step()

                # Print progress
                current_batch_index = i // batch_size
                total_batches = math.ceil(len(train_data) / batch_size)
                if current_batch_index % 50 == 0 or i >= len(train_data) - batch_size:
                     avg_batch_loss = accumulated_loss / valid_samples_in_batch if valid_samples_in_batch > 0 else 0
                     print(f"  Epoch {epoch}/{epochs} | Batch {current_batch_index + 1}/{total_batches} | Avg Batch Loss: {avg_batch_loss:.4f}")
            else:
                 skipped_batches += 1

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / processed_samples if processed_samples > 0 else 0
        print(f"--- Epoch {epoch} Completed ---")
        print(f"Time: {epoch_duration:.2f}s | Avg Epoch Loss: {avg_epoch_loss:.4f} | Skipped Batches: {skipped_batches}")
        print(f"--------------------------")

    model.eval() # Set model to evaluation mode after training
    print("\nTraining finished.")


def generate_graph(model, start_node_pos=(0.0, 0.0), initial_edges=None,
                   max_nodes=100, K_gen=3, L_gen=5, device='cpu',
                   constraints=None): # <-- constraints argument
    """
    Generates a road layout graph using the trained NTG model, enforcing constraints
    including planarity (no edge crossings).

    Args:
        model (NTGModel): The trained model instance.
        start_node_pos (tuple[float, float]): Position of the initial root node.
        initial_edges (list[tuple[int, int]], optional): Optional initial displacements.
        max_nodes (int): Maximum number of nodes to generate.
        K_gen (int): Number of incoming paths to sample during generation.
        L_gen (int): Max length of incoming paths during generation.
        device (str): Device to run generation on ('cpu' or 'cuda').
        constraints (dict, optional): Dictionary with generation constraints, e.g.,
                                      {'max_degree': int, 'min_angle_rad': float}.

    Returns:
        networkx.Graph: The generated graph with 'pos' attributes.
    """
    model.to(device)
    model.eval() # Ensure model is in evaluation mode

    G = nx.Graph()
    pos_dict = {} # Store positions separately for quick lookup
    next_node_id = 0

    # Initialize constraints with defaults if not provided or partially provided
    max_degree_constraint = float('inf')
    min_angle_constraint_rad = 0.0
    if constraints:
        max_degree_constraint = constraints.get('max_degree', float('inf'))
        min_angle_constraint_rad = constraints.get('min_angle_rad', 0.0)
        print(f"Generation constraints: Max Degree <= {max_degree_constraint}, Min Angle >= {math.degrees(min_angle_constraint_rad):.2f}°")
    else:
        print("No constraints provided for generation.")


    # Add the root node
    root_node_id = next_node_id
    G.add_node(root_node_id)
    pos_dict[root_node_id] = start_node_pos
    next_node_id += 1
    print(f"Generation starting with root node {root_node_id} at {start_node_pos}")

    queue = [] # Initialize queue
    nodes_in_queue = set() # Keep track of nodes currently in the queue

    # Add initial neighbors if specified (with constraint checks)
    initial_neighbor_ids = []
    if initial_edges:
        # --- Simplified initial edge adding (constraints checked similarly below) ---
        # We will re-check constraints robustly in the main loop anyway.
        # This part just sets up the initial queue.
        temp_initial_edges = [] # Store edges added initially
        for dx, dy in initial_edges:
             new_pos = (start_node_pos[0] + dx, start_node_pos[1] + dy)
             merged_node = check_and_merge(G, pos_dict, new_pos, NODE_MERGE_DISTANCE)
             if merged_node is not None:
                  if not G.has_edge(root_node_id, merged_node):
                      G.add_edge(root_node_id, merged_node)
                      temp_initial_edges.append((root_node_id, merged_node))
             else:
                  new_node_id = next_node_id
                  G.add_node(new_node_id)
                  pos_dict[new_node_id] = new_pos
                  G.add_edge(root_node_id, new_node_id)
                  temp_initial_edges.append((root_node_id, new_node_id))
                  initial_neighbor_ids.append(new_node_id)
                  next_node_id += 1
        # --- End Simplified ---

        # Add the *newly created* initial neighbors to the queue
        for nid in initial_neighbor_ids:
             if nid not in nodes_in_queue:
                 queue.append(nid)
                 nodes_in_queue.add(nid)
    else:
        # If no initial edges, the root node itself needs to be expanded
        if max_nodes > 1 and root_node_id not in nodes_in_queue:
            queue.append(root_node_id)
            nodes_in_queue.add(root_node_id)


    processed_nodes = 0
    gen_start_time = time.time()
    max_debug_nodes = 5 # Limit debug prints
    nodes_rejected_degree = 0
    nodes_rejected_angle = 0
    nodes_rejected_intersection = 0 # Add counter for intersection rejections

    with torch.no_grad(): # Disable gradient calculations during inference
        while queue and G.number_of_nodes() < max_nodes:
            current_node_id = queue.pop(0) # Get the next node to expand (FIFO)
            nodes_in_queue.remove(current_node_id)
            processed_nodes += 1

            if current_node_id not in G.nodes or current_node_id not in pos_dict:
                 continue

            current_pos = pos_dict[current_node_id]

            # --- Prepare input for the model ---
            try:
                nx.set_node_attributes(G, pos_dict, 'pos') # Ensure G has up-to-date pos
                from data import sample_paths_for_node
                incoming_paths = sample_paths_for_node(G, current_node_id, K=K_gen, L=L_gen)
            except ImportError:
                 # print("  Warning: Could not import sample_paths_for_node. Using empty paths.")
                 incoming_paths = []
            except Exception as e:
                 # print(f"Error sampling paths for node {current_node_id}: {e}")
                 incoming_paths = []


            # --- Run the model (Inference) ---
            latent = model.encoder(incoming_paths)
            predicted_deltas = model.decoder(latent, target_deltas=None)

            # --- Process predicted deltas ---
            num_added_this_step = 0
            current_degree = G.degree(current_node_id) # Get degree before adding new edges

            # Get existing neighbor vectors for angle checks
            existing_neighbors = list(G.neighbors(current_node_id))
            existing_vectors = []
            if current_degree >= 1:
                 for nbr_id in existing_neighbors:
                      if nbr_id in pos_dict:
                          nbr_pos = pos_dict[nbr_id]
                          vec = (nbr_pos[0] - current_pos[0], nbr_pos[1] - current_pos[1])
                          if vec[0] != 0 or vec[1] != 0:
                              existing_vectors.append(vec)

            # Sort deltas? Optional, might affect order if multiple valid options exist
            # predicted_deltas.sort(key=lambda d: math.atan2(d[1], d[0])) # Example sort

            for dx, dy in predicted_deltas:
                if G.number_of_nodes() >= max_nodes:
                    break

                if dx == 0 and dy == 0:
                     continue

                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                new_vector = (dx, dy)

                # --- Intersection Check ---
                intersects = False
                p1 = current_pos
                q1 = new_pos
                for u, v in G.edges():
                    # Skip if edge involves the current node (avoids self-intersection at start)
                    # Also skip if the edge involves the potential merge target
                    merged_node_id_temp = check_and_merge(G, pos_dict, new_pos, NODE_MERGE_DISTANCE) # Check merge target *before* intersection check
                    if u == current_node_id or v == current_node_id:
                        continue
                    if merged_node_id_temp is not None and (u == merged_node_id_temp or v == merged_node_id_temp):
                        continue

                    # Get positions of the existing edge endpoints
                    if u in pos_dict and v in pos_dict:
                        p2 = pos_dict[u]
                        q2 = pos_dict[v]
                        # Check intersection, ensuring segments are not just touching at endpoints included in the check
                        if do_intersect(p1, q1, p2, q2):
                             # Refine check: only reject if intersection is not at a shared endpoint
                             # This basic check might be too strict if roads meet at non-node points,
                             # but for simplified graphs it aims to prevent clear crossings.
                             # A more advanced check would calculate the intersection point.
                             # For now, we reject if the segments intersect at all (excluding endpoints checked above)
                             intersects = True
                             nodes_rejected_intersection += 1
                             break # No need to check other edges
                if intersects:
                    continue # Skip this delta
                # --- End Intersection Check ---


                # --- Constraint Checks (Degree, Angle) ---
                potential_degree = current_degree + 1
                merged_node_id = check_and_merge(G, pos_dict, new_pos, NODE_MERGE_DISTANCE) # Recalculate here after intersection check

                # 1. Max Degree Check
                if potential_degree > max_degree_constraint:
                    nodes_rejected_degree += 1
                    continue
                if merged_node_id is not None and G.degree(merged_node_id) + 1 > max_degree_constraint:
                    nodes_rejected_degree += 1
                    continue

                # 2. Min Angle Check
                if current_degree >= 1:
                    valid_angle = True
                    for exist_vec in existing_vectors:
                        angle = calculate_vector_angle(new_vector, exist_vec)
                        # Add small tolerance to angle check? e.g., 1e-6
                        if angle < min_angle_constraint_rad - 1e-6:
                             nodes_rejected_angle += 1
                             valid_angle = False
                             break
                    if not valid_angle:
                        continue
                # --- End Constraint Checks ---


                # --- Add edge/node if constraints passed ---
                if merged_node_id is not None:
                    # Add edge to existing node
                    if not G.has_edge(current_node_id, merged_node_id) and current_node_id != merged_node_id:
                        G.add_edge(current_node_id, merged_node_id)
                        current_degree += 1 # Update degree for next iteration
                        existing_vectors.append(new_vector) # Update vectors
                else:
                    # Add a new node and edge
                    new_node_id = next_node_id
                    G.add_node(new_node_id)
                    pos_dict[new_node_id] = new_pos
                    G.add_edge(current_node_id, new_node_id)
                    if new_node_id not in nodes_in_queue:
                         queue.append(new_node_id)
                         nodes_in_queue.add(new_node_id)
                    next_node_id += 1
                    num_added_this_step += 1
                    current_degree += 1 # Update degree
                    existing_vectors.append(new_vector) # Update vectors


    gen_duration = time.time() - gen_start_time
    print(f"\n--- Generation Finished ---")
    print(f"Time: {gen_duration:.2f}s")
    print(f"Processed {processed_nodes} nodes from queue.")
    print(f"Nodes rejected by degree constraint: {nodes_rejected_degree}")
    print(f"Nodes rejected by angle constraint: {nodes_rejected_angle}")
    print(f"Nodes rejected by intersection: {nodes_rejected_intersection}") # Report intersection rejections
    # Set final positions before returning
    nx.set_node_attributes(G, pos_dict, 'pos')
    print(f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"--------------------------")
    return G

# Updated check_and_merge to use pos_dict for efficiency
def check_and_merge(G, pos_dict, target_pos, threshold):
    """
    Checks if target_pos is close to any existing node in G within the threshold, using pos_dict.
    (Function content remains the same as previous version)
    """
    min_dist_sq = threshold * threshold # Compare squared distances
    closest_node = None
    tx, ty = target_pos

    # Iterate through the positions dictionary
    for node_id, node_pos in pos_dict.items():
        nx_node, ny_node = node_pos
        dist_sq = (tx - nx_node)**2 + (ty - ny_node)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_node = node_id

    return closest_node
