import random
import torch
import torch.optim as optim
import torch.nn as nn
import networkx as nx
import math
import time
from model import NTGModel # Assuming model.py is in the same directory

# Define the node merging distance threshold from the paper (Sec 3.5 Inference)
NODE_MERGE_DISTANCE = 5.0 # meters

def train_ntg(model, train_data, epochs=10, batch_size=32, lr=1e-3, weight_decay=1e-4, grad_clip=1.0, device='cpu'):
    """
    Trains the NTG model.

    Args:
        model (NTGModel): The model instance to train.
        train_data (list[tuple]): List of training samples (incoming_paths, target_deltas).
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per training step.
        lr (float): Learning rate for the Adam optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        grad_clip (float): Maximum norm for gradient clipping.
        device (str): Device to train on ('cpu' or 'cuda').
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

        # Simple batching (can be improved with DataLoader if data is very large)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            if not batch: continue

            optimizer.zero_grad()
            batch_loss = 0.0

            # Accumulate gradients over the batch
            # Note: This simple batching processes samples individually within the loop.
            # True batching would require padding paths/deltas to the same length,
            # which adds complexity. Let's stick to sequential processing with gradient accumulation.
            # For simplicity here, let's just process one by one and accumulate loss for reporting.
            # A more proper implementation would use a DataLoader and padding.
            # Let's refine this to accumulate gradients for the batch size.

            actual_batch_size = len(batch)
            accumulated_loss = 0.0

            for incoming_paths, target_deltas in batch:
                 # --- Data Preparation for Model ---
                 # The model expects paths as lists of tuples (floats ok)
                 # Target deltas should be list of tuples (ints)

                 # Skip if target_deltas is empty (shouldn't happen if data prep is correct)
                 if not target_deltas:
                     continue

                 # Ensure data is on the correct device (model handles internal tensors)
                 # The input data structure (lists of lists/tuples) stays on CPU usually.
                 # The model's forward pass will move tensors to the device.

                 # Forward pass - model handles teacher forcing internally
                 loss = model(incoming_paths, target_deltas)

                 # Check if loss is valid
                 if loss is None or not torch.isfinite(loss):
                      print(f"Warning: Invalid loss encountered ({loss}). Skipping sample.")
                      # Potentially log the problematic sample data here
                      # print("Problematic sample:")
                      # print("Incoming paths:", incoming_paths)
                      # print("Target deltas:", target_deltas)
                      continue

                 # Normalize loss by batch size for gradient accumulation
                 loss = loss / actual_batch_size
                 accumulated_loss += loss.item() * actual_batch_size # Track un-normalized loss

                 # Backward pass to accumulate gradients
                 loss.backward()
                 processed_samples += 1


            # Gradient Clipping (applied after accumulating gradients for the batch)
            # Check if any gradients exist before clipping
            if any(p.grad is not None for p in model.parameters()):
                 nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                 print("Warning: No gradients found to clip in this batch.")


            # Optimizer Step (updates weights based on accumulated gradients)
            optimizer.step()

            total_loss += accumulated_loss # Add the batch's total loss

            # Print progress
            # Use integer division for batch index
            current_batch_index = i // batch_size
            total_batches = math.ceil(len(train_data) / batch_size)
            if current_batch_index % 50 == 0 or i >= len(train_data) - batch_size:
                 print(f"  Epoch {epoch}/{epochs} | Batch {current_batch_index + 1}/{total_batches} | Avg Batch Loss: {accumulated_loss / actual_batch_size:.4f}")


        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / processed_samples if processed_samples > 0 else 0
        print(f"--- Epoch {epoch} Completed ---")
        print(f"Time: {epoch_duration:.2f}s | Avg Epoch Loss: {avg_epoch_loss:.4f}")
        print(f"--------------------------")

    model.eval() # Set model to evaluation mode after training
    print("\nTraining finished.")


def generate_graph(model, start_node_pos=(0.0, 0.0), initial_edges=None, max_nodes=100, K_gen=3, L_gen=5, device='cpu'):
    """
    Generates a road layout graph using the trained NTG model.

    Args:
        model (NTGModel): The trained model instance.
        start_node_pos (tuple[float, float]): Position of the initial root node.
        initial_edges (list[tuple[int, int]], optional): Optional initial displacements
                                                        to add neighbors to the root node.
        max_nodes (int): Maximum number of nodes to generate.
        K_gen (int): Number of incoming paths to sample during generation.
        L_gen (int): Max length of incoming paths during generation.
        device (str): Device to run generation on ('cpu' or 'cuda').

    Returns:
        networkx.Graph: The generated graph with 'pos' attributes.
    """
    model.to(device)
    model.eval() # Ensure model is in evaluation mode

    G = nx.Graph()
    next_node_id = 0

    # Add the root node
    root_node_id = next_node_id
    G.add_node(root_node_id, pos=start_node_pos)
    next_node_id += 1
    print(f"Generation starting with root node {root_node_id} at {start_node_pos}")

    queue = [] # Initialize queue

    # Add initial neighbors if specified *before* adding root to queue
    if initial_edges:
        initial_neighbor_ids = []
        for dx, dy in initial_edges:
            new_pos = (start_node_pos[0] + dx, start_node_pos[1] + dy)
            merged_node = check_and_merge(G, new_pos, NODE_MERGE_DISTANCE)
            if merged_node is not None:
                 if not G.has_edge(root_node_id, merged_node):
                     G.add_edge(root_node_id, merged_node)
                 print(f"  Initial edge merged with existing node {merged_node}")
                 # Don't add merged node to initial queue if it already exists
            else:
                 new_node_id = next_node_id
                 G.add_node(new_node_id, pos=new_pos)
                 G.add_edge(root_node_id, new_node_id)
                 initial_neighbor_ids.append(new_node_id) # Collect new neighbors
                 next_node_id += 1
                 print(f"  Added initial node {new_node_id} at {new_pos}")
        # Add the *newly created* initial neighbors to the queue
        queue.extend(initial_neighbor_ids)
    else:
        # If no initial edges, the root node itself needs to be expanded
        queue.append(root_node_id)


    processed_nodes = 0
    gen_start_time = time.time()
    max_debug_nodes = 5 # Limit debug prints to first few nodes

    with torch.no_grad(): # Disable gradient calculations during inference
        while queue and G.number_of_nodes() < max_nodes:
            current_node_id = queue.pop(0) # Get the next node to expand (FIFO)
            processed_nodes += 1

            if current_node_id not in G.nodes or 'pos' not in G.nodes[current_node_id]:
                 print(f"Warning: Node {current_node_id} not found or missing pos in generation queue. Skipping.")
                 continue

            current_pos = G.nodes[current_node_id]['pos']

            if processed_nodes <= max_debug_nodes or processed_nodes % 50 == 0 : # Print debug info more often initially
                print(f"\nProcessing Node ID: {current_node_id} (Node {processed_nodes}) | Pos: {current_pos} | Graph Size: {G.number_of_nodes()}/{max_nodes} | Queue: {len(queue)}")

            # --- Prepare input for the model ---
            try:
                from data import sample_paths_for_node # Lazy import or pass function
                incoming_paths = sample_paths_for_node(G, current_node_id, K=K_gen, L=L_gen)
                if processed_nodes <= max_debug_nodes:
                    print(f"  Input Paths (K={K_gen}, L={L_gen}, Found={len(incoming_paths)}):")
                    for i, p in enumerate(incoming_paths[:3]): # Print first 3 paths
                         print(f"    Path {i+1} (len {len(p)}): {p[:2]}...{p[-2:]}" if len(p) > 4 else f"    Path {i+1} (len {len(p)}): {p}")
                    if len(incoming_paths) > 3: print("    ...")
            except ImportError:
                 print("  Warning: Could not import sample_paths_for_node from data.py. Using empty paths.")
                 incoming_paths = [] # Fallback: generate based on no history


            # --- Run the model (Inference) ---
            # Get latent vector first for debugging
            latent = model.encoder(incoming_paths)
            if processed_nodes <= max_debug_nodes:
                 print(f"  Encoder Latent Norm: {torch.linalg.norm(latent).item():.4f}")

            # Decode - Model's forward pass handles inference when target_deltas is None
            predicted_deltas = model.decoder(latent, target_deltas=None) # Returns list of (dx, dy)

            if processed_nodes <= max_debug_nodes:
                 print(f"  Predicted Deltas ({len(predicted_deltas)}): {predicted_deltas[:5]}") # Print first 5 deltas
                 if len(predicted_deltas) == 0:
                     print("  >>> Decoder predicted EOS immediately or generated no deltas.")

            # --- Process predicted deltas ---
            num_added_this_step = 0
            for dx, dy in predicted_deltas:
                if G.number_of_nodes() >= max_nodes:
                    print("  Max node limit reached during delta processing.")
                    break # Stop if max nodes reached

                # Basic check for non-zero delta to avoid trivial self-merge
                if dx == 0 and dy == 0:
                     if processed_nodes <= max_debug_nodes: print("  Skipping (0,0) delta.")
                     continue

                new_pos = (current_pos[0] + dx, current_pos[1] + dy)

                # Check proximity to existing nodes for merging (using the 5m threshold)
                merged_node_id = check_and_merge(G, new_pos, NODE_MERGE_DISTANCE)

                if merged_node_id is not None:
                    # If close to an existing node, add edge to it (if not already present and not self)
                    if not G.has_edge(current_node_id, merged_node_id) and current_node_id != merged_node_id:
                        G.add_edge(current_node_id, merged_node_id)
                        if processed_nodes <= max_debug_nodes: print(f"    Merged: Added edge ({current_node_id} -> {merged_node_id})")
                    # else:
                        # if processed_nodes <= max_debug_nodes: print(f"    Merge skipped: Edge ({current_node_id} -> {merged_node_id}) exists or is self.")
                else:
                    # If not merging, add a new node and edge
                    new_node_id = next_node_id
                    G.add_node(new_node_id, pos=new_pos)
                    G.add_edge(current_node_id, new_node_id)
                    queue.append(new_node_id) # Add the new node to the queue for expansion
                    next_node_id += 1
                    num_added_this_step += 1
                    if processed_nodes <= max_debug_nodes: print(f"    Added: Node {new_node_id} at {new_pos}, Edge ({current_node_id} -> {new_node_id})")

            if num_added_this_step == 0 and not predicted_deltas and processed_nodes <= max_debug_nodes:
                 print(f"  Node {current_node_id} produced no new nodes to add to queue.")
            # Optional: Add constraints check here (e.g., max degree for current_node_id)

    gen_duration = time.time() - gen_start_time
    print(f"\n--- Generation Finished ---")
    print(f"Time: {gen_duration:.2f}s")
    print(f"Processed {processed_nodes} nodes from queue.")
    print(f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"--------------------------")
    return G

def check_and_merge(G, target_pos, threshold):
    """
    Checks if target_pos is close to any existing node in G within the threshold.

    Args:
        G (nx.Graph): The current graph.
        target_pos (tuple[float, float]): The position to check.
        threshold (float): The distance threshold for merging.

    Returns:
        int or None: The ID of the closest node within the threshold, or None if no node is close enough.
    """
    min_dist_sq = threshold * threshold # Compare squared distances to avoid sqrt
    closest_node = None

    # Optimization: If graph is large, consider spatial indexing (e.g., k-d tree)
    # For moderate graphs, iterating is acceptable.
    tx, ty = target_pos
    for node_id, data in G.nodes(data=True):
        if 'pos' not in data: continue
        nx_node, ny_node = data['pos']
        dist_sq = (tx - nx_node)**2 + (ty - ny_node)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_node = node_id

    # Ensure the closest node found is actually within the threshold (redundant check, but safe)
    # And also handle the case where the closest node is the node itself (distance is 0)
    # We only want to merge with *other* existing nodes.
    # However, the check `current_node_id != merged_node_id` in the calling loop handles self-merging.
    # So just return the closest node if found within threshold.
    return closest_node

