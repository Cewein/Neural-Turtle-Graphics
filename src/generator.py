import torch
import networkx as nx
import math
import time
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional


from src.model import NTGModel
from src.utils.graph_utils import sample_paths_for_node, check_and_merge, Coord, NodeID
from src.utils.geometry_utils import calculate_vector_angle, do_intersect

logger = logging.getLogger(__name__)

def generate_graph(
    model: NTGModel,
    config: Dict[str, Any],
    device: torch.device,
    start_node_pos: Coord = (0.0, 0.0),
    initial_edges: Optional[List[Tuple[int, int]]] = None,
    generation_constraints: Optional[Dict[str, float]] = None
) -> nx.Graph:
    """
    Generates a road layout graph using the trained NTG model, enforcing constraints.
    # [Sec 3.2: Graph Generation], [Sec 3.5: Inference]

    Args:
        model (NTGModel): The trained model instance.
        config (Dict[str, Any]): Configuration dictionary containing generation parameters.
        device (torch.device): Device to run generation on ('cpu' or 'cuda').
        start_node_pos (Coord): Position of the initial root node.
        initial_edges (Optional[List[Tuple[int, int]]]): Optional initial displacements
                                                          relative to start_node_pos.
        generation_constraints (Optional[Dict[str, float]]): Dictionary with calculated
                                                              constraints, e.g.,
                                                              {'max_degree': float, 'min_angle_rad': float}.

    Returns:
        nx.Graph: The generated graph with 'pos' attributes. Returns an empty graph on failure.
    """
    model.to(device)
    model.eval() # Ensure model is in evaluation mode

    # Extract generation parameters from config
    max_nodes = config['generation']['max_nodes']
    K_gen = config['generation']['k_gen']
    L_gen = config['generation']['l_gen']
    node_merge_distance = config['generation']['node_merge_distance']
    # Max steps for the decoder during inference
    decoder_max_steps = config.get('model', {}).get('decoder_max_steps', 50) # Default if not in config

    G = nx.Graph()
    pos_dict: Dict[NodeID, Coord] = {} # Store positions separately for quick lookup
    next_node_id: int = 0

    # Initialize constraints with defaults if not provided or partially provided
    max_degree_constraint = float('inf')
    min_angle_constraint_rad = 0.0
    if generation_constraints:
        max_degree_constraint = generation_constraints.get('max_degree', float('inf'))
        # Ensure max_degree is at least 1 (or maybe 2?)
        max_degree_constraint = max(1.0, max_degree_constraint)
        min_angle_constraint_rad = generation_constraints.get('min_angle_rad', 0.0)
        logger.info(f"Applying Generation constraints: Max Degree <= {max_degree_constraint:.1f}, Min Angle >= {math.degrees(min_angle_constraint_rad):.2f}°")
    else:
        logger.warning("No generation constraints provided. Generation might produce invalid topology.")

    # --- Initialization ---
    # Add the root node
    root_node_id = next_node_id
    G.add_node(root_node_id)
    pos_dict[root_node_id] = start_node_pos
    next_node_id += 1
    logger.info(f"Generation starting with root node {root_node_id} at {start_node_pos}")

    queue: List[NodeID] = [] # FIFO queue of nodes to expand
    nodes_in_queue: set[NodeID] = set() # Track nodes currently in the queue

    # Add initial neighbors if specified [Sec 3.2 - Initialization]
    initial_nodes_added = []
    if initial_edges:
        logger.debug(f"Adding initial edges from seed: {initial_edges}")
        for dx, dy in initial_edges:
            if dx == 0 and dy == 0: continue
            new_pos = (start_node_pos[0] + dx, start_node_pos[1] + dy)

            # Check merge only against the root initially
            if math.hypot(new_pos[0] - start_node_pos[0], new_pos[1] - start_node_pos[1]) < node_merge_distance:
                 logger.warning(f"Initial edge ({dx},{dy}) too close to root. Skipping.")
                 continue

            # Check degree constraint for root node
            if G.degree(root_node_id) + 1 > max_degree_constraint:
                 logger.warning(f"Adding initial edge ({dx},{dy}) would exceed max degree for root. Skipping.")
                 continue

            # Add the new node and edge (angle/intersection checks less critical here)
            new_node_id = next_node_id
            G.add_node(new_node_id)
            pos_dict[new_node_id] = new_pos
            G.add_edge(root_node_id, new_node_id)
            initial_nodes_added.append(new_node_id)
            next_node_id += 1

        # Add the *newly created* initial neighbors to the queue
        for nid in initial_nodes_added:
            if nid not in nodes_in_queue:
                queue.append(nid)
                nodes_in_queue.add(nid)
        logger.info(f"Added {len(initial_nodes_added)} initial nodes to the queue.")
    else:
        # If no initial edges, the root node itself needs to be expanded first
        if max_nodes > 1 and root_node_id not in nodes_in_queue:
            logger.info("No initial edges provided, adding root node to queue.")
            queue.append(root_node_id)
            nodes_in_queue.add(root_node_id)

    # --- Generation Loop ---
    processed_nodes_count = 0
    gen_start_time = time.time()
    nodes_rejected_degree = 0
    nodes_rejected_angle = 0
    nodes_rejected_intersection = 0

    # Add tqdm progress bar for the generation loop
    pbar_gen = tqdm(total=max_nodes, desc="Generating Nodes", unit="node")
    pbar_gen.update(G.number_of_nodes()) # Initial node count

    with torch.no_grad(): # Disable gradient calculations during inference
        while queue and G.number_of_nodes() < max_nodes:
            # Get the next node to expand from the queue (FIFO) [Sec 3.2]
            current_node_id = queue.pop(0)
            nodes_in_queue.remove(current_node_id)
            processed_nodes_count += 1

            # Basic checks
            if current_node_id not in G: continue # Node might have been removed if merged
            if current_node_id not in pos_dict:
                 logger.warning(f"Node {current_node_id} popped from queue but missing position. Skipping.")
                 continue

            current_pos = pos_dict[current_node_id]

            # --- Prepare input for the model ---
            # Sample incoming paths using the current graph state [Sec 3.2]
            # Need to temporarily set 'pos' on G for sample_paths_for_node
            nx.set_node_attributes(G, pos_dict, 'pos')
            incoming_paths = sample_paths_for_node(G, current_node_id, K=K_gen, L=L_gen)
            # Remove temporary attribute after use if desired, or leave it
            # for node in G.nodes: G.nodes[node].pop('pos', None)

            # --- Run the model (Inference) --- [Sec 3.2]
            # Pass max_steps to decoder
            predicted_deltas = model(incoming_paths, target_deltas=None, max_steps=decoder_max_steps)

            # --- Process predicted deltas ---
            current_degree = G.degree(current_node_id)

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

            nodes_added_from_current = 0
            for dx, dy in predicted_deltas:
                if G.number_of_nodes() >= max_nodes:
                    logger.info("Reached max_nodes limit during delta processing.")
                    break # Stop adding more nodes if limit reached

                if dx == 0 and dy == 0:
                     # logger.debug(f"Skipping zero delta prediction for node {current_node_id}.")
                     continue

                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                new_vector = (dx, dy)

                # --- Check for merging with existing nodes --- [Sec 3.5]
                # Use the efficient pos_dict version
                merged_node_id = check_and_merge(pos_dict, new_pos, node_merge_distance)

                # --- Intersection Check (Planarity) ---
                # Check if the potential new edge (current_node -> new_pos/merged_node) intersects existing edges
                intersects = False
                p1 = current_pos
                # Determine the endpoint of the potential new edge
                q1 = new_pos if merged_node_id is None else pos_dict[merged_node_id]

                # Iterate through existing edges in the graph
                for u, v in G.edges():
                    # Skip if the existing edge involves the current node or the merge target
                    if u == current_node_id or v == current_node_id: continue
                    if merged_node_id is not None and (u == merged_node_id or v == merged_node_id): continue

                    # Get positions of the existing edge endpoints
                    if u in pos_dict and v in pos_dict:
                        p2 = pos_dict[u]
                        q2 = pos_dict[v]

                        # Check intersection using the utility function
                        # Add a small tolerance check? Maybe not needed if positions are precise enough.
                        # Avoid checking intersection if endpoints are identical (p1=p2, q1=q2 etc.) - do_intersect handles collinearity
                        if do_intersect(p1, q1, p2, q2):
                             # Check if intersection is *only* at a shared endpoint (p1=p2, p1=q2, q1=p2, q1=q2)
                             # This is allowed. do_intersect should handle collinear cases correctly.
                             # If intersection is found and it's not just at a shared endpoint, reject.
                             # Let's trust do_intersect for now.
                             logger.debug(f"Intersection detected: ({p1} -> {q1}) with ({p2} -> {q2})")
                             intersects = True
                             nodes_rejected_intersection += 1
                             break # Stop checking other edges if one intersection found
                    # else: logger.warning(f"Edge ({u},{v}) missing position data during intersection check.")

                if intersects:
                    # logger.debug(f"Skipping delta ({dx},{dy}) due to intersection.")
                    continue # Skip this delta if it causes intersection
                # --- End Intersection Check ---


                # --- Constraint Checks (Degree, Angle) --- [Sec 3.5]
                # 1. Max Degree Check
                # Check degree of current node
                if G.degree(current_node_id) + 1 > max_degree_constraint:
                    nodes_rejected_degree += 1
                    # logger.debug(f"Skipping delta for node {current_node_id}: exceeds max degree ({max_degree_constraint}).")
                    continue # Skip if adding edge exceeds limit for current node
                # Check degree of merge target node (if applicable)
                if merged_node_id is not None and merged_node_id != current_node_id:
                     if G.degree(merged_node_id) + 1 > max_degree_constraint:
                         nodes_rejected_degree += 1
                         # logger.debug(f"Skipping delta for node {current_node_id}: merge target {merged_node_id} exceeds max degree.")
                         continue # Skip if adding edge exceeds limit for merge target

                # 2. Min Angle Check
                # Check angle between new edge and existing edges at current_node
                if existing_vectors: # Only check if there are existing edges
                    valid_angle_current = True
                    for exist_vec in existing_vectors:
                        angle_rad = calculate_vector_angle(new_vector, exist_vec)
                        # Use a small tolerance for floating point comparisons
                        if angle_rad < min_angle_constraint_rad - 1e-6:
                             nodes_rejected_angle += 1
                             valid_angle_current = False
                             # logger.debug(f"Skipping delta for node {current_node_id}: violates min angle ({math.degrees(angle_rad):.1f}° < {math.degrees(min_angle_constraint_rad):.1f}°).")
                             break
                    if not valid_angle_current:
                        continue

                # Check angle at the merge target node (if applicable)
                if merged_node_id is not None and merged_node_id != current_node_id:
                    merge_node_pos = pos_dict[merged_node_id]
                    # Vector from merge_node to current_node
                    vec_to_current = (current_pos[0] - merge_node_pos[0], current_pos[1] - merge_node_pos[1])
                    if vec_to_current[0] == 0 and vec_to_current[1] == 0: continue # Avoid zero vector

                    valid_angle_merge = True
                    for nbr_of_merge in G.neighbors(merged_node_id):
                        if nbr_of_merge == current_node_id: continue # Don't compare with the edge we are adding
                        if nbr_of_merge in pos_dict:
                            nbr_pos = pos_dict[nbr_of_merge]
                            existing_vec_at_merge = (nbr_pos[0] - merge_node_pos[0], nbr_pos[1] - merge_node_pos[1])
                            if existing_vec_at_merge[0] == 0 and existing_vec_at_merge[1] == 0: continue

                            angle_rad = calculate_vector_angle(vec_to_current, existing_vec_at_merge)
                            if angle_rad < min_angle_constraint_rad - 1e-6:
                                nodes_rejected_angle += 1
                                valid_angle_merge = False
                                # logger.debug(f"Skipping delta for node {current_node_id}: violates min angle at merge target {merged_node_id}.")
                                break
                    if not valid_angle_merge:
                        continue
                # --- End Constraint Checks ---


                # --- Add edge/node if all checks passed ---
                nodes_added_this_step = 0 # Track nodes added in this inner loop
                if merged_node_id is not None:
                    # Add edge to existing node, avoiding self-loops
                    if not G.has_edge(current_node_id, merged_node_id) and current_node_id != merged_node_id:
                        G.add_edge(current_node_id, merged_node_id)
                        # Update vectors for subsequent checks in this step
                        existing_vectors.append(new_vector)
                        # logger.debug(f"Added edge: {current_node_id} -> {merged_node_id} (merged)")
                else:
                    # Add a new node and edge
                    new_node_id = next_node_id
                    G.add_node(new_node_id)
                    pos_dict[new_node_id] = new_pos
                    G.add_edge(current_node_id, new_node_id)
                    nodes_added_this_step += 1
                    pbar_gen.update(1) # Update progress bar for new node

                    # Add the new node to the queue for expansion
                    if new_node_id not in nodes_in_queue:
                         queue.append(new_node_id)
                         nodes_in_queue.add(new_node_id)

                    next_node_id += 1
                    # Update vectors for subsequent checks in this step
                    existing_vectors.append(new_vector)
                    # logger.debug(f"Added node {new_node_id} at {new_pos} and edge {current_node_id} -> {new_node_id}")

            # If the queue becomes empty but we haven't reached max_nodes, maybe stop?
            # The loop condition `while queue and ...` handles this.

        # End of generation loop
        pbar_gen.close()

    gen_duration = time.time() - gen_start_time
    logger.info("--- Graph Generation Finished ---")
    logger.info(f"Time: {gen_duration:.2f}s")
    logger.info(f"Processed {processed_nodes_count} nodes from queue.")
    logger.info(f"Nodes rejected by degree constraint: {nodes_rejected_degree}")
    logger.info(f"Nodes rejected by angle constraint: {nodes_rejected_angle}")
    logger.info(f"Nodes rejected by intersection: {nodes_rejected_intersection}")

    # Set final positions on the graph object before returning
    nx.set_node_attributes(G, pos_dict, 'pos')
    logger.info(f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    logger.info("---------------------------------")

    # Final check for empty graph
    if G.number_of_nodes() == 0:
        logger.error("Generation resulted in an empty graph.")

    return G
