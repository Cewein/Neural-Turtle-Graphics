import math
import networkx as nx
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
import random

from src.utils.graph_utils import sample_paths_for_node, Coord, NodeID

logger = logging.getLogger(__name__)

# Type alias for a single training sample
# (Incoming paths coordinates, Outgoing deltas)
TrainingSample = Tuple[List[List[Coord]], List[Tuple[int, int]]]

def prepare_training_data_from_graphs(
    graphs: List[nx.Graph],
    config: Dict[str, Any]
) -> List[TrainingSample]:
    """
    Prepare training samples from a list of NetworkX graphs based on config.
    Assumes graphs have nodes with 'pos' attribute containing (x, y) coordinates in meters.
    Uses random walk sampling for incoming paths.
    # [Sec 3.5: Learning - Data Preparation]

    Args:
        graphs (List[nx.Graph]): List of graphs loaded (e.g., from OSM).
        config (Dict[str, Any]): Configuration dictionary containing parameters like
                                 'paths.k_paths', 'paths.l_paths', 'model.max_displacement'.

    Returns:
        List[TrainingSample]: A list of training samples.
    """
    data: List[TrainingSample] = []
    total_nodes_processed = 0
    nodes_with_neighbors = 0
    total_deltas_processed = 0
    deltas_clamped_count = 0

    # Extract necessary parameters from config
    K = config['paths']['k_paths']
    L = config['paths']['l_paths']
    MAX_DISPLACEMENT = config['model']['max_displacement']

    logger.info(f"Preparing training data with K={K}, L={L}, Max_Displacement={MAX_DISPLACEMENT}")

    # Add tqdm progress bar for processing graphs
    graph_iterator = tqdm(graphs, desc="Processing Graphs", unit="graph")
    for G_idx, G in enumerate(graph_iterator):
        graph_iterator.set_postfix({"Graph Nodes": G.number_of_nodes(), "Edges": G.number_of_edges()})

        if not G or G.number_of_nodes() == 0:
             logger.warning(f"Skipping empty graph {G_idx+1}.")
             continue

        pos_dict = nx.get_node_attributes(G, 'pos')
        if not pos_dict:
            logger.warning(f"Graph {G_idx+1} has no 'pos' attributes. Cannot process.")
            continue

        processed_nodes_in_graph = 0
        # Add tqdm progress bar for processing nodes within a graph
        node_iterator = tqdm(G.nodes(), desc=f"  Graph {G_idx+1} Nodes", unit="node", leave=False)
        for node in node_iterator:
            total_nodes_processed += 1
            processed_nodes_in_graph += 1
            # node_iterator.set_postfix({"Processed": f"{processed_nodes_in_graph}/{G.number_of_nodes()}"})

            # Check if node has position data
            if node not in pos_dict:
                 # logger.debug(f"Node {node} in graph {G_idx+1} missing 'pos'. Skipping.")
                 continue

            neighbors = list(G.neighbors(node))
            if not neighbors:
                # logger.debug(f"Node {node} in graph {G_idx+1} is isolated. Skipping.")
                continue # Skip isolated nodes (no outgoing edges to predict)

            nodes_with_neighbors += 1

            # Sample incoming paths using random walks [Sec 3.2]
            inc_paths_coords = sample_paths_for_node(G, node, K=K, L=L)

            # Determine outgoing edges (deltas) sorted by angle around the node [Sec 3.5]
            # Calculate deltas relative to the current node's position
            base_x, base_y = pos_dict[node]
            angle_delta_pairs = []
            for nbr in neighbors:
                 if nbr not in pos_dict:
                     logger.warning(f"Neighbor {nbr} of node {node} missing 'pos'. Skipping delta.")
                     continue

                 nbr_x, nbr_y = pos_dict[nbr]
                 dx_float = nbr_x - base_x
                 dy_float = nbr_y - base_y

                 # Calculate angle (counter-clockwise from positive x-axis)
                 angle = math.atan2(dy_float, dx_float)
                 # Normalize angle to [0, 2*pi)
                 if angle < 0:
                     angle += 2 * math.pi
                 angle_delta_pairs.append((angle, dx_float, dy_float))

            # Sort neighbors by angle (counter-clockwise)
            angle_delta_pairs.sort(key=lambda x: x[0])

            # Clamp displacements and convert to integers
            outgoing_deltas = []
            for _, dx_float, dy_float in angle_delta_pairs:
                total_deltas_processed += 2 # Counting dx and dy

                dx_clamped = int(round(dx_float))
                dy_clamped = int(round(dy_float))

                # Check for clamping before applying it
                clamped = False
                if dx_clamped > MAX_DISPLACEMENT or dx_clamped < -MAX_DISPLACEMENT:
                    deltas_clamped_count += 1
                    clamped = True
                if dy_clamped > MAX_DISPLACEMENT or dy_clamped < -MAX_DISPLACEMENT:
                    # Avoid double counting if both were clamped
                    if not clamped or (dx_clamped <= MAX_DISPLACEMENT and dx_clamped >= -MAX_DISPLACEMENT) :
                         deltas_clamped_count += 1

                # Apply clamping [Sec 3.4 - Discrete Representation]
                dx_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dx_clamped))
                dy_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dy_clamped))

                # Only add non-zero deltas (model predicts relative movement)
                if dx_clamped != 0 or dy_clamped != 0:
                    outgoing_deltas.append((dx_clamped, dy_clamped))
                # else:
                #     logger.debug(f"Skipping zero delta for node {node}, neighbor at same location?")


            # Add the sample (only if there are actual outgoing deltas to predict)
            # We include samples even if inc_paths is empty, allowing the model
            # to learn generation from minimal or no history (e.g., starting nodes).
            if outgoing_deltas:
                 data.append((inc_paths_coords, outgoing_deltas))

        logger.debug(f"Finished processing graph {G_idx+1}. Found {len(data)} samples so far.")

    logger.info(f"--- Data Preparation Summary ---")
    logger.info(f"Total nodes processed across all graphs: {total_nodes_processed}")
    logger.info(f"Nodes with neighbors (potential samples): {nodes_with_neighbors}")
    logger.info(f"Total training samples generated: {len(data)}")

    if total_deltas_processed > 0:
        clamping_percentage = (deltas_clamped_count / total_deltas_processed) * 100
        logger.info(f"Delta Clamping Statistics (Range [-{MAX_DISPLACEMENT}, {MAX_DISPLACEMENT}]):")
        logger.info(f"  Total delta values processed (dx + dy): {total_deltas_processed}")
        logger.info(f"  Number of delta values clamped: {deltas_clamped_count}")
        logger.info(f"  Clamping Percentage: {clamping_percentage:.2f}%")
    else:
        logger.info("No deltas processed, cannot calculate clamping statistics.")

    if not data:
        logger.error("No training data was generated. Check graph structures, parameters (K, L), and OSM files.")
    else:
        # Log an example sample
        sample_idx = min(5, len(data)-1) # Log one of the first few samples
        if sample_idx >= 0:
            example_in_paths, example_out_deltas = data[sample_idx]
            logger.debug("Example Training Sample:")
            logger.debug(f"  Incoming Paths (K={K}, L={L}, {len(example_in_paths)} sampled):")
            for i, p in enumerate(example_in_paths[:3]): # Log max 3 paths for brevity
                 path_str = ", ".join([f"({c[0]:.1f},{c[1]:.1f})" for c in p])
                 logger.debug(f"    Path {i+1} (len {len(p)}): [{path_str[:100]}{'...' if len(path_str)>100 else ''}]")
            if len(example_in_paths) > 3: logger.debug("    ...")
            logger.debug(f"  Outgoing Deltas (Sorted CCW, Clamped [-{MAX_DISPLACEMENT},{MAX_DISPLACEMENT}]): {example_out_deltas}")
    logger.info("--- Data Preparation Finished ---")

    return data

# --- Synthetic graph generator for testing/comparison if needed ---
def generate_synthetic_graphs(num_graphs: int = 5, grid_size: int = 5, spacing: float = 20.0, random_extra_edges: int = 2) -> List[nx.Graph]:
    """
    Generate a list of synthetic planar graphs (grids with random edges).
    Nodes are labeled by grid coordinates (i,j) with positions (i*spacing, j*spacing).
    """
    logger.info(f"Generating {num_graphs} synthetic grid graphs ({grid_size}x{grid_size})...")
    graphs = []
    for graph_idx in range(num_graphs):
        G = nx.Graph()
        pos_dict = {}
        # Create grid nodes and edges (4-neighbor connectivity)
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = (i, j)
                node_pos = (i * spacing, j * spacing)
                G.add_node(node_id)
                pos_dict[node_id] = node_pos
                # Add edges to neighbors if they exist (handle boundaries)
                for di, dj in [(0, 1), (1, 0)]: # Check right and up
                    neighbor_id = (i + di, j + dj)
                    if 0 <= i + di < grid_size and 0 <= j + dj < grid_size:
                        G.add_edge(node_id, neighbor_id)

        nx.set_node_attributes(G, pos_dict, 'pos') # Set positions

        # Add random extra edges (optional)
        nodes = list(G.nodes())
        added_edges = 0
        attempts = 0
        max_attempts = random_extra_edges * 10 * grid_size # Heuristic limit

        while added_edges < random_extra_edges and attempts < max_attempts and len(nodes) >= 2:
            attempts += 1
            u, v = random.sample(nodes, 2)

            if not G.has_edge(u, v):
                # Optional: Add only if nodes are relatively close
                ux, uy = G.nodes[u]['pos']
                vx, vy = G.nodes[v]['pos']
                # Add edge if distance is roughly diagonal or less
                if math.hypot(vx - ux, vy - uy) < 1.5 * spacing:
                    G.add_edge(u, v)
                    added_edges += 1
        graphs.append(G)
        logger.debug(f"Generated synthetic graph {graph_idx+1} with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    logger.info(f"Finished generating {len(graphs)} synthetic graphs.")
    return graphs
