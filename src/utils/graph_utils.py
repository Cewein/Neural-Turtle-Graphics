# utils/graph_utils.py
import math
import random
import networkx as nx
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

# Type alias for coordinates
Coord = Tuple[float, float]
NodeID = Union[int, str, Tuple] # Node IDs can be various types

def _perform_random_walk(G: nx.Graph, start_node: NodeID, max_length_L: int) -> Optional[List[NodeID]]:
    """
    Performs a single random walk backwards from start_node up to max_length_L.
    Ensures the walk is acyclic within the path itself.
    # [Sec 3.2: Sampling Incoming Paths]

    Args:
        G (nx.Graph): The input graph (must be undirected).
        start_node (NodeID): The node where the incoming path terminates (walk starts here).
        max_length_L (int): Maximum number of edges (steps) in the path.

    Returns:
        Optional[List[NodeID]]: A list of node IDs representing the path
                                (from ancestor to start_node), or None if a valid
                                path cannot be formed (e.g., isolated start_node).
                                Returns path even if shorter than max_length_L.
    """
    if start_node not in G:
        logger.warning(f"Start node {start_node} not found in graph during random walk.")
        return None

    path_nodes_rev = [start_node] # Path stored in reverse (start_node -> ancestor)
    current_node = start_node

    for _ in range(max_length_L): # Iterate up to L steps (edges)
        neighbors = list(G.neighbors(current_node))
        if not neighbors: # Dead end
             break

        # Filter neighbors: exclude nodes already in the current path to ensure acyclicity.
        valid_neighbors = [n for n in neighbors if n not in path_nodes_rev]

        if not valid_neighbors:
            # Only way back is into the path already traversed, stop here.
            break

        # Choose the next node randomly from valid neighbors
        next_node = random.choice(valid_neighbors)
        path_nodes_rev.append(next_node)
        current_node = next_node

    if len(path_nodes_rev) < 2: # Path needs at least two nodes (one edge)
        # This happens if the start node was isolated or only connected back to itself immediately
        # logger.debug(f"Random walk from {start_node} resulted in path < 2 nodes.")
        return None

    # Return the path in the correct order (ancestor -> ... -> start_node)
    return list(reversed(path_nodes_rev))


def sample_paths_for_node(G: nx.Graph, node: NodeID, K: int, L: int, max_attempts_factor: int = 5) -> List[List[Coord]]:
    """
    Sample up to K unique, acyclic incoming paths (length <= L) that terminate at 'node'
    using random walks starting from 'node' and moving backwards.
    # [Sec 3.2: Sampling Incoming Paths], [Sec 3.5: Learning]

    Args:
        G (nx.Graph): The input graph with 'pos' attributes for nodes.
        node (NodeID): The target node to find incoming paths for.
        K (int): Maximum number of unique paths to sample.
        L (int): Maximum length (number of edges) of each path.
        max_attempts_factor (int): Factor to determine max attempts (K * factor)
                                   to find unique paths.

    Returns:
        List[List[Coord]]: A list of unique paths, where each path is a list of
                           (x, y) coordinates. Returns fewer than K if walks fail
                           or duplicates are frequent.
    """
    if node not in G:
        logger.warning(f"Node {node} not in graph for path sampling.")
        return []

    pos_dict = nx.get_node_attributes(G, 'pos')
    if not pos_dict:
        logger.error("Graph missing 'pos' attributes, cannot sample coordinate paths.")
        return []
    if node not in pos_dict:
         logger.warning(f"Node {node} missing 'pos' attribute, cannot sample paths.")
         return []


    unique_paths_nodes_set = set() # Store tuples of node IDs to ensure uniqueness
    max_attempts = K * max_attempts_factor
    attempts = 0

    while len(unique_paths_nodes_set) < K and attempts < max_attempts:
        attempts += 1
        path_nodes = _perform_random_walk(G, node, L)

        if path_nodes:
            # Add the tuple representation of the node path to the set
            unique_paths_nodes_set.add(tuple(path_nodes))

    # Convert unique node paths to coordinate paths
    final_paths_coords = []
    for p_nodes_tuple in unique_paths_nodes_set:
        path_coords = []
        valid_path = True
        for n_id in p_nodes_tuple:
            # Check if node still exists and has position
            if n_id in pos_dict:
                path_coords.append(pos_dict[n_id])
            else:
                logger.warning(f"Node {n_id} in sampled path lacks 'pos'. Skipping path {p_nodes_tuple}.")
                valid_path = False
                break
        if valid_path and len(path_coords) >= 2: # Ensure path has at least one edge
            final_paths_coords.append(path_coords)

    if len(final_paths_coords) < K and attempts == max_attempts:
        logger.debug(f"Reached max attempts ({max_attempts}) sampling paths for node {node}, found {len(final_paths_coords)}/{K} unique paths.")

    return final_paths_coords


def calculate_graph_statistics(graphs: List[nx.Graph], degree_percentile: float = 99.0, angle_percentile: float = 1.0) -> Dict[str, float]:
    """
    Calculates statistics (max degree, min angle) from training graphs for generation constraints.
    # [Sec 3.5: Inference - discusses constraints]

    Args:
        graphs (List[nx.Graph]): List of training graphs.
        degree_percentile (float): Percentile for maximum allowed node degree.
        angle_percentile (float): Percentile for minimum allowed angle between incident edges (in degrees).

    Returns:
        Dict[str, float]: A dictionary containing constraint thresholds:
                          {'max_degree': float, 'min_angle_rad': float}
                          Returns default constraints if stats cannot be computed.
    """
    logger.info("Calculating statistics from training graphs for generation constraints...")
    all_degrees = []
    min_angles_at_nodes_rad = [] # Store the minimum angle (radians) found at each node with degree >= 2

    default_constraints = {'max_degree': 5.0, 'min_angle_rad': math.radians(10.0)} # Default fallbacks

    if not graphs:
        logger.warning("No graphs provided for statistics calculation. Returning default constraints.")
        return default_constraints

    # Local import to avoid circular dependency if geometry_utils imports this
    try:
        from .geometry_utils import calculate_vector_angle
    except ImportError:
        logger.error("Could not import calculate_vector_angle. Angle constraints will use default.")
        calculate_vector_angle = None # Set to None to skip angle calculation


    pos_missing_count = 0
    nodes_processed = 0
    nodes_with_angles = 0

    for G_idx, G in enumerate(graphs):
        if not G or G.number_of_nodes() == 0:
            continue
        pos_dict = nx.get_node_attributes(G, 'pos')
        if not pos_dict:
            logger.warning(f"Graph {G_idx} missing 'pos' attributes. Skipping statistics calculation for this graph.")
            continue

        # logger.debug(f"Processing graph {G_idx+1}/{len(graphs)} for stats...")
        for node in G.nodes():
            nodes_processed += 1
            degree = G.degree(node)
            all_degrees.append(degree)

            # Calculate minimum angle only if possible and needed
            if calculate_vector_angle and degree >= 2:
                if node not in pos_dict:
                     pos_missing_count += 1
                     continue

                node_pos = pos_dict[node]
                neighbor_vectors = []
                valid_neighbors = 0
                for nbr in G.neighbors(node):
                    if nbr in pos_dict:
                        nbr_pos = pos_dict[nbr]
                        vec = (nbr_pos[0] - node_pos[0], nbr_pos[1] - node_pos[1])
                        # Ensure vector is not zero length
                        if vec[0] != 0 or vec[1] != 0:
                           neighbor_vectors.append(vec)
                           valid_neighbors += 1
                    else:
                         pos_missing_count += 1

                if valid_neighbors >= 2:
                    nodes_with_angles += 1
                    min_angle_for_node_rad = math.pi # Initialize with max possible angle (180 deg)
                    # Calculate angle between all pairs of incident edge vectors
                    for i in range(len(neighbor_vectors)):
                        for j in range(i + 1, len(neighbor_vectors)):
                            angle_rad = calculate_vector_angle(neighbor_vectors[i], neighbor_vectors[j])
                            min_angle_for_node_rad = min(min_angle_for_node_rad, angle_rad)

                    # Only add valid angles (avoiding potential issues if min_angle_for_node remains pi)
                    if min_angle_for_node_rad < math.pi:
                         min_angles_at_nodes_rad.append(min_angle_for_node_rad)

    if pos_missing_count > 0:
         logger.warning(f"Encountered {pos_missing_count} missing node positions during angle calculation.")

    if not all_degrees:
        logger.warning("Could not calculate degree statistics. Returning default constraints.")
        return default_constraints

    # Calculate thresholds using percentiles
    # Use np.percentile for robustness, convert result to float
    max_degree_threshold = float(np.percentile(all_degrees, degree_percentile)) if all_degrees else default_constraints['max_degree']
    min_angle_threshold_rad = float(np.percentile(min_angles_at_nodes_rad, angle_percentile)) if min_angles_at_nodes_rad else default_constraints['min_angle_rad']

    # Convert min angle to degrees for logging
    min_angle_threshold_deg = math.degrees(min_angle_threshold_rad)

    logger.info(f"Statistics calculated from {len(graphs)} graphs ({nodes_processed} nodes):")
    logger.info(f"  Node Degrees: Min={np.min(all_degrees)}, Max={np.max(all_degrees)}, Mean={np.mean(all_degrees):.2f}, {degree_percentile}th percentile={max_degree_threshold:.2f}")
    if min_angles_at_nodes_rad and calculate_vector_angle:
        logger.info(f"  Min Angles (at {nodes_with_angles} nodes, radians): Min={np.min(min_angles_at_nodes_rad):.3f}, Max={np.max(min_angles_at_nodes_rad):.3f}, Mean={np.mean(min_angles_at_nodes_rad):.3f}")
        logger.info(f"  Min Angle Threshold ({angle_percentile}th percentile): {min_angle_threshold_deg:.2f}° ({min_angle_threshold_rad:.3f} rad)")
    else:
        logger.warning("Could not calculate angle statistics (no nodes with degree >= 2 or issues found). Using default.")

    constraints = {
        'max_degree': max_degree_threshold,
        'min_angle_rad': min_angle_threshold_rad
    }
    logger.info(f"Calculated Generation Constraints: {constraints}")
    return constraints

def check_and_merge(pos_dict: Dict[NodeID, Coord], target_pos: Coord, threshold: float) -> Optional[NodeID]:
    """
    Checks if target_pos is close to any existing node position in pos_dict.
    Returns the ID of the closest node within the threshold, or None.
    # [Sec 3.5: Inference - Node Merging]

    Args:
        pos_dict (Dict[NodeID, Coord]): Dictionary mapping node IDs to (x, y) positions.
        target_pos (Coord): The (x, y) position to check.
        threshold (float): The maximum distance to consider nodes "close".

    Returns:
        Optional[NodeID]: The ID of the closest node within the threshold, or None.
    """
    min_dist_sq = threshold * threshold # Compare squared distances for efficiency
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
