import os
import osmnx as ox
import networkx as nx
import logging
import warnings
from typing import List, Optional, Tuple

# Setup logger for this module
logger = logging.getLogger(__name__)

# Configure osmnx logging and settings
# Reduce osmnx verbosity by default, can be overridden by root logger config
logging.getLogger("osmnx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module='osmnx')
ox.settings.log_console = False # Don't let osmnx log to console directly
ox.settings.use_cache = True

def get_drive_filter() -> List[str]:
    """
    Returns a list of standard OSM highway tags considered drivable by osmnx.
    Based on osmnx.settings.useful_tags_way and common driving network types.
    Excludes footways, cycleways, steps, paths, etc.
    # [Ref: OSM highway tag documentation]
    """
    # Based on typical 'drive' network filters in osmnx
    drive_tags = [
        "motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        "living_street", "service", # Service roads are often drivable
    ]
    # logger.debug(f"Using drive filter tags: {drive_tags}")
    return drive_tags

def filter_graph_by_highway_tags(G: nx.MultiDiGraph, allowed_tags: List[str]) -> nx.MultiDiGraph:
    """
    Filters a graph, keeping only edges whose 'highway' tag is in the allowed_tags list.

    Args:
        G (nx.MultiDiGraph): The graph to filter (typically from osmnx).
        allowed_tags (List[str]): List of OSM highway tag values to keep.

    Returns:
        nx.MultiDiGraph: A new graph containing only the filtered edges and their nodes.
                         Returns an empty graph if no edges match.
    """
    edges_to_keep = []
    original_edge_count = G.number_of_edges()
    logger.debug(f"Filtering graph with {original_edge_count} edges by highway tags...")

    for u, v, key, data in G.edges(keys=True, data=True):
        highway_tag = data.get('highway', None)
        keep_edge = False
        if isinstance(highway_tag, list):
            # Keep edge if any tag in the list is allowed
            if any(tag in allowed_tags for tag in highway_tag):
                keep_edge = True
        elif isinstance(highway_tag, str):
            # Keep edge if the single tag is allowed
            if highway_tag in allowed_tags:
                keep_edge = True

        if keep_edge:
            edges_to_keep.append((u, v, key))

    # Create a new graph containing only the edges to keep
    # Using edge_subgraph preserves node attributes but only includes nodes incident to kept edges
    G_filtered = G.edge_subgraph(edges_to_keep).copy()

    removed_count = original_edge_count - G_filtered.number_of_edges()
    logger.info(f"Filtered graph by highway tags. Kept {G_filtered.number_of_edges()} / {original_edge_count} edges (Removed {removed_count}).")

    return G_filtered

def graph_from_osm(filepath: str, network_type: str = 'drive', simplify: bool = True) -> Optional[nx.Graph]:
    """
    Loads and filters a road network graph from an .osm file for a specific network type.

    Args:
        filepath (str): Path to the .osm file.
        network_type (str): Type of network to extract (e.g., 'drive', 'walk', 'bike', 'all').
                            Currently only 'drive' filtering is implemented robustly.
        simplify (bool): If True, simplify the graph topology *before* filtering.

    Returns:
        Optional[nx.Graph]: An undirected graph representing the filtered road network.
                            Nodes have 'pos' attributes (projected coordinates in meters).
                            Returns None if loading or filtering fails or results in an empty graph.
    """
    logger.info(f"Loading OSM data from: {filepath}")
    try:
        # Load the graph from the .osm file using graph_from_xml
        G_initial = ox.graph_from_xml(filepath, simplify=simplify)
        logger.info(f"Initial load (simplified={simplify}): {G_initial.number_of_nodes()} nodes, {G_initial.number_of_edges()} edges.")

        if G_initial.number_of_nodes() == 0:
             logger.warning(f"Initial graph loaded from {filepath} is empty.")
             return None

        # Project the graph to a suitable UTM zone to get coordinates in meters
        G_proj = ox.project_graph(G_initial)
        logger.info(f"Projected graph: {G_proj.number_of_nodes()} nodes, {G_proj.number_of_edges()} edges.")

        if network_type.lower() == 'drive':
            allowed_highway_tags = get_drive_filter()
            logger.info(f"Applying 'drive' network filter.")
            G_filtered = filter_graph_by_highway_tags(G_proj, allowed_highway_tags)
        # TODO: Implement filters for other network types ('walk', 'bike') if needed
        # elif network_type.lower() == 'walk':
        #     allowed_highway_tags = get_walk_filter() # Define this function
        #     G_filtered = filter_graph_by_highway_tags(G_proj, allowed_highway_tags)
        else:
            logger.warning(f"Network type '{network_type}' filtering not implemented or requested. Using projected graph as is.")
            G_filtered = G_proj # Use the projected graph without filtering

        if G_filtered.number_of_nodes() == 0:
             logger.warning(f"Graph became empty after filtering for network type '{network_type}'.")
             return None

        # Ensure graph is undirected for NTG (paper uses undirected graph G={V,E}) [Sec 3.1]
        logger.info("Converting filtered graph to undirected.")
        # Use to_undirected() which handles parallel edges better than nx.Graph() constructor
        G_undirected = G_filtered.to_undirected()

        # Extract node positions (x, y coordinates in meters) from the projected graph
        pos_dict = {}
        nodes_missing_pos = 0
        for node, data in G_proj.nodes(data=True):
             # Check if the node survived filtering and conversion to undirected
             if node in G_undirected:
                if 'x' in data and 'y' in data:
                    pos_dict[node] = (float(data['x']), float(data['y'])) # Ensure float
                else:
                     logger.warning(f"Node {node} missing x/y coordinates after projection.")
                     nodes_missing_pos +=1

        if nodes_missing_pos > 0:
             logger.warning(f"{nodes_missing_pos} nodes missing x/y coords in projected data.")

        # Set the 'pos' attribute on the final undirected graph
        nx.set_node_attributes(G_undirected, pos_dict, 'pos')

        # Remove nodes that ended up without a 'pos' attribute (should be rare)
        nodes_to_remove = [node for node, data in G_undirected.nodes(data=True) if 'pos' not in data]
        if nodes_to_remove:
            logger.warning(f"Removing {len(nodes_to_remove)} nodes without position data after processing.")
            G_undirected.remove_nodes_from(nodes_to_remove)

        # Keep only the largest connected component of the *final* undirected graph [Sec 3.1 assumes connected]
        num_nodes_before_cc = G_undirected.number_of_nodes()
        if num_nodes_before_cc > 0:
            # Check connectivity using the undirected graph
            if not nx.is_connected(G_undirected):
                logger.info("Graph is not connected. Keeping only the largest connected component.")
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                # Create a subgraph view and copy it to make it a new graph object
                G_undirected = G_undirected.subgraph(largest_cc).copy()
                logger.info(f"Largest CC: {G_undirected.number_of_nodes()} nodes / {num_nodes_before_cc} nodes.")
            else:
                 logger.debug("Graph is already connected.")
        elif num_nodes_before_cc == 0:
             logger.warning("Graph became empty before connectivity check.")
             return None

        logger.info(f"Final processed graph ({network_type} filter, largest CC, undirected): {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges.")

        # Final check if the graph is usable
        if G_undirected.number_of_nodes() == 0:
             logger.error(f"Graph from {filepath} resulted in 0 nodes after processing and filtering for '{network_type}'.")
             return None

        return G_undirected

    except Exception as e:
        logger.error(f"Error processing OSM file {filepath}: {e}", exc_info=True) # Log traceback
        return None
