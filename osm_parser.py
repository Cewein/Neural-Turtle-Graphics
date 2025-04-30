import osmnx as ox
import networkx as nx
import logging
import warnings

# Configure osmnx logging and settings
ox.settings.log_console = True
ox.settings.use_cache = True
logging.getLogger("osmnx").setLevel(logging.ERROR) # Reduce osmnx verbosity
warnings.filterwarnings("ignore", category=UserWarning, module='osmnx') # Ignore specific osmnx warnings if needed

def get_drive_filter():
    """
    Returns a list of standard OSM highway tags considered drivable by osmnx.
    Based on osmnx.settings.useful_tags_way and common driving network types.
    Excludes footways, cycleways, steps, paths, etc.
    """
    # Based on typical 'drive' network filters in osmnx
    drive_tags = [
        "motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        "living_street", "service", # Service roads are often drivable, though sometimes restricted
        # "road", # 'road' is ambiguous, often better to rely on more specific tags
        # Consider adding 'track' if relevant for your specific OSM data/area and definition of 'drive'
    ]
    # Explicitly exclude non-drivable types sometimes included in broad queries
    exclude_tags = [
        "footway", "cycleway", "steps", "path", "pedestrian", "track", "bus_guideway",
        "escape", "raceway", "bridleway", "proposed", "construction", "bus_stop", "crossing",
        "elevator", "emergency_access_point", "platform", "rest_area", "services"
    ]
    # The filter will check if the highway tag is in drive_tags and not in exclude_tags
    # For simplicity here, we primarily focus on including the drive_tags.
    # A more robust filter could use ox.downloader._get_osm_filter(network_type) logic if needed.
    return drive_tags

def filter_graph_by_highway_tags(G, allowed_tags):
    """
    Filters a graph, keeping only edges whose 'highway' tag is in the allowed_tags list.

    Args:
        G (nx.MultiDiGraph or nx.DiGraph): The graph to filter (typically from osmnx).
        allowed_tags (list[str]): List of OSM highway tag values to keep.

    Returns:
        nx.MultiDiGraph: A new graph containing only the filtered edges and their nodes.
                         Returns an empty graph if no edges match.
    """
    edges_to_keep = []
    original_edge_count = G.number_of_edges()

    for u, v, key, data in G.edges(keys=True, data=True):
        highway_tag = data.get('highway', None)

        # Handle cases where highway tag might be a list (less common but possible)
        if isinstance(highway_tag, list):
            # Keep edge if any tag in the list is allowed
            if any(tag in allowed_tags for tag in highway_tag):
                edges_to_keep.append((u, v, key))
        elif isinstance(highway_tag, str):
            # Keep edge if the single tag is allowed
            if highway_tag in allowed_tags:
                edges_to_keep.append((u, v, key))
        # else: highway_tag is None or not a string/list, ignore edge

    # Create a new graph containing only the edges to keep
    G_filtered = G.edge_subgraph(edges_to_keep).copy()

    removed_count = original_edge_count - G_filtered.number_of_edges()
    print(f"  Filtered graph by highway tags. Kept {G_filtered.number_of_edges()} / {original_edge_count} edges (Removed {removed_count}).")

    return G_filtered


def graph_from_osm(filepath, network_type='drive', simplify=True):
    """
    Loads and filters a road network graph from an .osm file for a specific network type.

    Args:
        filepath (str): Path to the .osm file.
        network_type (str): Type of network to extract (e.g., 'drive', 'walk', 'bike', 'all').
                            Currently only 'drive' filtering is implemented robustly.
        simplify (bool): If True, simplify the graph topology *before* filtering.
                         Simplification after filtering might also be desired sometimes.

    Returns:
        networkx.Graph: An undirected graph representing the filtered road network.
                        Nodes have 'pos' attributes (projected coordinates in meters).
                        Returns None if loading or filtering fails or results in an empty graph.
    """
    print(f"\nLoading OSM data from: {filepath}")
    try:
        # Load the graph from the .osm file using graph_from_xml
        # Simplify=True is generally recommended before filtering to handle topology correctly.
        G_initial = ox.graph_from_xml(filepath, simplify=simplify)
        print(f"  Initial load (simplified={simplify}): {G_initial.number_of_nodes()} nodes, {G_initial.number_of_edges()} edges.")

        if G_initial.number_of_nodes() == 0:
             print(f"  Warning: Initial graph loaded from {filepath} is empty.")
             return None

        # Project the graph to a suitable UTM zone to get coordinates in meters
        G_proj = ox.project_graph(G_initial)
        print(f"  Projected graph: {G_proj.number_of_nodes()} nodes, {G_proj.number_of_edges()} edges.")

        # --- Filtering Step ---
        if network_type.lower() == 'drive':
            allowed_highway_tags = get_drive_filter()
            print(f"  Applying 'drive' network filter (tags: {allowed_highway_tags[:5]}...).")
            G_filtered = filter_graph_by_highway_tags(G_proj, allowed_highway_tags)
        # elif network_type.lower() == 'walk': # Example for future extension
        #     allowed_highway_tags = get_walk_filter() # Define this function
        #     G_filtered = filter_graph_by_highway_tags(G_proj, allowed_highway_tags)
        else:
            print(f"  Warning: Network type '{network_type}' filtering not implemented or requested. Using projected graph as is.")
            G_filtered = G_proj # Use the projected graph without filtering

        if G_filtered.number_of_nodes() == 0:
             print(f"  Warning: Graph became empty after filtering for network type '{network_type}'.")
             return None
        # --- End Filtering Step ---

        # Ensure graph is undirected for NTG (paper uses undirected graph G={V,E})
        # Use the filtered graph here
        print("  Converting filtered graph to undirected.")
        G_undirected = nx.Graph(G_filtered)

        # Recalculate node positions from the projected graph if needed
        # (Usually not necessary as nx.Graph preserves node attributes)
        pos_dict = {}
        nodes_missing_pos = 0
        for node, data in G_proj.nodes(data=True): # Get positions from the projected graph before undirected conversion
             if node in G_undirected.nodes: # Check if the node survived filtering and conversion
                if 'x' in data and 'y' in data:
                    pos_dict[node] = (data['x'], data['y'])
                else:
                     # This shouldn't happen if projection worked, but good to check
                     # print(f"  Warning: Node {node} missing x/y coordinates after projection.")
                     nodes_missing_pos +=1

        if nodes_missing_pos > 0:
             print(f"  Warning: {nodes_missing_pos} nodes missing x/y coords in projected data.")

        nx.set_node_attributes(G_undirected, pos_dict, 'pos')

        # Remove nodes without 'pos' attribute if any failed (e.g., didn't survive filtering/conversion)
        nodes_to_remove = [node for node, data in G_undirected.nodes(data=True) if 'pos' not in data]
        if nodes_to_remove:
            print(f"  Warning: Removing {len(nodes_to_remove)} nodes without position data after processing.")
            G_undirected.remove_nodes_from(nodes_to_remove)

        # Keep only the largest connected component of the *final* undirected graph
        # Check connectivity only if the graph is not empty
        num_nodes_before_cc = G_undirected.number_of_nodes()
        if num_nodes_before_cc > 0:
            if not nx.is_connected(G_undirected):
                print("  Graph is not connected. Keeping only the largest connected component.")
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                G_undirected = G_undirected.subgraph(largest_cc).copy()
                print(f"  Largest CC: {G_undirected.number_of_nodes()} nodes / {num_nodes_before_cc} nodes.")
            else:
                 print("  Graph is already connected.")
        elif num_nodes_before_cc == 0:
             print("  Graph became empty before connectivity check.")


        print(f"Final processed graph ({network_type} filter, largest CC, undirected): {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges.")

        # Final check if the graph is usable
        if G_undirected.number_of_nodes() == 0:
             print(f"Warning: Graph from {filepath} resulted in 0 nodes after processing and filtering for '{network_type}'.")
             return None

        return G_undirected

    except Exception as e:
        print(f"Error processing OSM file {filepath}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None

if __name__ == '__main__':
    # Example Usage: Replace with the path to one of your OSM files
    osm_file = 'path/to/your/map.osm' # <--- CHANGE THIS PATH
    network = 'drive' # Specify the network type

    if not os.path.exists(osm_file):
         print(f"Error: OSM file not found at {osm_file}")
         print("Please change the 'osm_file' variable in osm_parser.py to a valid path.")
    else:
        # Call the updated function
        graph = graph_from_osm(osm_file, network_type=network, simplify=True)

        if graph:
            print(f"\nSuccessfully loaded and filtered '{network}' graph from {osm_file}.")
            # Optional: Visualize the filtered graph
            try:
                from visualisation import plot_graph
                plot_graph(graph, title=f"Filtered '{network}' Network from {os.path.basename(osm_file)}")
            except ImportError:
                print("Visualisation module not found, skipping plot.")
        else:
            print(f"\nFailed to load or filter '{network}' graph from {osm_file}.")

