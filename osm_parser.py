import osmnx as ox
import networkx as nx
import logging

# Configure osmnx logging
ox.settings.log_console = True
ox.settings.use_cache = True
logging.getLogger().setLevel(logging.ERROR) # Reduce osmnx verbosity

def graph_from_osm(filepath, network_type='drive', simplify=True): # network_type argument still accepted by *this* function, but not passed down
    """
    Loads a road network graph from an .osm file.

    Args:
        filepath (str): Path to the .osm file.
        network_type (str): Type of network to extract (e.g., 'drive', 'walk', 'bike', 'all').
                            Note: This argument is currently ignored when loading from file
                            but kept for consistency with main.py. Filtering might need
                            to happen post-load if required.
        simplify (bool): If True, simplify the graph topology. Recommended to match NTG paper's
                         concept where nodes are intersections/endpoints.

    Returns:
        networkx.Graph: A graph representing the road network.
                               Nodes have 'x', 'y' attributes (projected coordinates in meters).
                               Returns None if loading fails.
    """
    print(f"Loading OSM data from: {filepath}")
    try:
        # --- CORRECTED LINE ---
        # Load the graph from the .osm file - removed network_type argument
        G = ox.graph_from_xml(filepath, simplify=simplify)
        # --- END CORRECTION ---

        # Project the graph to a suitable UTM zone to get coordinates in meters
        G_proj = ox.project_graph(G)

        print(f"Loaded and projected graph: {len(G_proj.nodes)} nodes, {len(G_proj.edges)} edges.")

        # Ensure graph is undirected for NTG (paper uses undirected graph G={V,E})
        G_undirected = nx.Graph(G_proj)

        # Recalculate node positions if needed
        pos_dict = {}
        for node, data in G_proj.nodes(data=True):
             if node in G_undirected.nodes:
                if 'x' in data and 'y' in data:
                    pos_dict[node] = (data['x'], data['y'])
                else:
                     print(f"Warning: Node {node} missing x/y coordinates after projection.")

        nx.set_node_attributes(G_undirected, pos_dict, 'pos')

        # Remove nodes without 'pos' attribute if any failed
        nodes_to_remove = [node for node, data in G_undirected.nodes(data=True) if 'pos' not in data]
        if nodes_to_remove:
            print(f"Warning: Removing {len(nodes_to_remove)} nodes without position data.")
            G_undirected.remove_nodes_from(nodes_to_remove)

        # Keep only the largest connected component, as NTG assumes connected graphs
        # Check connectivity only if the graph is not empty
        if G_undirected.number_of_nodes() > 0 and not nx.is_connected(G_undirected):
            print("Graph is not connected. Keeping only the largest connected component.")
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            G_undirected = G_undirected.subgraph(largest_cc).copy()
        elif G_undirected.number_of_nodes() == 0:
             print("Graph became empty after processing, skipping connectivity check.")


        print(f"Final graph (largest CC, undirected): {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges.")
        # Add a check to ensure the final graph is not empty before returning
        if G_undirected.number_of_nodes() == 0:
             print(f"Warning: Graph from {filepath} resulted in 0 nodes after processing.")
             return None

        return G_undirected

    except Exception as e:
        print(f"Error processing OSM file {filepath}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage: Replace with the path to one of your OSM files
    # Ensure you have osmnx installed: pip install osmnx
    # You might need other dependencies like geopandas, rtree depending on your system.
    osm_file = 'path/to/your/map.osm' # <--- CHANGE THIS PATH

    if not os.path.exists(osm_file):
         print(f"Error: OSM file not found at {osm_file}")
         print("Please change the 'osm_file' variable in osm_parser.py to a valid path.")
    else:
        graph = graph_from_osm(osm_file)

        if graph:
            print(f"\nSuccessfully loaded graph from {osm_file}.")
            # You can add visualization here if needed
            # from visualisation import plot_graph
            # plot_graph(graph, title=f"Road Network from {os.path.basename(osm_file)}")
        else:
            print(f"\nFailed to load graph from {osm_file}.")

