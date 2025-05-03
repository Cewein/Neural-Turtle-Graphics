import matplotlib.pyplot as plt
import networkx as nx
import os
import logging
from typing import Optional, Dict, Any

# Import contextily for adding basemaps
try:
    import contextily as cx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("Warning: contextily package not found. Basemaps will not be added to plots.")
    print("Install with: pip install contextily")


logger = logging.getLogger(__name__)

def plot_graph(
    G: nx.Graph,
    title: str = "Road Layout",
    output_dir: str = "output_plots",
    filename: Optional[str] = None,
    show: bool = True,
    add_basemap: bool = True,
    basemap_source: Optional[Any] = None, # e.g., cx.providers.OpenStreetMap.Mapnik
    edge_color: str = 'gray',
    node_color: str = 'blue',
    node_size: int = 10,
    edge_width: float = 0.6,
    edge_alpha: float = 0.7,
    node_alpha: float = 0.8,
) -> None:
    """
    Plots the graph using matplotlib and networkx, using 'pos' attributes for layout.
    Optionally adds a background map using contextily if available and requested.

    Args:
        G (nx.Graph): The graph to plot. Must have 'pos' node attributes
                      (projected coordinates) and ideally G.graph['crs'] attribute
                      set to the correct CRS (e.g., EPSG code) for basemap plotting.
        title (str): Title for the plot.
        output_dir (str): Directory to save the plot image.
        filename (Optional[str]): If provided, saves the plot to this file (e.g., 'graph.png').
                                  If None, only shows the plot interactively (if show=True).
        show (bool): Whether to display the plot using plt.show().
        add_basemap (bool): If True and contextily is available, attempt to add a basemap.
        basemap_source (Optional[Any]): Contextily provider for the basemap.
                                        Defaults to OpenStreetMap.Mapnik if None.
                                        Examples: cx.providers.Esri.WorldImagery for satellite.
    """
    if not G or G.number_of_nodes() == 0:
        logger.warning("Cannot plot empty graph.")
        return

    pos = nx.get_node_attributes(G, 'pos')

    # Cannot plot spatially without positions
    if not pos:
        logger.warning("Graph has no 'pos' attributes. Cannot plot spatial layout.")
        return

    fig, ax = plt.subplots(figsize=(12, 12)) # Get figure and axes objects

    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=edge_width, alpha=edge_alpha)
    # Draw smaller nodes if adding a basemap for better visibility
    node_size = 5 if (add_basemap and CONTEXTILY_AVAILABLE) else node_size
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, alpha=node_alpha)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.grid(True, linestyle='--', alpha=0.5)
    # Let matplotlib handle aspect initially, contextily might adjust it
    # ax.set_aspect('equal', adjustable='box') # May conflict with contextily

    # Adjust plot limits slightly beyond the data range
    try:
        x_coords, y_coords = zip(*pos.values())
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # Add a small buffer to the limits
        buffer_x = (x_max - x_min) * 0.05 if x_max > x_min else 5
        buffer_y = (y_max - y_min) * 0.05 if y_max > y_min else 5
        ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
        ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
    except ValueError:
        logger.warning("Could not determine plot limits from node positions.")

    if add_basemap and CONTEXTILY_AVAILABLE:
        crs = G.graph.get('crs', None) # Get CRS from graph attributes
        if crs:
            logger.info(f"Adding basemap using CRS: {crs}")
            try:
                # Choose default provider if none specified
                provider = basemap_source or cx.providers.OpenStreetMap.Mapnik
                cx.add_basemap(ax, crs=str(crs), source=provider, zoom='auto') # zoom='auto' is often good
                logger.info("Basemap added successfully.")
            except Exception as e:
                logger.error(f"Failed to add basemap using contextily: {e}", exc_info=True)
                logger.error("Ensure the graph's CRS attribute (G.graph['crs']) is correct and contextily can access the internet.")
        else:
            logger.warning("Cannot add basemap: Graph object G is missing the 'crs' attribute in G.graph.")
            logger.warning("Modify osm_utils.py to preserve the CRS from osmnx projection.")
    elif add_basemap and not CONTEXTILY_AVAILABLE:
        logger.warning("Cannot add basemap: 'contextily' package not installed.")
    

    # Save the plot if filename is provided
    if filename:
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Error creating directory {output_dir}: {e}")
                output_dir = "." # Fallback

        filepath = os.path.join(output_dir, filename)
        try:
            # Use tight layout to adjust plot elements
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving plot to {filepath}: {e}")

    # Show the plot if requested
    if show:
        plt.show()

    # Close the figure explicitly to release memory, especially if not showing
    plt.close(fig)



def plot_loss_curve(loss_data: Dict[str, Any], output_dir: str, filename: str, show: bool = True) -> None:
    """
    Plots the training loss curve from logged data.
    (Implementation remains unchanged from previous version)

    Args:
        loss_data (Dict[str, Any]): Dictionary containing 'epochs' (list) and 'avg_loss' (list).
        output_dir (str): Directory to save the plot image.
        filename (str): Filename for the saved plot (e.g., 'loss_curve.png').
        show (bool): Whether to display the plot using plt.show().
    """
    if not loss_data or 'epochs' not in loss_data or 'avg_loss' not in loss_data or not loss_data['epochs']:
        logger.warning("Insufficient data to plot loss curve.")
        return

    fig, ax = plt.subplots(figsize=(10, 6)) # Get figure and axes
    ax.plot(loss_data['epochs'], loss_data['avg_loss'], marker='o', linestyle='-', color='royalblue', markersize=4)
    ax.set_title('Training Loss per Epoch', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Determine number of ticks based on number of epochs
    epochs = loss_data['epochs']
    if epochs:
        min_epoch, max_epoch = min(epochs), max(epochs)
        num_epochs_range = max_epoch - min_epoch + 1
        tick_step = max(1, num_epochs_range // 10) if num_epochs_range > 1 else 1 # Aim for ~10 ticks
        ax.set_xticks(range(min_epoch, max_epoch + 1, tick_step))

    # Save the plot
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Error creating directory {output_dir}: {e}")
            output_dir = "."

    filepath = os.path.join(output_dir, filename)
    try:
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Loss curve plot saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving loss plot to {filepath}: {e}")

    # Show the plot if requested
    if show:
        plt.show()

    # Close the figure
    plt.close(fig)

