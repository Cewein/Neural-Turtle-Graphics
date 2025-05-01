import matplotlib.pyplot as plt
import networkx as nx
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def plot_graph(G: nx.Graph, title: str = "Road Layout", output_dir: str = "output_plots", filename: Optional[str] = None, show: bool = True) -> None:
    """
    Plots the graph using matplotlib and networkx, using 'pos' attributes for layout.

    Args:
        G (nx.Graph): The graph to plot.
        title (str): Title for the plot.
        output_dir (str): Directory to save the plot image.
        filename (Optional[str]): If provided, saves the plot to this file (e.g., 'graph.png').
                                  If None, only shows the plot interactively (if show=True).
        show (bool): Whether to display the plot using plt.show().
    """
    if not G or G.number_of_nodes() == 0:
        logger.warning("Cannot plot empty graph.")
        return

    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        logger.warning("Graph has no 'pos' attributes. Plotting with default spring layout.")
        # Use a layout that doesn't require positions
        pos = nx.spring_layout(G, seed=42) # Use seed for reproducibility

    plt.figure(figsize=(12, 12)) # Use a slightly larger figure size

    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.6, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='blue', alpha=0.8)

    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Ensure aspect ratio is equal for spatial data

    # Adjust plot limits slightly beyond the data range
    if pos:
        x_coords, y_coords = zip(*pos.values())
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_range = x_max - x_min if x_max > x_min else 10
        y_range = y_max - y_min if y_max > y_min else 10
        plt.xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)


    # Save the plot if filename is provided
    if filename:
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Error creating directory {output_dir}: {e}")
                # Fallback: try saving in the current directory
                output_dir = "."

        filepath = os.path.join(output_dir, filename)
        try:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving plot to {filepath}: {e}")

    # Show the plot if requested
    if show:
        plt.show()
    else:
        # Close the figure explicitly if not showing to release memory
        plt.close()

def plot_loss_curve(loss_data: Dict[str, Any], output_dir: str, filename: str, show: bool = True) -> None:
    """
    Plots the training loss curve from logged data.

    Args:
        loss_data (Dict[str, Any]): Dictionary containing 'epochs' (list) and 'avg_loss' (list).
        output_dir (str): Directory to save the plot image.
        filename (str): Filename for the saved plot (e.g., 'loss_curve.png').
        show (bool): Whether to display the plot using plt.show().
    """
    if not loss_data or 'epochs' not in loss_data or 'avg_loss' not in loss_data or not loss_data['epochs']:
        logger.warning("Insufficient data to plot loss curve.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(loss_data['epochs'], loss_data['avg_loss'], marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(range(min(loss_data['epochs']), max(loss_data['epochs']) + 1, max(1, len(loss_data['epochs']) // 10))) # Adjust ticks

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
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Loss curve plot saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving loss plot to {filepath}: {e}")

    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()

