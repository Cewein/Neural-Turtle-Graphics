import matplotlib.pyplot as plt
import networkx as nx
import os # For creating output directory

def plot_graph(G, title="Generated Road Layout", output_dir="output_plots", filename=None, show=True):
    """
    Plots the graph using matplotlib and networkx.

    Args:
        G (nx.Graph): The graph to plot.
        title (str): Title for the plot.
        output_dir (str): Directory to save the plot image.
        filename (str, optional): If provided, saves the plot to this file (e.g., 'graph.png').
                                  If None, only shows the plot interactively.
        show (bool): Whether to display the plot using plt.show().
    """
    if not G or G.number_of_nodes() == 0:
        print("Cannot plot empty graph.")
        return

    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        print("Warning: Graph has no 'pos' attributes. Plotting with default layout.")
        pos = nx.spring_layout(G) # Fallback layout

    plt.figure(figsize=(10, 10)) # Increase figure size for potentially larger graphs
    nx.draw(G, pos, node_size=15, node_color='blue', edge_color='gray', width=0.5, with_labels=False)
    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.axis('equal') # Ensure aspect ratio is equal for spatial data
    plt.grid(True, linestyle='--', alpha=0.6)

    if filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        try:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    if show:
        plt.show()
    else:
        plt.close() # Close the figure if not showing to save memory



