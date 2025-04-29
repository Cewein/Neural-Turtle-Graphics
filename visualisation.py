import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(G, title="Generated Road Layout"):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_size=50, node_color='blue', edge_color='gray', with_labels=False)
    plt.title(title)
    plt.axis('equal')
    plt.show()

# Visualize the generated graph
# plot_graph(generated_G, title="Generated Road Layout")
