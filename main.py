# %%
from data import generate_synthetic_graphs, prepare_training_data
from model import NTGModel
from train import train_ntg, generate_graph
from visualisation import plot_graph

# %%
graphs = generate_synthetic_graphs(num_graphs=1, grid_size=5, spacing=20, random_extra_edges=0)
train_data = prepare_training_data(graphs, K=3, L=5)
print(f"Prepared {len(train_data)} training samples from {len(graphs)} graph(s).")

# Each sample: (incoming_paths, outgoing_deltas)
print("Sample data format:", train_data[0][0], "=>", train_data[0][1])

plot_graph(graphs[0], title="Dataset Example Road Layout")

# %%

model = NTGModel()
train_ntg(model, train_data, epochs=30)  # training for a few epochs for demonstration


# %%
# Generate a new road layout graph
generated_G = generate_graph(model, max_nodes=20)
print(f"Generated graph has {generated_G.number_of_nodes()} nodes and {generated_G.number_of_edges()} edges.")

plot_graph(generated_G, title="Generated Road Layout")
# %%
