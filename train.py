import random
import torch
import torch.nn as nn
import networkx as nx
import math

from data import generate_synthetic_graphs, sample_paths_for_node, prepare_training_data
from model import NTGModel


def train_ntg(model, data, epochs=5, lr=1e-3, weight_decay=1e-4, grad_clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs+1):
        random.shuffle(data)
        total_loss = 0.0
        for incoming_paths, target_deltas in data:
            optimizer.zero_grad()
            loss = model(incoming_paths, target_deltas)  # forward with teacher forcing
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")
    model.eval()

def generate_graph(model, max_nodes=50):
    model.eval()
    G = nx.Graph()
    # Start with a single root node
    G.add_node(0, pos=(0.0, 0.0))
    queue = [0]
    next_id = 1
    while queue and G.number_of_nodes() < max_nodes:
        curr = queue.pop(0)
        curr_pos = G.nodes[curr]['pos']
        # Prepare incoming paths for curr (one-hop paths from each neighbor)
        incoming_paths = []
        for nbr in G.neighbors(curr):
            incoming_paths.append([G.nodes[nbr]['pos'], curr_pos])
        # Generate outgoing moves for curr
        moves = model.encoder(incoming_paths)
        moves = model.decoder(moves)  # this returns a list of (dx, dy) displacements
        for dx, dy in moves:
            new_x = curr_pos[0] + dx
            new_y = curr_pos[1] + dy
            # Check proximity to existing nodes to merge close nodes (within 5m)
            merged = False
            for existing, data in G.nodes(data=True):
                ex, ey = data['pos']
                if math.hypot(new_x - ex, new_y - ey) < 1.0:  # within 5 meters
                    G.add_edge(curr, existing)  # connect to existing node (merge)
                    merged = True
                    break
            if merged:
                continue
            # Otherwise, add a new node
            G.add_node(next_id, pos=(new_x, new_y))
            G.add_edge(curr, next_id)
            queue.append(next_id)
            next_id += 1
    return G

# Example usage:
# Create a synthetic training set and train the model
# graphs = generate_synthetic_graphs(num_graphs=2, grid_size=3, spacing=30)
# train_data = prepare_training_data(graphs, K=3, L=5)
# model = NTGModel()
# train_ntg(model, train_data, epochs=3)  # training for a few epochs for demonstration

# # Generate a new road layout graph
# generated_G = generate_graph(model, max_nodes=20)
# print(f"Generated graph has {generated_G.number_of_nodes()} nodes and {generated_G.number_of_edges()} edges.")
