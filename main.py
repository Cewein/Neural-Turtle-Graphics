# %% main.py
import os
import torch
import math
import networkx as nx
import numpy as np # Needed for stats calculation
from osm_parser import graph_from_osm
import random
# Import the new function along with others
from data import prepare_training_data_from_graphs, generate_synthetic_graphs, calculate_graph_statistics
from model import NTGModel
from train import train_ntg, generate_graph
from visualisation import plot_graph

# %%
#  Configuration
#

# Data Loading
# USE_OSM_DATA = True # Set to True to load from OSM files, False for synthetic data
USE_OSM_DATA = True # Set to True to load from OSM files, False for synthetic data
OSM_FILE_PATHS = [
    # !!! IMPORTANT !!!
    'data\\aussie\\map_mel_2.osm',
]
OSM_NETWORK_TYPE = 'drive' # Type of network to extract ('drive', 'walk', 'bike', 'all')

# Synthetic Data Generation (if USE_OSM_DATA = False)
NUM_SYNTHETIC_GRAPHS = 5
GRID_SIZE = 10
SPACING = 20
RANDOM_EDGES = 0

# Data Preparation Parameters (from paper/model)
K_PATHS = 5 # Number of incoming paths to sample (Sec 3.2, 3.5) - Keep relatively small
L_PATHS = 7 # Max length of incoming paths (Sec 3.2, 3.5) - Keep relatively small

# Statistics Calculation Parameters
DEGREE_PERCENTILE = 98 # Use a high percentile to allow for some complex intersections
ANGLE_PERCENTILE = 2   # Use a low percentile to allow for slightly sharper turns than absolute min

# Training Parameters
EPOCHS = 200 # Adjust as needed
BATCH_SIZE = 128 # Adjust based on memory
LEARNING_RATE = 1e-3 # Adjusted learning rate
WEIGHT_DECAY = 1e-4 # Regularization
GRAD_CLIP = 1.0 # Gradient clipping to prevent exploding gradients
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generation Parameters
MAX_GENERATED_NODES = 1000 # Max nodes for the generated graph
GENERATION_K = K_PATHS # Use same K for generation as training
GENERATION_L = L_PATHS # Use same L for generation as training
OUTPUT_DIR = "ntg_output" # Directory for saving models and plots
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "ntg_model.pth")
GENERATED_GRAPH_PLOT_PATH = "generated_map_tokyo.png" # Updated filename
TRAINING_GRAPH_PLOT_PATH = "training_map_example.png"
MAX_DISPLACEMENT = 300 # Max displacement for edges (in meters) - Should match model/data

# --- Define some initial edges for generation ---
# Simple initial structure: root node connects to two other nodes
INITIAL_GENERATION_EDGES = [
    (50, 0),  # Node 1: 50m East of root
    (0, 50)   # Node 2: 50m North of root
]
# --- END ---


# %%
#  Create output directory
#

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %%
# 1. Load Data
#

all_graphs = []
if USE_OSM_DATA:
    print("--- Loading OSM Data ---")
    # Check if paths exist
    valid_osm_paths = [p for p in OSM_FILE_PATHS if os.path.exists(p)]
    if not valid_osm_paths:
         print(f"Error: No valid OSM file paths found in {OSM_FILE_PATHS}.")
         print("Please edit main.py and add correct paths.")
         # Use exit() or raise an error if running as a script
         # For interactive environments, just print the error.
         raise FileNotFoundError("No valid OSM files provided.")
    if len(valid_osm_paths) < len(OSM_FILE_PATHS):
         print(f"Warning: Found {len(valid_osm_paths)} valid paths out of {len(OSM_FILE_PATHS)} provided.")

    for filepath in valid_osm_paths:
        # Use the updated parser which includes filtering
        graph = graph_from_osm(filepath, network_type=OSM_NETWORK_TYPE, simplify=True)
        if graph and graph.number_of_nodes() > 0:
            all_graphs.append(graph)
        else:
            print(f"Skipping invalid or empty graph from {filepath}")
    if not all_graphs:
         print("Error: Failed to load any valid graphs from the provided OSM files.")
         raise ValueError("No usable graphs loaded.")
    print(f"--- Successfully loaded {len(all_graphs)} graphs from OSM files ---")
else:
    print("--- Generating Synthetic Data ---")
    all_graphs = generate_synthetic_graphs(
        num_graphs=NUM_SYNTHETIC_GRAPHS,
        grid_size=GRID_SIZE,
        spacing=SPACING,
        random_extra_edges=RANDOM_EDGES
    )
    print(f"--- Generated {len(all_graphs)} synthetic graphs ---")

# %% Plot an example graph from the loaded data
if all_graphs:
    for graph in all_graphs:
        plot_graph(graph, title="Example Training Graph",
                output_dir=OUTPUT_DIR, filename=TRAINING_GRAPH_PLOT_PATH, show=True)

# %%
#  2. Prepare Training Data
#

print("\n--- Preparing Training Data Samples ---")
# This function now takes the list of loaded graphs
train_data = prepare_training_data_from_graphs(all_graphs, K=K_PATHS, L=L_PATHS)

if not train_data:
    print("\nError: No training data could be prepared. Exiting.")
    raise ValueError("Training data preparation failed.")

# %%
print("\n--- Calculating Statistics for Generation Constraints ---")
# Calculate stats from the actual graphs used for training data
generation_constraints = calculate_graph_statistics(
    all_graphs,
    degree_percentile=DEGREE_PERCENTILE,
    angle_percentile=ANGLE_PERCENTILE
)
if generation_constraints is None:
    print("Warning: Failed to calculate generation constraints. Generation will proceed without them.")
    generation_constraints = {} # Use empty dict to avoid errors later



# %%
#  3. Initialize Model
#

print("\n--- Initializing Model ---")
model = NTGModel()
model.to(DEVICE)
print(f"Model initialized on {DEVICE}")
# print(model) # Optional: Print model structure

# %%
#  4. Train Model
#

print("\n--- Starting Model Training ---")
if not train_data:
     print("Skipping training as no training data was prepared.")
else:
    train_ntg(model, train_data,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNING_RATE,
              weight_decay=WEIGHT_DECAY,
              grad_clip=GRAD_CLIP,
              device=DEVICE)

# %%
#  5. Save Trained Model
#

print(f"\n--- Saving Trained Model to {MODEL_SAVE_PATH} ---")
try:
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# %%
# 6. Prepare Real Seed for Generation
#

print("\n--- Preparing Real Seed for Generation ---")

start_node_id = None
start_node_pos = (0.0, 0.0) # Default start position
real_initial_deltas = INITIAL_GENERATION_EDGES # Default initial edges
seed_graph = None

# Find the first graph with enough nodes and degree complexity
if all_graphs:
    for g in all_graphs:
        if g.number_of_nodes() > 10: # Ensure graph is somewhat substantial
            seed_graph = g
            break

if seed_graph:
    pos_dict = nx.get_node_attributes(seed_graph, 'pos')
    # Try finding a node with degree > 2 for a more interesting start
    candidate_nodes = [n for n, d in seed_graph.degree() if d >= 2] # Look for degree 2 or more
    if not candidate_nodes: # Fallback: use any node if none have degree >= 2
        candidate_nodes = list(seed_graph.nodes())

    if candidate_nodes:
        random.shuffle(candidate_nodes)
        start_node_id = candidate_nodes[0]

        if start_node_id in pos_dict:
            start_node_pos = pos_dict[start_node_id]
            print(f"Selected start node {start_node_id} from graph with {seed_graph.number_of_nodes()} nodes.")
            base_x, base_y = start_node_pos
            neighbors = list(seed_graph.neighbors(start_node_id))
            neighbor_data = []
            temp_deltas = []

            for nbr in neighbors:
                if nbr in pos_dict:
                    dx_float = pos_dict[nbr][0] - base_x
                    dy_float = pos_dict[nbr][1] - base_y
                    # Calculate clamped integer deltas
                    dx_clamped = int(round(dx_float))
                    dy_clamped = int(round(dy_float))
                    # Apply clamping (using MAX_DISPLACEMENT from config/data.py)
                    dx_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dx_clamped))
                    dy_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dy_clamped))
                    # Avoid adding zero deltas
                    if dx_clamped != 0 or dy_clamped != 0:
                         temp_deltas.append((dx_clamped, dy_clamped))

            if temp_deltas:
                 real_initial_deltas = temp_deltas # Use the calculated deltas
                 print(f"Start node position: ({start_node_pos[0]:.2f}, {start_node_pos[1]:.2f})")
                 print(f"Using initial deltas from real neighbors: {real_initial_deltas}")
            else:
                print("Warning: Could not calculate valid initial deltas from neighbors. Using default.")
                start_node_pos = (0.0, 0.0) # Reset position if deltas failed
                real_initial_deltas = INITIAL_GENERATION_EDGES
        else:
            print(f"Warning: Selected start node {start_node_id} missing position. Using default start.")
            start_node_pos = (0.0, 0.0)
            real_initial_deltas = INITIAL_GENERATION_EDGES
    else:
         print("Warning: Seed graph has no nodes. Using default start.")
         start_node_pos = (0.0, 0.0)
         real_initial_deltas = INITIAL_GENERATION_EDGES

else:
    print("Warning: No suitable seed graph found (or using synthetic data). Using default start.")
    start_node_pos = (0.0, 0.0)
    real_initial_deltas = INITIAL_GENERATION_EDGES

# %%
# 7. Generate a New Graph
#

# Optional: Load model if needed:
# print(f"--- Loading Trained Model from {MODEL_SAVE_PATH} ---")
# if os.path.exists(MODEL_SAVE_PATH):
#     try:
#         model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
#         model.to(DEVICE)
#         model.eval()
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model: {e}. Proceeding with potentially untrained model.")
# else:
#     print("Model file not found. Proceeding with potentially untrained model.")


print("\n--- Generating New Road Layout Graph ---")
# Pass the calculated constraints to the generation function
generated_G = generate_graph(model,
                             start_node_pos=start_node_pos,
                             initial_edges=real_initial_deltas,
                             max_nodes=MAX_GENERATED_NODES,
                             K_gen=GENERATION_K,
                             L_gen=GENERATION_L,
                             device=DEVICE,
                             constraints=generation_constraints) # Pass constraints here

# %%
# 8. Visualize Generated Graph
#

if generated_G and generated_G.number_of_nodes() > 0:
    print("\n--- Visualizing Generated Graph ---")
    plot_graph(generated_G, title=f"Generated Road Layout ({generated_G.number_of_nodes()} nodes, Constrained)",
               output_dir=OUTPUT_DIR, filename=GENERATED_GRAPH_PLOT_PATH, show=True)
else:
    print("\nGeneration failed or produced an empty graph.")

print("\n--- Script Finished ---")
# %%
