# %% main.py
import os
import torch
from osm_parser import graph_from_osm
from data import prepare_training_data_from_graphs, generate_synthetic_graphs
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
    # Add the full paths to your .osm files here
    'data/map_tokyo_0.osm', 
    'data/map_tokyo_1.osm',
    'data/map_tokyo_2.osm',
    'data/map_tokyo_3.osm',
    'data/map_tokyo_4.osm',
    'data/map_tokyo_5.osm',
    'data/map_tokyo_6.osm',
    'data/map_tokyo_7.osm',
    'data/map_tokyo_8.osm',
    'data/map_tokyo_9.osm',
    'data/map_tokyo_10.osm'
        
]
OSM_NETWORK_TYPE = 'drive' # Type of network to extract ('drive', 'walk', 'bike', 'all')

# Synthetic Data Generation (if USE_OSM_DATA = False)
NUM_SYNTHETIC_GRAPHS = 5
GRID_SIZE = 6
SPACING = 20
RANDOM_EDGES = 0

# Data Preparation Parameters (from paper/model)
K_PATHS = 5 # Number of incoming paths to sample (Sec 3.2, 3.5)
L_PATHS = 10 # Max length of incoming paths (Sec 3.2, 3.5)

# Training Parameters
EPOCHS = 50 # Adjust as needed (paper doesn't specify, start moderately)
BATCH_SIZE = 16 # Adjust based on memory
LEARNING_RATE = 1e-4 # Adjusted learning rate (start lower for potentially complex data)
WEIGHT_DECAY = 1e-5 # Regularization
GRAD_CLIP = 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generation Parameters
MAX_GENERATED_NODES = 200 # Max nodes for the generated graph
GENERATION_K = K_PATHS # Use same K for generation as training
GENERATION_L = L_PATHS # Use same L for generation as training
OUTPUT_DIR = "ntg_output" # Directory for saving models and plots
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "ntg_model.pth")
GENERATED_GRAPH_PLOT_PATH = "generated_map.png"
TRAINING_GRAPH_PLOT_PATH = "training_map_example.png"

# --- ADDED: Define some initial edges for generation ---
# Simple initial structure: root node connects to two other nodes
# Deltas are (dx, dy) relative to the root node's position (0,0)
# Let's create one edge going right and one going up.
INITIAL_GENERATION_EDGES = [
    (50, 0),  # Node 1: 50m East of root
    (0, 50)   # Node 2: 50m North of root
]
# You can experiment with different initial structures.
# --- END ADDED ---


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
         print("Error: No valid OSM file paths found in OSM_FILE_PATHS list.")
         print("Please edit main.py and add correct paths.")
         exit()
    if len(valid_osm_paths) < len(OSM_FILE_PATHS):
         print(f"Warning: Found {len(valid_osm_paths)} valid paths out of {len(OSM_FILE_PATHS)} provided.")

    for filepath in valid_osm_paths:
        graph = graph_from_osm(filepath, network_type=OSM_NETWORK_TYPE, simplify=True)
        if graph and graph.number_of_nodes() > 0:
            all_graphs.append(graph)
        else:
            print(f"Skipping invalid or empty graph from {filepath}")
    if not all_graphs:
         print("Error: Failed to load any valid graphs from the provided OSM files.")
         exit()
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
    plot_graph(all_graphs[5], title="Example Training Graph",
               output_dir=OUTPUT_DIR, filename=TRAINING_GRAPH_PLOT_PATH, show=False)

# %%
#  2. Prepare Training Data
#

print("\n--- Preparing Training Data Samples ---")
# This function now takes the list of loaded graphs
train_data = prepare_training_data_from_graphs(all_graphs, K=K_PATHS, L=L_PATHS)

if not train_data:
    print("\nError: No training data could be prepared. Exiting.")
    # Common reasons: graphs too small, K/L values too large, issues with node positions.
    exit()

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
# 6. Generate a New Graph
#

# Optional: Load model if needed:
# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# model.to(DEVICE)
# model.eval()

print("\n--- Generating New Road Layout Graph ---")
generated_G = generate_graph(model,
                             initial_edges=INITIAL_GENERATION_EDGES, # Use the defined initial edges
                             max_nodes=MAX_GENERATED_NODES,
                             K_gen=GENERATION_K,
                             L_gen=GENERATION_L,
                             device=DEVICE)

# %%- 7. Visualize Generated Graph ---
if generated_G and generated_G.number_of_nodes() > 0:
    print("\n--- Visualizing Generated Graph ---")
    plot_graph(generated_G, title=f"Generated Road Layout ({generated_G.number_of_nodes()} nodes)",
               output_dir=OUTPUT_DIR, filename=GENERATED_GRAPH_PLOT_PATH, show=True)
else:
    print("\nGeneration failed or produced an empty graph.")

print("\n--- Script Finished ---")