# main.py - Refactored Entry Point for Neural Turtle Graphics
import os
import sys
import torch
import networkx as nx
import numpy as np
import random
import yaml
import logging
import argparse
from tqdm import tqdm # For progress bars during data loading/prep
from typing import Dict, Any, List, Optional, Tuple
import math

# --- Import Refactored Modules ---
# Assume the script is run from the root directory where main.py resides
# Add root directory to path to ensure imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.osm_utils import graph_from_osm
from src.utils.graph_utils import calculate_graph_statistics, Coord, NodeID
from src.utils.plotter import plot_graph
from src.data import prepare_training_data_from_graphs, generate_synthetic_graphs, TrainingSample
from src.model import NTGModel
from src.train import train_ntg
from src.generator import generate_graph

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Basic validation
            if not isinstance(config, dict):
                raise ValueError("Config file is not a valid YAML dictionary.")
            # Resolve references like ${paths.k_paths}
            resolve_config_references(config)
            return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing configuration file {config_path}: {e}")
        sys.exit(1)
    except ValueError as e:
         print(f"ERROR: Invalid configuration format: {e}")
         sys.exit(1)

def resolve_config_references(config: Dict, current_level: Optional[Dict] = None):
    """Recursively resolves ${...} references in the config dictionary."""
    if current_level is None:
        current_level = config

    for key, value in current_level.items():
        if isinstance(value, dict):
            resolve_config_references(config, value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    resolve_config_references(config, item)
                elif isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                     ref_key_path = item[2:-1].split('.')
                     resolved_value = config
                     try:
                         for k in ref_key_path:
                             resolved_value = resolved_value[k]
                         value[i] = resolved_value
                     except KeyError:
                         print(f"Warning: Config reference '{item}' not found.")
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            ref_key_path = value[2:-1].split('.')
            resolved_value = config
            try:
                for k in ref_key_path:
                    resolved_value = resolved_value[k]
                current_level[key] = resolved_value
            except KeyError:
                print(f"Warning: Config reference '{value}' not found.")


# --- Logging Setup ---
def setup_logging(config: Dict[str, Any]):
    """Configures root logger based on config settings."""
    log_config = config.get('logging', {})
    log_level_str = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    log_level = getattr(logging, log_level_str, logging.INFO)

    # Create output dir if it doesn't exist for the log file
    output_dir = config.get('output_dir', 'ntg_output')
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f"{config.get('project_name', 'ntg')}.log")

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_date_format,
        handlers=[
            logging.FileHandler(log_filename, mode='w'), # Write mode to overwrite log each run
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )
    # Suppress verbose logging from specific libraries if desired
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__) # Get logger for main script
    logger.info("Logging configured.")
    logger.info(f"Log level: {log_level_str}")
    logger.info(f"Log file: {log_filename}")


# --- Main Execution ---
def main(config_path: str):
    """Main function to run the NTG workflow."""

    # 1. Load Configuration
    config = load_config(config_path)

    # 2. Setup Logging
    setup_logging(config)
    logger = logging.getLogger(__name__) # Get logger after setup
    logger.info(f"Loaded configuration from: {config_path}")

    # 3. Determine Device
    device_str = config.get('device', 'auto').lower()
    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 4. Load or Generate Graph Data
    all_graphs: List[nx.Graph] = []
    data_cfg = config.get('data_source', {})
    if data_cfg.get('use_osm', True):
        logger.info("--- Loading OSM Data ---")
        osm_cfg = data_cfg.get('osm', {})
        osm_files = osm_cfg.get('file_paths', [])
        network_type = osm_cfg.get('network_type', 'drive')
        simplify = osm_cfg.get('simplify_graph', True)

        if not osm_files:
             logger.error("Configuration error: 'data_source.osm.file_paths' is empty or missing.")
             sys.exit(1)

        valid_osm_paths = [p for p in osm_files if os.path.exists(p)]
        if not valid_osm_paths:
            logger.error(f"No valid OSM files found in configured paths: {osm_files}")
            sys.exit(1)
        if len(valid_osm_paths) < len(osm_files):
            logger.warning(f"Found {len(valid_osm_paths)} valid paths out of {len(osm_files)} provided.")

        # Add tqdm progress bar for loading multiple OSM files
        osm_iterator = tqdm(valid_osm_paths, desc="Loading OSM Files", unit="file")
        for filepath in osm_iterator:
            osm_iterator.set_postfix({"File": os.path.basename(filepath)})
            graph = graph_from_osm(filepath, network_type=network_type, simplify=simplify)
            if graph and graph.number_of_nodes() > 0:
                all_graphs.append(graph)
            else:
                logger.warning(f"Skipping invalid or empty graph from {filepath}")

        if not all_graphs:
             logger.error("Failed to load any valid graphs from the provided OSM files.")
             sys.exit(1)
        logger.info(f"--- Successfully loaded {len(all_graphs)} graphs from OSM files ---")
    else:
        logger.info("--- Generating Synthetic Data ---")
        synth_cfg = data_cfg.get('synthetic', {})
        all_graphs = generate_synthetic_graphs(
            num_graphs=synth_cfg.get('num_graphs', 5),
            grid_size=synth_cfg.get('grid_size', 10),
            spacing=synth_cfg.get('spacing', 20),
            random_extra_edges=synth_cfg.get('random_edges', 0)
        )
        logger.info(f"--- Generated {len(all_graphs)} synthetic graphs ---")

    # 5. Plot Example Training Graph (Optional)
    vis_cfg = config.get('visualization', {})
    if vis_cfg.get('plot_training_example', True) and all_graphs:
        logger.info("Plotting an example training graph...")
        plot_graph(
            all_graphs[0], # Plot the first loaded/generated graph
            title="Example Training Graph",
            output_dir=config['output_dir'],
            filename=vis_cfg.get('training_plot_filename', 'training_map_example.png'),
            show=vis_cfg.get('show_plots', True)
        )

    # 6. Prepare Training Data
    logger.info("--- Preparing Training Data Samples ---")
    # Pass the main config dictionary to the preparation function
    train_data: List[TrainingSample] = prepare_training_data_from_graphs(all_graphs, config)

    if not train_data:
        logger.error("No training data could be prepared. Exiting.")
        sys.exit(1)

    # 7. Calculate Generation Constraints
    logger.info("--- Calculating Statistics for Generation Constraints ---")
    gen_cfg = config.get('generation', {})
    constraints_cfg = gen_cfg.get('constraints', {})
    generation_constraints = calculate_graph_statistics(
        all_graphs,
        degree_percentile=constraints_cfg.get('degree_percentile', 98.0),
        angle_percentile=constraints_cfg.get('angle_percentile', 2.0)
    )
    if generation_constraints is None:
        logger.warning("Failed to calculate generation constraints. Generation will proceed without them.")
        generation_constraints = {} # Use empty dict to avoid errors later

    # 8. Initialize Model
    logger.info("--- Initializing Model ---")
    # Pass the main config dictionary to the model constructor
    model = NTGModel(config)
    model.to(device)
    logger.info(f"Model initialized on {device}")
    # logger.debug(model) # Optional: Log model structure

    # 9. Train Model
    logger.info("--- Starting Model Training ---")
    if not train_data:
         logger.warning("Skipping training as no training data was prepared.")
         loss_history = []
    else:
        # Pass the main config and device to the training function
        loss_history = train_ntg(model, train_data, config, device)
        # Note: Loss history is also saved to CSV by train_ntg

    # 10. Save Trained Model
    model_filename = f"{config.get('project_name', 'ntg')}_model.pth"
    model_save_path = os.path.join(config['output_dir'], model_filename)
    logger.info(f"--- Saving Trained Model to {model_save_path} ---")
    try:
        torch.save(model.state_dict(), model_save_path)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    # 11. Prepare Seed for Generation (Optional: Use real data)
    logger.info("--- Preparing Seed for Generation ---")
    start_node_pos: Coord = (0.0, 0.0) # Default start position
    real_initial_deltas: Optional[List[Tuple[int, int]]] = None
    max_displacement = config['model']['max_displacement']

    # Try to find a suitable starting point from the loaded graphs
    seed_graph: Optional[nx.Graph] = None
    if all_graphs:
        # Prioritize graphs with some complexity
        candidate_graphs = [g for g in all_graphs if g.number_of_nodes() > 10]
        if candidate_graphs:
             seed_graph = random.choice(candidate_graphs)
        else: # Fallback to any graph
             seed_graph = random.choice(all_graphs)

    if seed_graph:
        pos_dict = nx.get_node_attributes(seed_graph, 'pos')
        # Try finding a node with degree >= 2 for a more interesting start
        candidate_nodes = [n for n, d in seed_graph.degree() if d >= 2 and n in pos_dict]
        if not candidate_nodes: # Fallback: use any node with position
            candidate_nodes = [n for n in seed_graph.nodes() if n in pos_dict]

        if candidate_nodes:
            start_node_id = random.choice(candidate_nodes)
            start_node_pos = pos_dict[start_node_id]
            logger.info(f"Selected start node {start_node_id} from graph with {seed_graph.number_of_nodes()} nodes.")
            logger.info(f"Start node position: ({start_node_pos[0]:.2f}, {start_node_pos[1]:.2f})")

            # Calculate initial deltas from neighbors
            base_x, base_y = start_node_pos
            neighbors = list(seed_graph.neighbors(start_node_id))
            temp_deltas = []
            for nbr in neighbors:
                if nbr in pos_dict:
                    dx_float = pos_dict[nbr][0] - base_x
                    dy_float = pos_dict[nbr][1] - base_y
                    # Clamp and discretize
                    dx_clamped = max(-max_displacement, min(max_displacement, int(round(dx_float))))
                    dy_clamped = max(-max_displacement, min(max_displacement, int(round(dy_float))))
                    if dx_clamped != 0 or dy_clamped != 0:
                         temp_deltas.append((dx_clamped, dy_clamped))

            if temp_deltas:
                 # Sort deltas CCW for potentially better initial structure? Optional.
                 temp_deltas.sort(key=lambda d: math.atan2(d[1], d[0]))
                 real_initial_deltas = temp_deltas
                 logger.info(f"Using initial deltas from real neighbors: {real_initial_deltas}")
            else:
                logger.warning("Could not calculate valid initial deltas from neighbors. Using default seed.")
                start_node_pos = (0.0, 0.0) # Reset position if deltas failed
                real_initial_deltas = config['generation']['initial_seed_edges']
        else:
             logger.warning("Seed graph has no nodes with positions. Using default seed.")
             start_node_pos = (0.0, 0.0)
             real_initial_deltas = config['generation']['initial_seed_edges']
    else:
        logger.warning("No suitable seed graph found (or using synthetic data). Using default seed.")
        start_node_pos = (0.0, 0.0)
        real_initial_deltas = config['generation']['initial_seed_edges']


    # 12. Generate a New Graph
    logger.info("--- Generating New Road Layout Graph ---")
    # Pass config, device, seed, and constraints to the generation function
    generated_G = generate_graph(
        model,
        config,
        device,
        start_node_pos=start_node_pos,
        initial_edges=real_initial_deltas,
        generation_constraints=generation_constraints
    )

    # 13. Visualize Generated Graph
    if generated_G and generated_G.number_of_nodes() > 0:
        logger.info("--- Visualizing Generated Graph ---")
        plot_graph(
            generated_G,
            title=f"Generated Road Layout ({generated_G.number_of_nodes()} nodes)",
            output_dir=config['output_dir'],
            filename=vis_cfg.get('generated_plot_filename', 'generated_map.png'),
            show=vis_cfg.get('show_plots', True)
        )
    else:
        logger.error("Generation failed or produced an empty graph.")

    logger.info("--- NTG Workflow Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Turtle Graphics (NTG) - Train and Generate Road Layouts")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (YAML format)."
    )
    args = parser.parse_args()

    # Check if config file exists before starting
    if not os.path.exists(args.config):
         print(f"ERROR: Configuration file not found at {args.config}")
         sys.exit(1)

    main(args.config)
