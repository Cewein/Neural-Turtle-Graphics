# ----------------------------------------------------------------------
# This script orchestrates the entire process of loading data, training
# the NTG model, and generating a new road layout graph based on the
# learned patterns, as described in the paper:
# "Neural Turtle Graphics for Modeling City Road Layouts" (CVPR 2020)
# https://arxiv.org/abs/1911.10270
# ----------------------------------------------------------------------

import os
import sys
import torch
import networkx as nx
import numpy as np # Used in constraint calculation
import random
import yaml # For loading configuration
import logging
import argparse
import math # For angle calculations in seeding
from tqdm import tqdm # For progress bars
from typing import Dict, Any, List, Optional, Tuple

# --- Import Project Modules ---

# Data handling utilities
from src.utils.osm_utils import graph_from_osm           # See: src/utils/osm_utils.py
from src.utils.graph_utils import calculate_graph_statistics # See: src/utils/graph_utils.py
from src.utils.plotter import plot_graph                 # See: src/utils/plotter.py
from src.data import prepare_training_data_from_graphs, generate_synthetic_graphs # See: src/data/dataset.py
# Core model and training/generation logic
from src.model import NTGModel                           # See: src/model/ntg_model.py
from src.train import train_ntg                          # See: src/train/trainer.py
from src.generator import generate_graph                 # See: src/train/generator.py
# Type definitions (if needed, often defined where used)
from src.utils.graph_utils import Coord, NodeID
from src.data import TrainingSample

# Get a logger instance for this main script
# Logging needs to be configured later via setup_logging()
logger = logging.getLogger(__name__)

# ======================================================================
# Configuration Loading
# ======================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the main YAML configuration file and resolves internal references.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        SystemExit: If the file is not found or cannot be parsed.
    """
    logger.debug(f"Attempting to load configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            # Load the raw YAML structure
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Config file content is not a valid YAML dictionary.")

        # Recursively resolve internal references like ${section.key}
        _resolve_config_references(config)

        logger.info(f"Configuration loaded successfully from: {config_path}")
        return config
    except FileNotFoundError:
        # Use print here as logging might not be set up yet
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1) # Critical error, cannot proceed
    except (yaml.YAMLError, ValueError) as e:
        print(f"ERROR: Error parsing configuration file {config_path}: {e}")
        sys.exit(1) # Critical error, cannot proceed

def _resolve_config_references(config: Dict, current_level: Optional[Dict] = None):
    """
    Recursively traverses the config dict and replaces string values
    like "${path.to.key}" with the actual value found at that path within
    the main config dictionary. Helper for load_config.
    """
    if current_level is None:
        current_level = config # Start at the root

    # Iterate through items in the current dictionary level
    for key, value in current_level.items():
        if isinstance(value, dict):
            # Recurse into sub-dictionaries
            _resolve_config_references(config, value)
        elif isinstance(value, list):
            # Iterate through lists, checking items
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    # Recurse into dictionaries within lists
                    _resolve_config_references(config, item)
                elif isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                     # Resolve reference if item is a string reference
                     try:
                         resolved_value = _get_nested_config_value(config, item[2:-1])
                         value[i] = resolved_value # Replace item in list
                     except KeyError:
                         # Log warning if reference path is invalid
                         print(f"Warning: Config reference '{item}' in list could not be resolved.")
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # Resolve reference if value is a string reference
            try:
                resolved_value = _get_nested_config_value(config, value[2:-1])
                current_level[key] = resolved_value # Replace value in dict
            except KeyError:
                # Log warning if reference path is invalid
                print(f"Warning: Config reference '{value}' for key '{key}' could not be resolved.")

def _get_nested_config_value(config: Dict, key_path_str: str) -> Any:
    """
    Retrieves a value from a nested dictionary using a dot-separated key path string.
    Helper for _resolve_config_references. E.g., "data.paths.k_paths".
    """
    keys = key_path_str.split('.')
    value = config
    # Traverse the dictionary according to the key path
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            # Raise error if path is invalid
            raise KeyError(f"Path '{key_path_str}' not found in config.")
    return value

# ======================================================================
# Environment Setup (Logging & Device)
# ======================================================================

def setup_environment(config: Dict[str, Any]) -> torch.device:
    """
    Sets up logging based on configuration and determines the compute device (CPU/GPU).

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary.

    Returns:
        torch.device: The selected compute device.
    """
    # --- Logging Configuration ---
    log_config = config.get('logging', {})
    log_level_str = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid level

    # Ensure output directory exists for the log file
    output_dir = config.get('output_dir', 'ntg_output')
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f"{config.get('project_name', 'ntg')}.log")

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_date_format,
        handlers=[
            logging.FileHandler(log_filename, mode='w'), # 'w' overwrites log file each run
            logging.StreamHandler(sys.stdout)            # Log to console as well
        ],
        force=True # Override any existing logger configurations
    )
    # Suppress overly verbose logs from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("osmnx").setLevel(logging.WARNING) # Control osmnx verbosity

    logger.info("Logging configured.")
    logger.info(f"Log level set to: {log_level_str}")
    logger.info(f"Log file path: {log_filename}")

    # --- Device Selection ---
    device_str = config.get('device', 'auto').lower()
    if device_str == 'auto':
        # Automatically select GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == 'cuda':
        # Explicitly request CUDA, fallback to CPU with warning if unavailable
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        # Default to CPU for any other value
        device = torch.device("cpu")

    logger.info(f"Compute device selected: {device}")
    return device

# ======================================================================
# Data Loading and Preparation
# ======================================================================

def load_graph_data(config: Dict[str, Any]) -> List[nx.Graph]:
    """
    Loads graph data, either from configured OSM files or by generating
    synthetic grid graphs, based on the 'data_source' section of the config.

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary.

    Returns:
        List[nx.Graph]: A list of NetworkX graphs ready for processing.

    Raises:
        SystemExit: If OSM data is configured but no valid files are found/loaded.
    """
    all_graphs: List[nx.Graph] = []
    data_cfg = config.get('data_source', {})

    # Check configuration to decide data source
    if data_cfg.get('use_osm', True):
        # --- Load from OpenStreetMap ---
        logger.info("--- Loading OSM Data ---")
        osm_cfg = data_cfg.get('osm', {})
        osm_files = osm_cfg.get('file_paths', [])
        network_type = osm_cfg.get('network_type', 'drive')
        simplify = osm_cfg.get('simplify_graph', True)

        # Validate configuration
        if not osm_files:
             logger.error("Config Error: 'data_source.osm.file_paths' is missing or empty.")
             sys.exit(1)

        # Filter for existing files
        valid_osm_paths = [p for p in osm_files if os.path.exists(p)]
        if not valid_osm_paths:
            logger.error(f"Config Error: No valid OSM files found at specified paths: {osm_files}")
            sys.exit(1)
        if len(valid_osm_paths) < len(osm_files):
            logger.warning(f"Found {len(valid_osm_paths)} valid OSM paths out of {len(osm_files)} provided.")

        # Process each valid OSM file
        # See: src/utils/osm_utils.py::graph_from_osm
        osm_iterator = tqdm(valid_osm_paths, desc="Loading OSM Files", unit="file", ncols=100)
        for filepath in osm_iterator:
            osm_iterator.set_postfix({"File": os.path.basename(filepath)}, refresh=False)
            # graph_from_osm handles loading, projection, filtering, connectivity checks
            graph = graph_from_osm(filepath, network_type=network_type, simplify=simplify)
            if graph and graph.number_of_nodes() > 0:
                all_graphs.append(graph)
            else:
                logger.warning(f"Skipping invalid or empty graph processed from {filepath}")

        # Final check after processing all files
        if not all_graphs:
             logger.error("Failed to load any valid graphs from the provided OSM files.")
             sys.exit(1)
        logger.info(f"--- Successfully loaded {len(all_graphs)} graphs from OSM files ---")

    else:
        # --- Generate Synthetic Data ---
        logger.info("--- Generating Synthetic Data ---")
        synth_cfg = data_cfg.get('synthetic', {})
        # See: src/data/dataset.py::generate_synthetic_graphs
        all_graphs = generate_synthetic_graphs(
            num_graphs=synth_cfg.get('num_graphs', 5),
            grid_size=synth_cfg.get('grid_size', 10),
            spacing=synth_cfg.get('spacing', 20),
            random_extra_edges=synth_cfg.get('random_edges', 0)
        )
        logger.info(f"--- Generated {len(all_graphs)} synthetic graphs ---")

    return all_graphs

def prepare_data_and_constraints(
    all_graphs: List[nx.Graph],
    config: Dict[str, Any]
) -> Tuple[List[TrainingSample], Dict[str, float]]:
    """
    Prepares training samples (input paths, target deltas) from loaded graphs
    and calculates statistics from these graphs to use as constraints during generation.

    Args:
        all_graphs (List[nx.Graph]): List of loaded/generated graphs.
        config (Dict[str, Any]): The loaded configuration dictionary.

    Returns:
        Tuple[List[TrainingSample], Dict[str, float]]:
            - A list of training samples.
            - A dictionary containing calculated generation constraints (e.g., max_degree).

    Raises:
        SystemExit: If no training data can be prepared.
    """
    # --- Prepare Training Data ---
    # Converts graph structures into (incoming_paths, outgoing_deltas) pairs
    # See: src/data/dataset.py::prepare_training_data_from_graphs
    # Paper Reference: [Sec 3.5 Learning] describes the data format.
    logger.info("--- Preparing Training Data Samples ---")
    train_data = prepare_training_data_from_graphs(all_graphs, config)
    if not train_data:
        logger.error("No training data could be prepared. Check input graphs and config. Exiting.")
        sys.exit(1)

    # --- Calculate Generation Constraints ---
    # Analyzes training graphs to find typical degree/angle values
    # Used later to guide the generation process towards more realistic outputs.
    # See: src/utils/graph_utils.py::calculate_graph_statistics
    # Paper Reference: [Sec 3.5 Inference] mentions using constraints.
    logger.info("--- Calculating Statistics for Generation Constraints ---")
    gen_cfg = config.get('generation', {})
    constraints_cfg = gen_cfg.get('constraints', {})
    generation_constraints = calculate_graph_statistics(
        all_graphs,
        degree_percentile=constraints_cfg.get('degree_percentile', 98.0),
        angle_percentile=constraints_cfg.get('angle_percentile', 2.0)
    )
    if generation_constraints is None:
        logger.warning("Failed to calculate generation constraints. Using defaults/no constraints.")
        generation_constraints = {} # Ensure it's a dict even if empty

    return train_data, generation_constraints

# ======================================================================
# Model Training
# ======================================================================

def initialize_and_train_model(
    train_data: List[TrainingSample],
    config: Dict[str, Any],
    device: torch.device
) -> NTGModel:
    """
    Initializes the NTG model, runs the training loop, and saves the trained model state.

    Args:
        train_data (List[TrainingSample]): The prepared training data.
        config (Dict[str, Any]): The loaded configuration dictionary.
        device (torch.device): The compute device to use for training.

    Returns:
        NTGModel: The trained model instance.
    """
    # --- Initialize Model ---
    # Creates the Encoder-Decoder architecture based on config parameters.
    # See: src/model/ntg_model.py::NTGModel
    # Paper Reference: [Sec 3.2, Sec 3.4] describe the architecture.
    logger.info("--- Initializing Model ---")
    model = NTGModel(config)
    model.to(device)
    logger.info(f"Model initialized on device: {device}")
    # logger.debug(model) # Uncomment to log detailed model structure

    # --- Train Model ---
    # Runs the main training loop using the prepared data.
    # See: src/train/trainer.py::train_ntg
    # Paper Reference: [Sec 3.5 Learning] describes teacher forcing and optimization.

    # Get training configuration parameters
    trn_cfg = config.get('training', {})

    # Get pretraining parameters
    # Pretraining mode is optional, can be set in config to True/False
    pretrain = trn_cfg.get('pretrain', False)
    pretrained_model_path = trn_cfg.get('pretrained_model_path', None)

    logger.info("--- Starting Model Training ---")
    if not train_data:
        logger.warning("Skipping training as no training data was prepared.")
    elif pretrain:
        logger.info("Pretraining mode enabled")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"Loading pretrained model from {pretrained_model_path}")
            try:
                model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
                logger.info("Pretrained model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading pretrained model: {e}")
                sys.exit(1)
    else:
        # train_ntg handles epoch loops, batching, loss calculation, backprop, logging loss to CSV.
        _ = train_ntg(model, train_data, config, device) # Loss history is saved by trainer

        # --- Save Trained Model ---
        # Saves the model's learned parameters (state dictionary) to a file.
        model_filename = f"{config.get('project_name', 'ntg')}_model.pth"
        model_save_path = os.path.join(config['output_dir'], model_filename)
        logger.info(f"--- Saving Trained Model to {model_save_path} ---")
        try:
            torch.save(model.state_dict(), model_save_path)
            logger.info("Model state dictionary saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model state dictionary: {e}")

    return model

# ======================================================================
# Graph Generation
# ======================================================================

def prepare_generation_seed(
    all_graphs: List[nx.Graph],
    config: Dict[str, Any]
) -> Tuple[Coord, Optional[List[Tuple[int, int]]]]:
    """
    Prepares the starting conditions for graph generation:
    - Selects a starting position (root node coordinates).
    - Determines initial edge displacements from the root node.
    Attempts to use a seed from the loaded data, falling back to defaults from config.

    Args:
        all_graphs (List[nx.Graph]): List of loaded/generated graphs (used for seeding).
        config (Dict[str, Any]): The loaded configuration dictionary.

    Returns:
        Tuple[Coord, Optional[List[Tuple[int, int]]]]:
            - The (x, y) coordinates for the root node.
            - A list of initial (dx, dy) displacements from the root, or None.
    """
    logger.info("--- Preparing Seed for Generation ---")
    # Paper Reference: [Sec 3.2 Graph Generation] mentions initialization.

    # Default seed values
    start_node_pos: Coord = (0.0, 0.0)
    initial_deltas: Optional[List[Tuple[int, int]]] = None
    max_displacement = config['model']['max_displacement']
    # Get default seed edges from config if real data seed fails
    default_seed_edges = config.get('generation', {}).get('initial_seed_edges', [])

    # Get minimum number of neighbors for seed node
    min_neighbors = config.get('generation', {}).get('min_seed_node_neighbors', 2)

    # Attempt to find a suitable seed node and edges from the input graphs
    seed_graph: Optional[nx.Graph] = None
    if all_graphs:
        # Prioritize graphs with some complexity (e.g., > 10 nodes)
        candidate_graphs = [g for g in all_graphs if g.number_of_nodes() > 10]
        seed_graph = random.choice(candidate_graphs) if candidate_graphs else random.choice(all_graphs)

    if seed_graph:
        pos_dict = nx.get_node_attributes(seed_graph, 'pos')
        # Prefer nodes with degree >= 2 (intersections) as starting points
        candidate_nodes = [n for n, d in seed_graph.degree() if d >= min_neighbors and n in pos_dict]
        if not candidate_nodes: # Fallback: any node with a position
            candidate_nodes = [n for n in seed_graph.nodes() if n in pos_dict]

        if candidate_nodes:
            # Select a random candidate node as the root
            start_node_id = random.choice(candidate_nodes)
            start_node_pos = pos_dict[start_node_id]
            logger.info(f"Seeding generation from node {start_node_id} in graph with {seed_graph.number_of_nodes()} nodes.")
            logger.info(f"Seed node position: ({start_node_pos[0]:.2f}, {start_node_pos[1]:.2f})")

            # Calculate initial displacements based on its actual neighbors
            base_x, base_y = start_node_pos
            neighbors = list(seed_graph.neighbors(start_node_id))
            temp_deltas = []
            for nbr in neighbors:
                if nbr in pos_dict:
                    # Calculate, clamp, and discretize delta
                    dx_float = pos_dict[nbr][0] - base_x
                    dy_float = pos_dict[nbr][1] - base_y
                    dx_clamped = max(-max_displacement, min(max_displacement, int(round(dx_float))))
                    dy_clamped = max(-max_displacement, min(max_displacement, int(round(dy_float))))
                    # Add if non-zero
                    if dx_clamped != 0 or dy_clamped != 0:
                         temp_deltas.append((dx_clamped, dy_clamped))

            if temp_deltas:
                 # Sort deltas counter-clockwise for consistent initialization order
                 temp_deltas.sort(key=lambda d: math.atan2(d[1], d[0]))
                 initial_deltas = temp_deltas
                 logger.info(f"Using initial deltas derived from real neighbors: {initial_deltas}")
            else:
                 # If neighbors only yielded zero deltas, use default seed
                 logger.warning("Could not derive valid initial deltas from neighbors. Using default seed from config.")
                 start_node_pos = (0.0, 0.0) # Reset position for default seed
                 initial_deltas = default_seed_edges
        else:
             # If no suitable nodes found in the seed graph, use default seed
             logger.warning("Selected seed graph has no nodes with positions. Using default seed from config.")
             initial_deltas = default_seed_edges
    else:
        # If no graphs were loaded/generated, use default seed
        logger.warning("No suitable seed graph available. Using default seed from config.")
        initial_deltas = default_seed_edges

    return start_node_pos, initial_deltas

def generate_and_visualize(
    model: NTGModel,
    config: Dict[str, Any],
    device: torch.device,
    start_node_pos: Coord,
    initial_deltas: Optional[List[Tuple[int, int]]],
    generation_constraints: Dict[str, float],
    edge_width: float = 0.6

):
    """
    Generates a new road layout graph using the trained model and visualizes the result.

    Args:
        model (NTGModel): The trained NTG model.
        config (Dict[str, Any]): The loaded configuration dictionary.
        device (torch.device): The compute device.
        start_node_pos (Coord): The starting coordinates for the root node.
        initial_deltas (Optional[List[Tuple[int, int]]]): Initial displacements from root.
        generation_constraints (Dict[str, float]): Constraints (e.g., max_degree) to apply.
    """
    # --- Generate Graph ---
    # Uses the trained model iteratively to build a new graph.
    # See: src/train/generator.py::generate_graph
    # Paper Reference: [Sec 3.2 Graph Generation], [Sec 3.5 Inference] describe the process.
    logger.info("--- Generating New Road Layout Graph ---")
    generated_G = generate_graph(
        model=model,
        config=config,
        device=device,
        start_node_pos=start_node_pos,
        initial_edges=initial_deltas,
        generation_constraints=generation_constraints # Apply calculated constraints
    )

    # --- Visualize Generated Graph ---
    # See: src/utils/plotter.py::plot_graph
    vis_cfg = config.get('visualization', {})
    if generated_G and generated_G.number_of_nodes() > 0:
        logger.info("--- Visualizing Generated Graph ---")
        plot_graph(
            G=generated_G,
            title=f"Generated Road Layout ({generated_G.number_of_nodes()} nodes)",
            output_dir=config['output_dir'],
            filename=vis_cfg.get('generated_plot_filename', 'generated_map.png'),
            show=vis_cfg.get('show_plots', True),
            edge_width=3.0, # Thicker edges for visibility
            edge_color='red'
        )
    else:
        # Log error if generation failed
        logger.error("Graph generation failed or produced an empty graph.")

# ======================================================================
# Main Workflow Orchestration
# ======================================================================

def run_workflow(config_path: str):
    """
    Executes the complete NTG workflow: setup, data loading, training, generation.

    Args:
        config_path (str): Path to the main configuration file.
    """
    # Step 1: Load configuration
    config = load_config(config_path)

    # Step 2: Setup environment (logging, device)
    device = setup_environment(config)

    # Step 3: Load graph data (OSM or synthetic)
    all_graphs = load_graph_data(config)

    # Step 4: Plot an example training graph (optional)
    vis_cfg = config.get('visualization', {})
    if vis_cfg.get('plot_training_example', True) and all_graphs:
        logger.info("Plotting an example training graph...")
        # See: src/utils/plotter.py::plot_graph
        plot_graph(
            all_graphs[0], # Plot the first loaded graph
            title="Example Training Graph",
            output_dir=config['output_dir'],
            filename=vis_cfg.get('training_plot_filename', 'training_map_example_with_base.png'),
            show=vis_cfg.get('show_plots', True),
            add_basemap=True,
            edge_width=3.0, # Thicker edges for visibility
            edge_color='red'
        )

        plot_graph(
            all_graphs[0], # Plot the first loaded graph
            title="Example Training Graph",
            output_dir=config['output_dir'],
            filename=vis_cfg.get('training_plot_filename', 'training_map_example.png'),
            show=vis_cfg.get('show_plots', True),
            add_basemap=False,
            edge_width=3.0, # Thicker edges for visibility
            edge_color='red'
        )
        

    # Step 5: Prepare training data and calculate generation constraints
    train_data, generation_constraints = prepare_data_and_constraints(all_graphs, config)

    # Step 6: Initialize and train the NTG model
    model = initialize_and_train_model(train_data, config, device)

    # Step 7: Prepare the seed (start position, initial edges) for generation
    start_pos, initial_deltas = prepare_generation_seed(all_graphs, config)

    # Step 8: Generate the final graph using the trained model and visualize it
    generate_and_visualize(
        model, config, device, start_pos, initial_deltas, generation_constraints
    )

    logger.info("--- NTG Workflow Finished ---")


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Neural Turtle Graphics (NTG): Train a model on road networks and generate new layouts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml", # Default path relative to script location
        help="Path to the main configuration file (YAML format)."
    )
    args = parser.parse_args()

    # --- Configuration File Check ---
    # Perform a basic check before attempting to load and set up logging
    if not os.path.exists(args.config):
         # Use print as logging isn't set up yet
         print(f"ERROR: Configuration file not found at the specified path: {args.config}")
         print("Please provide a valid path using the --config argument.")
         sys.exit(1) # Exit if config is missing

    # --- Run the Workflow ---
    # Calls the main orchestrator function
    run_workflow(args.config)
