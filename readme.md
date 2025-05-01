# Neural Turtle Graphics (NTG) - Refactored Implementation

This repository contains a refactored Python implementation of the Neural Turtle Graphics (NTG) model, based on the paper:

> **Neural Turtle Graphics for Modeling City Road Layouts**
> Hang Chu, Daiqing Li, David Acuna, Amlan Kar, Maria Shugrina, Xinkai Wei, Ming-Yu Liu, Antonio Torralba, Sanja Fidler
> *CVPR 2020*
> [[Paper Link](https://arxiv.org/abs/1911.10270)]
> [[Project Page](https://nv-tlabs.github.io/NTG)]

This refactoring focuses on modularity, configurability, logging, and maintainability while preserving the core algorithmic logic described in the paper.

## Features

* **Modular Structure:** Code is organized into logical directories (`data`, `model`, `train`, `utils`, `scripts`, `config`).
* **Centralized Configuration:** All hyperparameters, file paths, and settings are managed in `config/config.yaml`.
* **Structured Logging:** Uses Python's `logging` module for informative output, replacing `print` statements. Logs are saved to a file and printed to the console.
* **Progress Bars:** Uses `tqdm` to display progress for data loading, training epochs/batches, and graph generation.
* **Persistent Loss Tracking:** Training loss per epoch is saved to a CSV file (`training_loss.csv` by default).
* **Loss Plotting Script:** Includes `scripts/plot_loss.py` to visualize the training loss curve from the CSV file.
* **Generation Constraints:** Implements constraints during graph generation (max degree, min angle, planarity check) based on statistics calculated from training data or defaults.
* **Paper References:** Code includes comments linking implementations back to relevant sections of the original NTG paper (e.g., `[Sec 3.2]`).
* **Type Hinting:** Added type hints for improved code clarity and static analysis.

## Project Structure

.├── config/│   └── config.yaml           # Central configuration file├── data/│   ├── dataset.py            # Data loading and preparation logic│   └── aussie/               # Example directory for OSM data│       └── map_mel_2.osm     # <<< --- PLACE YOUR OSM FILE(S) HERE --- <<<├── model/│   └── ntg_model.py          # NTG Encoder/Decoder model definition├── train/│   ├── trainer.py            # Training loop logic│   └── generator.py          # Graph generation logic├── utils/│   ├── graph_utils.py        # Graph-related helpers (sampling, stats, merge)│   ├── geometry_utils.py     # Geometric helpers (angles, intersection)│   ├── osm_utils.py          # OSM data loading and filtering│   └── plotter.py            # Graph and loss plotting functions├── scripts/│   └── plot_loss.py          # Script to plot training loss curve├── ntg_output/               # Default output directory (created automatically)│   ├── ntg_model.pth         # Saved trained model│   ├── training_loss.csv     # Logged training loss per epoch│   ├── *.png                 # Saved plots (training example, generated graph, loss curve)│   └── *.log                 # Log file├── main.py                   # Main entry point script├── requirements.txt          # Python dependencies└── README.md                 # This file
## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    * **PyTorch:** Install PyTorch matching your system (CPU or GPU) from the [official website](https://pytorch.org/).
    * **Other dependencies:**
        ```bash
        pip install -r requirements.txt
        ```

4.  **Prepare Data:**
    * **OSM Data:**
        * Download `.osm` files for your desired regions (e.g., from [OpenStreetMap](https://www.openstreetmap.org/export), [Geofabrik](https://download.geofabrik.de/)).
        * Place the `.osm` files in a location accessible by the script (e.g., the `data/` directory).
        * **Crucially, update the `data_source.osm.file_paths` list in `config/config.yaml`** to point to your `.osm` file(s).
    * **Synthetic Data:** If you want to test without OSM data, set `data_source.use_osm: false` in `config/config.yaml`.

5.  **Configure `config/config.yaml`:**
    * Review and adjust parameters as needed, especially:
        * `data_source`: Ensure `use_osm` and `osm.file_paths` are correct.
        * `output_dir`: Change if you want output saved elsewhere.
        * `training`: Adjust `epochs`, `batch_size`, etc.
        * `generation`: Modify `max_nodes`.
        * `device`: Set to `cuda` if you have a compatible GPU and installed the correct PyTorch version.

## Usage

1.  **Run Training and Generation:**
    Execute the main script from the project's root directory:
    ```bash
    python main.py --config config/config.yaml
    ```
    * The script will:
        * Load/generate data.
        * Prepare training samples.
        * Calculate generation constraints.
        * Initialize the NTG model.
        * Train the model (saving loss to CSV).
        * Save the trained model (`.pth`).
        * Generate a new graph using the trained model and constraints.
        * Save plots (training example, generated graph) to the `output_dir`.
        * Log detailed information to the console and a `.log` file in `output_dir`.

2.  **Plot Training Loss:**
    After training, run the plotting script:
    ```bash
    python scripts/plot_loss.py --config config/config.yaml
    ```
    * This will read the `training_loss.csv` file (path determined from config) and save/show the loss curve plot (`training_loss_curve.png` by default).
    * You can override the CSV path or output settings using command-line arguments (see `python scripts/plot_loss.py --help`).

## Notes

* **Core Logic:** The fundamental NTG encoder/decoder architecture, random walk sampling, and generation process aim to replicate the paper's description. No changes were made to the core algorithm's forward pass or sampling logic.
* **Performance:** Training time depends heavily on the dataset size, hardware (CPU/GPU), and configured hyperparameters (epochs, model size).
* **OSM Data:** The quality and density of OSM data significantly impact the training and the quality of generated graphs. Ensure your OSM files cover the areas of interest well.
* **Constraints:** The generation constraints (degree, angle, intersection) help produce more realistic and topologically valid graphs, but might sometimes limit the generation process if the model predicts moves that violate them. The strictness can be tuned via percentiles in the config.
