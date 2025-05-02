# Neural Turtle Graphics (NTG) - Python Implementation

This repository contains a Python implementation of the Neural Turtle Graphics (NTG) model for generating city road layouts, based on the paper:

> **Neural Turtle Graphics for Modeling City Road Layouts**
> Hang Chu, Daiqing Li, David Acuna, Amlan Kar, Maria Shugrina, Xinkai Wei, Ming-Yu Liu, Antonio Torralba, Sanja Fidler
> *CVPR 2020*
> [[Paper Link](https://arxiv.org/abs/1911.10270)]
> [[Project Page](https://nv-tlabs.github.io/NTG)]

## Overview

The NTG model operates like a virtual "turtle" drawing a road network graph. It represents the layout as a graph where nodes are control points (intersections or bends) and edges are road segments. The core of the model is an **Encoder-Decoder architecture** based on Recurrent Neural Networks (specifically GRUs). The **Encoder** analyzes the local structure leading into a node by processing incoming paths represented as sequences of relative movements. It summarizes this local topology into a latent vector. The **Decoder** then takes this latent vector and sequentially predicts the relative coordinates (as discrete displacements) of new nodes connected to the current node, effectively drawing outgoing road segments. The generation process is iterative: starting from an initial seed, the model expands the graph node by node, using a queue to manage which nodes to process next, until a desired size or stopping condition is met.

This implementation uses **OpenStreetMap (OSM)** data as the primary source for learning road network structures. It includes utilities to parse `.osm` files, filter for specific network types (like drivable roads), project coordinates to a metric space (meters), and prepare the data into the format required by the NTG model (sequences of incoming path coordinates and outgoing relative displacements). During generation, constraints derived from the training data (like maximum node degree and minimum angle between roads) can be applied to enhance the realism and topological validity of the generated layouts. Planarity is also enforced by preventing new road segments from crossing existing ones.

#  Getting Started

## 1. Clone & Set Up the Environment

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows (Powershell):
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   First install PyTorch (CPU or CUDA) per the instructions at [https://pytorch.org/](https://pytorch.org/). Then:

   ```bash
   pip install -r requirements.txt
   ```


## 2. Prepare OpenStreetMap (OSM) Data

The model learns from real road networks via `.osm` files.

1. **Download OSM files**

   * Use the OpenStreetMap website’s **Export** feature, or
   * Download bulk extracts from [Geofabrik](https://download.geofabrik.de/).

2. **Place your `.osm` files**
   Create a folder in the project (e.g. `data/`) and move your files there:

   ```bash
   mkdir -p data
   mv ~/Downloads/your_city.osm data/
   ```


## 3. Configure the Project

Open `config/config.yaml` and update the following:

```yaml
data_source:
  use_osm: true
  osm:
    file_paths:
      - "data/your_city.osm"        # ← Update this path
      # - "data/another_region.osm" # ← Add more if needed
    network_type: "drive"          # "drive" for typical road networks
```

Feel free to tweak other settings (learning rate, output directories, etc.) as needed.

## 4. Run Training & Generation

From the project root:

```bash
python main.py --config config/config.yaml
```

* **Logs** will print to the console and be saved in `ntg_output/` (by default).
* **Artifacts** saved in the output directory:

  * Trained model (`.pth`)
  * Epoch loss CSV (`training_loss.csv`)
  * Example graphs (`.png`)


## 5. Visualize Training Loss

After training completes, plot the loss curve:

```bash
python scripts/plot_loss.py --config config/config.yaml
```

Options:

* `--no_show`: don’t open the plot window.
* The script reads `training_loss.csv` and writes `training_loss_curve.png` to the same output directory.

A downward trend indicates potential successful training.
