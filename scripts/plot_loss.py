# scripts/plot_loss.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging
import yaml # To potentially read config for paths

# Setup logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_loss_from_csv(csv_filepath: str, output_dir: str, plot_filename: str, show_plot: bool = True):
    """
    Reads training loss data from a CSV file and plots the loss curve.

    Args:
        csv_filepath (str): Path to the training loss CSV file.
        output_dir (str): Directory to save the plot image.
        plot_filename (str): Filename for the saved plot (e.g., 'loss_curve.png').
        show_plot (bool): Whether to display the plot interactively.
    """
    logger.info(f"Attempting to read loss data from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        logger.error(f"Loss CSV file not found: {csv_filepath}")
        return

    try:
        df = pd.read_csv(csv_filepath)
        logger.info(f"Successfully loaded loss data with columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_filepath}: {e}")
        return

    # Check for required columns
    required_cols = ['epoch', 'avg_loss']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"CSV file must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        return
    if df.empty:
        logger.warning("Loss CSV file is empty. Cannot plot.")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['avg_loss'], marker='o', linestyle='-', color='royalblue', markersize=4, label='Average Epoch Loss')
    plt.title('Training Loss per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Determine number of ticks based on number of epochs
    num_epochs = df['epoch'].max()
    tick_step = max(1, num_epochs // 10) # Aim for ~10 ticks
    plt.xticks(range(df['epoch'].min(), df['epoch'].max() + 1, tick_step))


    # --- Saving ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Error creating directory {output_dir}: {e}")
            output_dir = "." # Fallback to current directory

    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss curve plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving loss plot to {save_path}: {e}")

    # --- Showing ---
    if show_plot:
        plt.show()
    else:
        plt.close() # Close the figure if not showing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss curve from a CSV file.")
    parser.add_argument("--config", type=str, default="../config/config.yaml", help="Path to the main configuration file (config.yaml).")
    parser.add_argument("--csv_file", type=str, default=None, help="Path to the training loss CSV file (overrides config).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the plot (overrides config).")
    parser.add_argument("--filename", type=str, default=None, help="Filename for the output plot (overrides config).")
    parser.add_argument("--no_show", action="store_true", help="Do not display the plot interactively.")

    args = parser.parse_args()

    # Load config to get defaults if arguments are not provided
    config = {}
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config file not found at {args.config}. Using defaults or command-line args.")
    except Exception as e:
        logger.error(f"Error loading config file {args.config}: {e}")

    # Determine paths, prioritizing command-line args over config
    output_dir = args.output_dir or config.get('output_dir', 'ntg_output')
    csv_filename = config.get('training', {}).get('loss_log_file', 'training_loss.csv')
    csv_filepath = args.csv_file or os.path.join(output_dir, csv_filename)
    plot_filename = args.filename or config.get('visualization', {}).get('loss_plot_filename', 'training_loss_curve.png')
    show_plot = not args.no_show

    # Call the plotting function
    plot_loss_from_csv(csv_filepath, output_dir, plot_filename, show_plot=show_plot)

    logger.info("Loss plotting script finished.")
