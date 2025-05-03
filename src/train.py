import random
import torch
import torch.optim as optim
import torch.nn as nn
import time
import math
import logging
import os
import csv
from tqdm import tqdm
from typing import List, Dict, Any

from src.model import NTGModel
from src.data import TrainingSample

logger = logging.getLogger(__name__)

def train_ntg(
    model: NTGModel,
    train_data: List[TrainingSample],
    config: Dict[str, Any],
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Trains the NTG model.

    Args:
        model (NTGModel): The model instance to train.
        train_data (List[TrainingSample]): The prepared training data.
        config (Dict[str, Any]): Configuration dictionary containing training parameters.
        device (torch.device): The device to train on ('cpu' or 'cuda').

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing epoch loss history.
                              Example: [{'epoch': 1, 'avg_loss': 0.5}, ...]
    """
    if not train_data:
        logger.error("Training data is empty. Cannot train.")
        return []

    # Extract training parameters from config
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']
    output_dir = config['output_dir']
    loss_log_filename = os.path.join(output_dir, config['training']['loss_log_file'])

    # Ensure output directory exists for saving loss log
    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Consider using a learning rate scheduler? (Not mentioned in paper)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    logger.info("--- Starting Model Training ---")
    logger.info(f"Dataset size: {len(train_data)} samples")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")
    logger.info(f"Weight Decay: {weight_decay}, Grad Clip: {grad_clip}")
    logger.info(f"Device: {device}")
    logger.info(f"Loss log file: {loss_log_filename}")
    logger.info("-------------------------------")

    epoch_loss_history = []

    # Open CSV file for appending loss data
    try:
        is_new_file = not os.path.exists(loss_log_filename)
        csv_file = open(loss_log_filename, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if is_new_file:
            csv_writer.writerow(['epoch', 'avg_loss', 'duration_sec']) # Write header
    except IOError as e:
        logger.error(f"Could not open loss log file {loss_log_filename}: {e}. Loss will not be saved.")
        csv_file = None
        csv_writer = None

    # Wrap epoch loop with tqdm for overall progress
    epoch_iterator = tqdm(range(1, epochs + 1), desc="Training Epochs", unit="epoch")
    for epoch in epoch_iterator:
        model.train() # Set model to training mode
        random.shuffle(train_data) # Shuffle data at the beginning of each epoch
        epoch_start_time = time.time()
        total_epoch_loss = 0.0
        processed_samples_epoch = 0
        batches_processed = 0

        # Batching
        num_batches = math.ceil(len(train_data) / batch_size)
        batch_iterator = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)

        for i in batch_iterator:
            batch = train_data[i : i + batch_size]
            if not batch: continue

            optimizer.zero_grad()
            batch_loss_sum = 0.0 # Sum loss within the batch
            valid_samples_in_batch = 0

            for incoming_paths, target_deltas in batch:
                 # Ensure target_deltas is not empty (should be guaranteed by data prep)
                 if not target_deltas:
                     logger.warning("Encountered training sample with empty target_deltas. Skipping.")
                     continue

                 # Forward pass - model handles teacher forcing internally [Sec 3.5]
                 # The model's forward pass returns the sum of losses for the sequence
                 loss = model(incoming_paths, target_deltas=target_deltas)

                 # Check for invalid loss (NaN or Inf)
                 if loss is None or not torch.isfinite(loss):
                      logger.warning(f"Invalid loss encountered ({loss}). Skipping sample.")
                      continue

                 # Accumulate loss for the batch
                 batch_loss_sum += loss.item()
                 valid_samples_in_batch += 1

                 # Backpropagate loss for this sample (scaled by batch size for averaging effect)
                 # Average loss *before* backward pass to keep gradients scaled appropriately
                 (loss / len(batch)).backward()

            # Only step and clip if valid samples were processed in the batch
            if valid_samples_in_batch > 0:
                # Clip gradients to prevent exploding gradients [Sec 3.5]
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # Optimizer Step (updates weights based on accumulated gradients)
                optimizer.step()

                total_epoch_loss += batch_loss_sum
                processed_samples_epoch += valid_samples_in_batch
                batches_processed += 1

                # Update batch progress bar description (optional)
                avg_batch_loss = batch_loss_sum / valid_samples_in_batch
                batch_iterator.set_postfix({"Avg Batch Loss": f"{avg_batch_loss:.4f}"})
            # else:
            #     logger.debug(f"Skipped batch starting at index {i} due to only invalid samples.")

        # End of Epoch
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = total_epoch_loss / processed_samples_epoch if processed_samples_epoch > 0 else 0.0

        # Update epoch progress bar description with average loss
        epoch_iterator.set_postfix({"Avg Epoch Loss": f"{avg_epoch_loss:.4f}", "Duration": f"{epoch_duration:.2f}s"})

        # Log epoch results
        logger.debug(f"Epoch {epoch}/{epochs} | Duration: {epoch_duration:.2f}s | Avg Loss: {avg_epoch_loss:.4f} | Samples Processed: {processed_samples_epoch}")

        # Store and save epoch loss
        epoch_result = {'epoch': epoch, 'avg_loss': avg_epoch_loss, 'duration_sec': epoch_duration}
        epoch_loss_history.append(epoch_result)
        if csv_writer:
            try:
                csv_writer.writerow([epoch, avg_epoch_loss, epoch_duration])
                csv_file.flush() # Ensure data is written to disk
            except Exception as e:
                 logger.error(f"Error writing to loss log file: {e}")


        # Optional: Update learning rate scheduler
        # if scheduler: scheduler.step()

    # End of Training
    if csv_file:
        csv_file.close()

    model.eval() # Set model to evaluation mode after training
    logger.info("--- Model Training Finished ---")

    return epoch_loss_history
