import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

# Type alias for coordinates
Coord = Tuple[float, float]

logger = logging.getLogger(__name__)

class NTGEncoder(nn.Module):
    """
    Encodes a set of incoming paths to a node into a latent vector.
    Uses a Bidirectional GRU and sums the final hidden states for order invariance.
    # [Sec 3.2, Fig 2(b)]
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.max_displacement = config['model']['max_displacement']
        self.vocab_size = 2 * self.max_displacement + 1
        embed_size = config['model']['embed_size']
        hidden_size = config['model']['hidden_size']

        logger.info(f"Initializing NTGEncoder: vocab_size={self.vocab_size}, embed_size={embed_size}, hidden_size={hidden_size}")

        # Embeddings for discretized Δx and Δy motion vectors
        self.embed_x = nn.Embedding(self.vocab_size, embed_size)
        self.embed_y = nn.Embedding(self.vocab_size, embed_size)

        # Bidirectional GRU [Sec 3.2]
        self.gru = nn.GRU(
            input_size=embed_size * 2, # Concatenated x and y embeddings
            hidden_size=hidden_size,
            num_layers=1,             # Paper implies single layer
            batch_first=True,         # Input shape: (batch, seq_len, input_size)
            bidirectional=True
        )

    def forward(self, incoming_paths: List[List[Coord]]) -> torch.Tensor:
        """
        Args:
            incoming_paths (List[List[Coord]]):
                A list of paths. Each path is a list of (x, y) coordinates,
                ordered from an ancestor node to the current node.

        Returns:
            torch.Tensor: A latent vector of size hidden_size * 2 (sum of fwd/bwd states),
                          representing the encoded local topology. Returns zeros if no valid paths.
        """
        # Determine device from parameters
        device = next(self.parameters()).device
        hidden_size = self.gru.hidden_size # Get hidden size from GRU instance

        if not incoming_paths:
            # Handle case with no incoming paths (e.g., root node during generation)
            # Return zero vector matching the expected output dimension (summed hidden states)
            return torch.zeros(hidden_size * 2, device=device)

        latent_sum = torch.zeros(hidden_size * 2, device=device)
        num_valid_paths = 0

        for path in incoming_paths:
            if len(path) < 2:
                # logger.debug("Skipping path with < 2 points in encoder.")
                continue # Need at least two points to form a motion vector

            # Compute sequence of motion vectors (deltas) for this path
            deltas = []
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                dx = x2 - x1
                dy = y2 - y1

                # Discretize and clamp dx, dy [Sec 3.4]
                dx_clamped = int(round(dx))
                dy_clamped = int(round(dy))
                dx_clamped = max(-self.max_displacement, min(self.max_displacement, dx_clamped))
                dy_clamped = max(-self.max_displacement, min(self.max_displacement, dy_clamped))

                # Only add non-zero deltas? Paper doesn't specify, but likely intended.
                # However, zero deltas might encode staying put, let's include them.
                deltas.append((dx_clamped, dy_clamped))

            if not deltas:
                # logger.debug("Skipping path that resulted in no deltas.")
                continue # Skip if path resulted in no valid deltas

            # Convert deltas to vocabulary indices (0 to vocab_size-1)
            # Index = value + max_displacement
            dx_indices = torch.tensor([dx + self.max_displacement for dx, dy in deltas], dtype=torch.long, device=device)
            dy_indices = torch.tensor([dy + self.max_displacement for dx, dy in deltas], dtype=torch.long, device=device)

            # Embed the indices
            embedded_x = self.embed_x(dx_indices)  # shape [seq_len, embed_size]
            embedded_y = self.embed_y(dy_indices)  # shape [seq_len, embed_size]

            # Concatenate embeddings for GRU input
            sequence_embeddings = torch.cat([embedded_x, embedded_y], dim=1) # Shape: [seq_len, embed_size * 2]

            # Add batch dimension for GRU: [1, seq_len, embed_size * 2]
            sequence_embeddings = sequence_embeddings.unsqueeze(0)

            # Pass through Bidirectional GRU
            # output shape: [1, seq_len, hidden_size * 2] (fwd+bwd for each step)
            # hidden shape: [num_layers * 2, 1, hidden_size] (final hidden states)
            _, hidden = self.gru(sequence_embeddings)
            # hidden contains [fwd_last_state, bwd_last_state] for layer 0

            # Extract final forward and backward hidden states
            # hidden shape: [2, 1, hidden_size] -> squeeze batch dim -> [2, hidden_size]
            # hidden[0] = final forward state, hidden[1] = final backward state
            # Concatenate final forward and backward states for this path
            path_vector = torch.cat((hidden[0, 0, :], hidden[1, 0, :]), dim=0) # Shape: [hidden_size * 2]

            # Sum this path's vector into the total latent representation [Sec 3.2, Fig 2(b)]
            latent_sum += path_vector
            num_valid_paths += 1

        # Optional: Average the latent sum? Paper just says "summing".
        # Sticking to sum as per text.
        # if num_valid_paths > 0:
        #     latent_sum /= num_valid_paths

        if num_valid_paths == 0:
             logger.debug("No valid paths found to encode, returning zero vector.")

        return latent_sum

class NTGDecoder(nn.Module):
    """
    Decodes a latent vector into a sequence of outgoing node displacements (dx, dy)
    relative to the current node, until an End-of-Sequence (EOS) token is predicted.
    Uses a unidirectional GRU. Supports teacher forcing for training.
    # [Sec 3.2, Fig 2(b)]
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.max_displacement = config['model']['max_displacement']
        self.vocab_size = 2 * self.max_displacement + 1
        embed_size = config['model']['embed_size']
        hidden_size = config['model']['hidden_size'] # Decoder hidden size matches encoder per direction

        logger.info(f"Initializing NTGDecoder: vocab_size={self.vocab_size}, embed_size={embed_size}, hidden_size={hidden_size}")

        # Embeddings for discretized Δx and Δy (used as input during generation/teacher forcing)
        self.embed_x = nn.Embedding(self.vocab_size, embed_size)
        self.embed_y = nn.Embedding(self.vocab_size, embed_size)

        # Project encoder output (bidirectional) to decoder initial hidden state (unidirectional)
        self.latent_to_hidden = nn.Linear(hidden_size * 2, hidden_size)
        # Activation function for projection (optional, TanH is common for hidden states)
        self.latent_activation = nn.Tanh()

        # Decoder GRU (unidirectional) [Sec 3.2]
        self.gru = nn.GRU(
            input_size=embed_size * 2, # Concatenated embeddings of previous delta
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True          # Input shape: (batch, seq_len, input_size)
        )

        # Output layers to predict dx, dy indices and EOS flag [Sec 3.2]
        self.out_x = nn.Linear(hidden_size, self.vocab_size) # Predict index for dx
        self.out_y = nn.Linear(hidden_size, self.vocab_size) # Predict index for dy
        self.out_end = nn.Linear(hidden_size, 2)             # Predict EOS (class 1) or Continue (class 0)

    def forward(self, latent: torch.Tensor, target_deltas: Optional[List[Tuple[int, int]]] = None, max_steps: int = 50) -> Union[torch.Tensor, List[Tuple[int, int]]]:
        """
        Args:
            latent (torch.Tensor): The encoded latent vector from the NTGEncoder.
                                   Expected shape [hidden_size * 2].
            target_deltas (Optional[List[Tuple[int, int]]]):
                Ground truth sequence of outgoing (dx, dy) deltas for teacher forcing.
                Each delta should be clamped and integer.
            max_steps (int): Maximum number of decoding steps during inference.

        Returns:
            Union[torch.Tensor, List[Tuple[int, int]]]:
                - Training (target_deltas provided): Total cross-entropy loss (torch.Tensor).
                - Inference (target_deltas is None): List of predicted moves (List[Tuple[int, int]]).
        """
        device = latent.device
        batch_size = 1 # Process one node at a time

        # Initialize decoder hidden state from the latent vector [Fig 2(b)]
        # Project latent vector (hidden_size*2) down to decoder hidden size (hidden_size)
        h0 = self.latent_to_hidden(latent)
        h0 = self.latent_activation(h0)
        # Shape required for GRU: [num_layers, batch_size, hidden_size] -> [1, 1, hidden_size]
        hidden = h0.unsqueeze(0).unsqueeze(0) # Add layer and batch dims


        # Start-of-sequence (SOS) token: Use zero displacement (dx=0, dy=0)
        # Index for 0 is max_displacement
        sos_dx_idx = torch.tensor([self.max_displacement], dtype=torch.long, device=device)
        sos_dy_idx = torch.tensor([self.max_displacement], dtype=torch.long, device=device)

        if target_deltas is not None:
            # --- Training Mode (Teacher Forcing) --- [Sec 3.5, Ref: Williams & Zipser 1989]
            total_loss = torch.tensor(0.0, device=device)
            # Use reduction='mean' or 'sum'? Summing individual losses and averaging per batch later.
            loss_fn = nn.CrossEntropyLoss(reduction='sum')

            # Prepare inputs: SOS followed by target deltas
            prev_dx_idx = sos_dx_idx
            prev_dy_idx = sos_dy_idx

            # Iterate through the target sequence + EOS prediction
            num_steps = len(target_deltas) + 1

            for t in range(num_steps):
                # Embed the *previous* ground truth delta (or SOS for t=0)
                embedded_x = self.embed_x(prev_dx_idx) # Shape: [1, embed_size]
                embedded_y = self.embed_y(prev_dy_idx) # Shape: [1, embed_size]
                gru_input = torch.cat([embedded_x, embedded_y], dim=1) # Shape: [1, embed_size * 2]
                # Add batch and sequence length dimensions: [1, 1, embed_size * 2]
                gru_input = gru_input.unsqueeze(1)

                # Run one step through the GRU
                output, hidden = self.gru(gru_input, hidden) # hidden updates automatically

                # Get GRU output vector for prediction
                output_vec = output.squeeze(1).squeeze(0) # Shape: [hidden_size]

                # Compute logits for x, y, and end flag
                x_logits = self.out_x(output_vec)     # Shape: [vocab_size]
                y_logits = self.out_y(output_vec)     # Shape: [vocab_size]
                end_logits = self.out_end(output_vec) # Shape: [2]

                # Calculate loss for this step
                if t < len(target_deltas):
                    # Predicting a neighbor delta (not EOS)
                    true_dx, true_dy = target_deltas[t]
                    true_dx_idx = torch.tensor([true_dx + self.max_displacement], dtype=torch.long, device=device)
                    true_dy_idx = torch.tensor([true_dy + self.max_displacement], dtype=torch.long, device=device)
                    true_end_flag = torch.tensor([0], dtype=torch.long, device=device) # 0 = Continue

                    # Add batch dim for loss_fn: [1, vocab_size], [1, 2]
                    loss_x = loss_fn(x_logits.unsqueeze(0), true_dx_idx)
                    loss_y = loss_fn(y_logits.unsqueeze(0), true_dy_idx)
                    loss_end = loss_fn(end_logits.unsqueeze(0), true_end_flag)
                    total_loss += loss_x + loss_y + loss_end

                    # Prepare next input using current ground truth (teacher forcing)
                    prev_dx_idx = true_dx_idx
                    prev_dy_idx = true_dy_idx
                else:
                    # Predicting EOS (after the last neighbor)
                    true_end_flag = torch.tensor([1], dtype=torch.long, device=device) # 1 = Stop
                    loss_end = loss_fn(end_logits.unsqueeze(0), true_end_flag)
                    total_loss += loss_end
                    # No need to prepare next input, loop ends

            # Return the sum of losses for the sequence (will be averaged over batch later)
            return total_loss

        else:
            # --- Inference Mode ---
            predicted_moves = []
            current_dx_idx = sos_dx_idx
            current_dy_idx = sos_dy_idx

            for step in range(max_steps):
                # Embed the *previous* predicted delta (or SOS for first step)
                embedded_x = self.embed_x(current_dx_idx)
                embedded_y = self.embed_y(current_dy_idx)
                gru_input = torch.cat([embedded_x, embedded_y], dim=1).unsqueeze(1) # Add batch/seq dims

                # Run one step through the GRU
                output, hidden = self.gru(gru_input, hidden)
                output_vec = output.squeeze(1).squeeze(0) # Shape: [hidden_size]

                # Predict EOS flag first
                end_logits = self.out_end(output_vec)
                # Use argmax for greedy decoding
                # Apply softmax for probabilities before argmax (though argmax on logits is equivalent)
                end_prob = F.softmax(end_logits, dim=-1)
                end_pred = torch.argmax(end_prob).item()

                if end_pred == 1: # EOS predicted
                    # logger.debug(f"Decoder predicted EOS at step {step+1}.")
                    break

                # Predict dx and dy (greedy decoding)
                x_logits = self.out_x(output_vec)
                y_logits = self.out_y(output_vec)
                pred_dx_idx = torch.argmax(F.softmax(x_logits, dim=-1)).item()
                pred_dy_idx = torch.argmax(F.softmax(y_logits, dim=-1)).item()

                # Convert indices back to delta values
                pred_dx = pred_dx_idx - self.max_displacement
                pred_dy = pred_dy_idx - self.max_displacement

                # Avoid adding zero moves if they occur (optional, but often desired)
                if pred_dx == 0 and pred_dy == 0:
                    # logger.debug("Decoder predicted zero move, potentially stopping.")
                    # Should we stop here or continue? Let's continue, EOS handles termination.
                    pass # Allow zero moves if predicted, but don't add to list? Or add? Let's add.

                predicted_moves.append((pred_dx, pred_dy))

                # Prepare next input using the current prediction
                current_dx_idx = torch.tensor([pred_dx_idx], dtype=torch.long, device=device)
                current_dy_idx = torch.tensor([pred_dy_idx], dtype=torch.long, device=device)

            # Safety check message if max_steps reached
            if len(predicted_moves) == max_steps:
                logger.warning(f"Decoder reached max_steps ({max_steps}) during inference. Output might be truncated.")

            return predicted_moves

class NTGModel(nn.Module):
    """
    Neural Turtle Graphics (NTG) Model combining the Encoder and Decoder.
    # [Sec 3: Neural Turtle Graphics]
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        logger.info("Initializing NTGModel...")
        self.encoder = NTGEncoder(config)
        self.decoder = NTGDecoder(config)
        logger.info("NTGModel initialized.")

    def forward(self, incoming_paths: List[List[Coord]], target_deltas: Optional[List[Tuple[int, int]]] = None, max_steps: int = 50) -> Union[torch.Tensor, List[Tuple[int, int]]]:
        """
        Main forward pass for the NTG model.

        Args:
            incoming_paths (List[List[Coord]]): Input for the encoder.
            target_deltas (Optional[List[Tuple[int, int]]]): Ground truth for the decoder (training).
            max_steps (int): Max generation steps for decoder inference.

        Returns:
            Union[torch.Tensor, List[Tuple[int, int]]]:
            - Training (target_deltas provided): Total loss for the sequence (torch.Tensor).
            - Inference (target_deltas is None): List of predicted moves (List[Tuple[int, int]]).
        """
        # Encode the incoming paths to get the latent representation [Sec 3.2]
        latent = self.encoder(incoming_paths)

        # Decode the latent representation [Sec 3.2]
        # The decoder handles both training (with teacher forcing) and inference modes internally
        output = self.decoder(latent, target_deltas, max_steps)

        return output
