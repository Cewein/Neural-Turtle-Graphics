import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For safety break comparison

# Discrete motion vector range and embedding dimensions
MAX_DISPLACEMENT = 200  # corresponds to +/- 100m (inclusive)
VOCAB_SIZE = 2 * MAX_DISPLACEMENT + 1  # indices 0..200 representing -100..+100
EMBED_SIZE = 64 # Embedding dimension (not specified in paper, 64 is reasonable)
HIDDEN_SIZE = 500  # GRU hidden size as mentioned in NTG paper (Sec 3.4)

class NTGEncoder(nn.Module):
    """
    Encodes a set of incoming paths to a node into a latent vector.
    Uses a Bidirectional GRU and sums the final hidden states for order invariance.
    """
    def __init__(self):
        super(NTGEncoder, self).__init__()
        # Embeddings for discretized Δx and Δy motion vectors
        self.embed_x = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.embed_y = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        # Bidirectional GRU (single layer as implied by paper figures/text)
        self.gru = nn.GRU(input_size=EMBED_SIZE*2, # Concatenated x and y embeddings
                          hidden_size=HIDDEN_SIZE,
                          batch_first=True,        # Input shape: (batch, seq_len, input_size)
                          bidirectional=True)

    def forward(self, incoming_paths):
        """
        Args:
            incoming_paths (list[list[tuple[float, float]]]):
                A list of paths. Each path is a list of (x, y) coordinates,
                ordered from an ancestor node to the current node.

        Returns:
            torch.Tensor: A latent vector of size HIDDEN_SIZE*2 (sum of fwd/bwd states),
                          representing the encoded local topology. Returns zeros if no valid paths.
        """
        device = next(self.parameters()).device
        if not incoming_paths:
            # If no incoming paths (e.g., root node during generation), return zero vector.
            # Paper implies generation starts from a root with some edges,
            # but handling this case robustly is good.
            # The size should match the *output* of the GRU processing (sum of fwd/bwd).
            return torch.zeros(HIDDEN_SIZE * 2, device=device) # Match summed output dim

        latent_sum = torch.zeros(HIDDEN_SIZE * 2, device=device) # BiGRU output size
        num_valid_paths = 0

        for path in incoming_paths:
            if len(path) < 2:
                continue # Need at least two points to form a motion vector

            # Compute sequence of motion vectors (deltas) for this path
            deltas = []
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                dx = x2 - x1
                dy = y2 - y1

                # Discretize and clamp dx, dy to the defined range [-100, 100]
                dx_clamped = int(round(dx))
                dy_clamped = int(round(dy))
                dx_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dx_clamped))
                dy_clamped = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dy_clamped))

                deltas.append((dx_clamped, dy_clamped))

            if not deltas:
                continue # Skip if path resulted in no valid deltas (e.g., single node path passed in)

            # Convert deltas to vocabulary indices (0 to VOCAB_SIZE-1)
            # Index = value + MAX_DISPLACEMENT
            dx_indices = torch.tensor([dx + MAX_DISPLACEMENT for dx, dy in deltas], dtype=torch.long, device=device)
            dy_indices = torch.tensor([dy + MAX_DISPLACEMENT for dx, dy in deltas], dtype=torch.long, device=device)

            # Embed the indices
            embedded_x = self.embed_x(dx_indices)  # shape [seq_len, EMBED_SIZE]
            embedded_y = self.embed_y(dy_indices)  # shape [seq_len, EMBED_SIZE]

            # Concatenate embeddings for GRU input
            # Shape: [seq_len, EMBED_SIZE * 2]
            sequence_embeddings = torch.cat([embedded_x, embedded_y], dim=1)

            # Add batch dimension for GRU: [1, seq_len, EMBED_SIZE * 2]
            sequence_embeddings = sequence_embeddings.unsqueeze(0)

            # Pass through Bidirectional GRU
            # output shape: [1, seq_len, HIDDEN_SIZE * 2] (fwd+bwd for each step)
            # hidden shape: [num_layers * 2, 1, HIDDEN_SIZE] (final hidden states)
            _, hidden = self.gru(sequence_embeddings)
            # hidden contains [fwd_last_state, bwd_last_state] for layer 0

            # Extract final forward and backward hidden states
            # hidden[0] is the last forward state, hidden[1] is the last backward state
            # Each has shape [1, HIDDEN_SIZE]
            # Concatenate or sum them? Paper (Sec 3.2) says "summing the last hidden states".
            # Let's assume sum of final forward and final backward states.
            # However, the diagram suggests summing *across paths*. Let's clarify.
            # Fig 2(b) shows summing the GRU outputs (last hidden states) across paths.
            # Let's sum the fwd[last] and bwd[last] for *this* path first, then sum across paths.
            # Or does it mean sum fwd[0]+bwd[0], fwd[1]+bwd[1], ...? No, "last hidden states".
            # Let's try concatenating fwd/bwd final states for this path, then summing these concatenated vectors across paths.
            # This preserves info from both directions before summing.

            # hidden shape: [2, 1, HIDDEN_SIZE] -> squeeze batch dim -> [2, HIDDEN_SIZE]
            # hidden[0] = final forward, hidden[1] = final backward
            # Concatenate: [HIDDEN_SIZE * 2]
            path_vector = torch.cat((hidden[0, 0, :], hidden[1, 0, :]), dim=0)

            # Sum this path's vector into the total latent representation
            latent_sum += path_vector
            num_valid_paths += 1

        # Optional: Average the latent sum? Paper just says "summing". Summing might favor nodes with more paths.
        # Let's stick to sum as per text.
        # if num_valid_paths > 0:
        #     latent_sum /= num_valid_paths

        return latent_sum

class NTGDecoder(nn.Module):
    """
    Decodes a latent vector into a sequence of outgoing node displacements (dx, dy)
    relative to the current node, until an End-of-Sequence (EOS) token is predicted.
    Uses a unidirectional GRU. Supports teacher forcing for training.
    """
    def __init__(self):
        super(NTGDecoder, self).__init__()
        # Embeddings for discretized Δx and Δy (used as input during generation/teacher forcing)
        self.embed_x = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.embed_y = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)

        # Decoder GRU (unidirectional)
        # Input size matches the concatenated embeddings of the previous step's output delta
        # Hidden size matches the encoder's hidden size per direction (as it initializes the state)
        # The input latent vector dimension might be HIDDEN_SIZE*2 if concatenating fwd/bwd.
        # The initial hidden state h_0 should match the GRU's hidden_size.
        # Let's assume the latent vector from encoder (size HIDDEN_SIZE*2) is projected
        # down to HIDDEN_SIZE to initialize the decoder GRU state.
        self.latent_to_hidden = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE) # Project encoder output

        self.gru = nn.GRU(input_size=EMBED_SIZE*2,
                          hidden_size=HIDDEN_SIZE,
                          batch_first=True) # Output shape: (batch, seq_len, hidden_size)

        # Output layers to predict dx, dy indices and EOS flag
        self.out_x = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE) # Predict index for dx
        self.out_y = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE) # Predict index for dy
        self.out_end = nn.Linear(HIDDEN_SIZE, 2)        # Predict EOS (class 1) or Continue (class 0)

    def forward(self, latent, target_deltas=None, max_steps=50):
        """
        Args:
            latent (torch.Tensor): The encoded latent vector from the NTGEncoder.
                                   Expected shape [HIDDEN_SIZE * 2].
            target_deltas (list[tuple[int, int]], optional):
                Ground truth sequence of outgoing (dx, dy) deltas for teacher forcing during training.
                Each delta should be clamped and integer.
            max_steps (int): Maximum number of decoding steps during inference to prevent infinite loops.

        Returns:
            If target_deltas is provided (training mode):
                torch.Tensor: The total cross-entropy loss for the sequence.
            If target_deltas is None (inference mode):
                list[tuple[int, int]]: The list of predicted (dx, dy) moves relative to the input node.
        """
        device = latent.device
        batch_size = 1 # We process one node at a time

        # Initialize decoder hidden state from the latent vector
        # Project latent vector (HIDDEN_SIZE*2) down to decoder hidden size (HIDDEN_SIZE)
        # Shape required for GRU: [num_layers, batch_size, hidden_size]
        h0 = self.latent_to_hidden(latent).unsqueeze(0) # Shape: [1, HIDDEN_SIZE]
        hidden = h0.unsqueeze(0) # Shape: [1, 1, HIDDEN_SIZE]

        # Start-of-sequence (SOS) token: Use zero displacement (dx=0, dy=0)
        # Index for 0 is MAX_DISPLACEMENT
        sos_dx_idx = torch.tensor([MAX_DISPLACEMENT], dtype=torch.long, device=device)
        sos_dy_idx = torch.tensor([MAX_DISPLACEMENT], dtype=torch.long, device=device)

        # Embed the SOS token
        prev_dx_idx = sos_dx_idx
        prev_dy_idx = sos_dy_idx

        if target_deltas is not None:
            # --- Training Mode (Teacher Forcing) ---
            total_loss = 0.0
            loss_fn = nn.CrossEntropyLoss()

            # Include EOS prediction step (+1)
            num_steps = len(target_deltas) + 1

            for t in range(num_steps):
                # Embed the *previous* ground truth delta (or SOS for t=0)
                # Shape: [1, EMBED_SIZE]
                embedded_x = self.embed_x(prev_dx_idx)
                embedded_y = self.embed_y(prev_dy_idx)
                # Shape: [1, EMBED_SIZE * 2]
                gru_input = torch.cat([embedded_x, embedded_y], dim=1)
                # Add batch and sequence length dimensions: [1, 1, EMBED_SIZE * 2]
                gru_input = gru_input.unsqueeze(1)

                # Run one step through the GRU
                # output shape: [1, 1, HIDDEN_SIZE], hidden shape: [1, 1, HIDDEN_SIZE]
                output, hidden = self.gru(gru_input, hidden)

                # Get GRU output vector for prediction
                # Shape: [HIDDEN_SIZE]
                output_vec = output.squeeze(1).squeeze(0) # Or output[0, 0, :]

                # Compute logits for x, y, and end flag
                # Shapes: [VOCAB_SIZE], [VOCAB_SIZE], [2]
                x_logits = self.out_x(output_vec)
                y_logits = self.out_y(output_vec)
                end_logits = self.out_end(output_vec)

                # Calculate loss for this step
                if t < len(target_deltas):
                    # If predicting a neighbor (not EOS)
                    true_dx, true_dy = target_deltas[t]
                    true_dx_idx = torch.tensor([true_dx + MAX_DISPLACEMENT], dtype=torch.long, device=device)
                    true_dy_idx = torch.tensor([true_dy + MAX_DISPLACEMENT], dtype=torch.long, device=device)
                    true_end_flag = torch.tensor([0], dtype=torch.long, device=device) # 0 = Continue

                    # Add batch dim for loss_fn: [1, VOCAB_SIZE], [1, 2]
                    loss_x = loss_fn(x_logits.unsqueeze(0), true_dx_idx)
                    loss_y = loss_fn(y_logits.unsqueeze(0), true_dy_idx)
                    loss_end = loss_fn(end_logits.unsqueeze(0), true_end_flag)
                    total_loss += loss_x + loss_y + loss_end

                    # Prepare next input using current ground truth (teacher forcing)
                    prev_dx_idx = true_dx_idx
                    prev_dy_idx = true_dy_idx
                else:
                    # If predicting EOS (after the last neighbor)
                    true_end_flag = torch.tensor([1], dtype=torch.long, device=device) # 1 = Stop
                    loss_end = loss_fn(end_logits.unsqueeze(0), true_end_flag)
                    total_loss += loss_end
                    # No need to prepare next input, loop ends here

            return total_loss

        else:
            # --- Inference Mode ---
            predicted_moves = []
            current_dx_idx = sos_dx_idx
            current_dy_idx = sos_dy_idx

            for _ in range(max_steps):
                # Embed the *previous* predicted delta (or SOS for first step)
                embedded_x = self.embed_x(current_dx_idx)
                embedded_y = self.embed_y(current_dy_idx)
                gru_input = torch.cat([embedded_x, embedded_y], dim=1).unsqueeze(1)

                # Run one step through the GRU
                output, hidden = self.gru(gru_input, hidden)
                output_vec = output.squeeze(1).squeeze(0)

                # Predict EOS flag first
                end_logits = self.out_end(output_vec)
                # Use argmax (greedy decoding)
                end_pred = torch.argmax(F.softmax(end_logits, dim=-1)).item()

                if end_pred == 1: # EOS predicted
                    break

                # Predict dx and dy (greedy decoding)
                x_logits = self.out_x(output_vec)
                y_logits = self.out_y(output_vec)
                pred_dx_idx = torch.argmax(F.softmax(x_logits, dim=-1)).item()
                pred_dy_idx = torch.argmax(F.softmax(y_logits, dim=-1)).item()

                # Convert indices back to delta values
                pred_dx = pred_dx_idx - MAX_DISPLACEMENT
                pred_dy = pred_dy_idx - MAX_DISPLACEMENT

                predicted_moves.append((pred_dx, pred_dy))

                # Prepare next input using the current prediction
                current_dx_idx = torch.tensor([pred_dx_idx], dtype=torch.long, device=device)
                current_dy_idx = torch.tensor([pred_dy_idx], dtype=torch.long, device=device)

            # Safety check message if max_steps reached
            if len(predicted_moves) == max_steps:
                print(f"Warning: Decoder reached max_steps ({max_steps}) during inference.")

            return predicted_moves

class NTGModel(nn.Module):
    """
    Neural Turtle Graphics (NTG) Model combining the Encoder and Decoder.
    """
    def __init__(self):
        super(NTGModel, self).__init__()
        self.encoder = NTGEncoder()
        self.decoder = NTGDecoder()

    def forward(self, incoming_paths, target_deltas=None):
        """
        Main forward pass for the NTG model.

        Args:
            incoming_paths (list[list[tuple[float, float]]]): Input for the encoder.
            target_deltas (list[tuple[int, int]], optional): Ground truth for the decoder (training).

        Returns:
            Output from the decoder:
            - Training (target_deltas provided): Total loss (torch.Tensor).
            - Inference (target_deltas is None): List of predicted moves (list[tuple[int, int]]).
        """
        # Encode the incoming paths to get the latent representation
        latent = self.encoder(incoming_paths)

        # Decode the latent representation
        # The decoder handles both training (with teacher forcing) and inference modes
        output = self.decoder(latent, target_deltas)

        return output

