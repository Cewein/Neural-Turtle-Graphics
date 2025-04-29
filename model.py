import torch
import torch.nn as nn
import torch.nn.functional as F

# Discrete motion vector range and embedding dimensions
MAX_DISPLACEMENT = 100  # corresponds to 100m
VOCAB_SIZE = 2 * MAX_DISPLACEMENT + 1  # indices 0..200 representing -100..100
EMBED_SIZE = 64
HIDDEN_SIZE = 500  # GRU hidden size as in NTG paper

class NTGEncoder(nn.Module):
    def __init__(self):
        super(NTGEncoder, self).__init__()
        # Embeddings for Δx and Δy (each in range [0, VOCAB_SIZE-1])
        self.embed_x = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.embed_y = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        # Bidirectional GRU (single layer)
        self.gru = nn.GRU(input_size=EMBED_SIZE*2, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
    def forward(self, incoming_paths):
        """
        incoming_paths: list of paths, each a list of (x, y) coordinates, ending at the current node.
        Returns a latent vector encoding all incoming paths.
        """
        device = next(self.parameters()).device
        if len(incoming_paths) == 0:
            # If no incoming path (should not happen for a connected graph except maybe root), return 0
            return torch.zeros(HIDDEN_SIZE, device=device)
        latent_sum = torch.zeros(HIDDEN_SIZE, device=device)
        for path in incoming_paths:
            if len(path) < 2:
                continue  # no motion in this path
            # Compute sequence of motion vectors for this path
            deltas = []
            for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                dx = int(round(x2 - x1)); dy = int(round(y2 - y1))
                # Clamp to [-100,100]
                dx = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dx))
                dy = max(-MAX_DISPLACEMENT, min(MAX_DISPLACEMENT, dy))
                deltas.append((dx, dy))
            if not deltas:
                continue
            # Convert to index tensors
            dx_idx = torch.tensor([dx + MAX_DISPLACEMENT for dx, dy in deltas], dtype=torch.long, device=device)
            dy_idx = torch.tensor([dy + MAX_DISPLACEMENT for dx, dy in deltas], dtype=torch.long, device=device)
            # Embed and run through GRU
            ex = self.embed_x(dx_idx)  # shape [seq_len, EMBED_SIZE]
            ey = self.embed_y(dy_idx)  # shape [seq_len, EMBED_SIZE]
            seq = torch.cat([ex, ey], dim=1).unsqueeze(0)  # [1, seq_len, 2*EMBED_SIZE]
            _, h = self.gru(seq)  # h shape: [2, 1, HIDDEN_SIZE] (bidirectional GRU)
            # Sum forward and backward final states to get path encoding
            fwd_h = h[0, 0, :]  # forward GRU last state
            bwd_h = h[1, 0, :]  # backward GRU last state
            path_vector = fwd_h + bwd_h
            latent_sum += path_vector
        return latent_sum

class NTGDecoder(nn.Module):
    def __init__(self):
        super(NTGDecoder, self).__init__()
        # Embeddings for Δx and Δy
        self.embed_x = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.embed_y = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        # Unidirectional GRU (single layer)
        self.gru = nn.GRU(input_size=EMBED_SIZE*2, hidden_size=HIDDEN_SIZE, batch_first=True)
        # Output layers for Δx, Δy, and End-of-Sequence flag
        self.out_x = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)
        self.out_y = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)
        self.out_end = nn.Linear(HIDDEN_SIZE, 2)  # 2 classes: 0=continue, 1=stop (EOS)
    def forward(self, latent, target_deltas=None):
        """
        If target_deltas is provided (list of ground-truth (dx, dy) for outgoing neighbors), 
        runs in training mode (teacher-forcing) and returns the total loss.
        If target_deltas is None, runs in inference mode and returns a list of predicted (dx, dy) moves.
        """
        device = next(self.parameters()).device
        # Initialize decoder hidden state with the latent vector
        h = latent.unsqueeze(0).unsqueeze(0)  # shape [1, 1, HIDDEN_SIZE]
        # Start-of-sequence token: use zero displacement (dx=dy=0 corresponds to index MAX_DISPLACEMENT)
        prev_dx_idx = torch.tensor([MAX_DISPLACEMENT], device=device)  # index for 0
        prev_dy_idx = torch.tensor([MAX_DISPLACEMENT], device=device)
        prev_embed = torch.cat([self.embed_x(prev_dx_idx), self.embed_y(prev_dy_idx)], dim=1).unsqueeze(1)  # [1,1,2*EMBED_SIZE]
        if target_deltas is not None:
            # Training mode with teacher forcing
            loss_fn = nn.CrossEntropyLoss()
            total_loss = 0.0
            # Go through each true neighbor delta, then one extra step for EOS
            for t in range(len(target_deltas) + 1):
                out, h = self.gru(prev_embed, h)        # out: [1,1,HIDDEN_SIZE]
                out_vec = out[0, 0, :]                  # [HIDDEN_SIZE]
                # Compute logits for x, y, and end
                x_logits = self.out_x(out_vec)          # [VOCAB_SIZE]
                y_logits = self.out_y(out_vec)          # [VOCAB_SIZE]
                end_logits = self.out_end(out_vec)      # [2]
                if t < len(target_deltas):
                    # Ground truth exists (we are predicting a neighbor)
                    dx, dy = target_deltas[t]
                    dx_idx = dx + MAX_DISPLACEMENT
                    dy_idx = dy + MAX_DISPLACEMENT
                    # Compute loss for this step (x, y, and "continue" for end flag)
                    total_loss += loss_fn(x_logits.unsqueeze(0), torch.tensor([dx_idx], device=device))
                    total_loss += loss_fn(y_logits.unsqueeze(0), torch.tensor([dy_idx], device=device))
                    total_loss += loss_fn(end_logits.unsqueeze(0), torch.tensor([0], device=device))  # 0 = continue
                    # Prepare next input using ground truth (teacher forcing)
                    prev_dx_idx = torch.tensor([dx_idx], device=device)
                    prev_dy_idx = torch.tensor([dy_idx], device=device)
                    prev_embed = torch.cat([self.embed_x(prev_dx_idx), self.embed_y(prev_dy_idx)], dim=1).unsqueeze(1)
                else:
                    # After last neighbor, expect EOS
                    total_loss += loss_fn(end_logits.unsqueeze(0), torch.tensor([1], device=device))  # 1 = stop
                # (No need to update prev_embed after EOS step)
            return total_loss
        else:
            # Inference mode: generate neighbors until EOS
            moves = []
            while True:
                out, h = self.gru(prev_embed, h)
                out_vec = out[0, 0, :]
                # Get probabilities
                end_probs = F.softmax(self.out_end(out_vec), dim=-1)
                end_pred = torch.argmax(end_probs).item()
                if end_pred == 1:  # EOS predicted
                    break
                # Predict dx, dy with highest probability
                dx_idx = torch.argmax(F.softmax(self.out_x(out_vec), dim=-1)).item()
                dy_idx = torch.argmax(F.softmax(self.out_y(out_vec), dim=-1)).item()
                dx = dx_idx - MAX_DISPLACEMENT
                dy = dy_idx - MAX_DISPLACEMENT
                moves.append((dx, dy))
                # Prepare next input as the predicted move
                prev_dx_idx = torch.tensor([dx_idx], device=device)
                prev_dy_idx = torch.tensor([dy_idx], device=device)
                prev_embed = torch.cat([self.embed_x(prev_dx_idx), self.embed_y(prev_dy_idx)], dim=1).unsqueeze(1)
                if len(moves) > 50:  # safety break to avoid infinite loop
                    break
            return moves

class NTGModel(nn.Module):
    def __init__(self):
        super(NTGModel, self).__init__()
        self.encoder = NTGEncoder()
        self.decoder = NTGDecoder()
    def forward(self, incoming_paths, target_deltas=None):
        # Encode incoming paths and then decode either by teacher forcing (if target provided) or by inference.
        latent = self.encoder(incoming_paths)
        return self.decoder(latent, target_deltas)
