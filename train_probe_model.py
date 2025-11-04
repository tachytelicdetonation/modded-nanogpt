"""
Probe Model Training Script for Entropy-Based Data Filtering

This script trains a small GPT-2 model (50M params) on a subset of FineWeb data.
The trained model will be used to score documents by perplexity for filtering.

Usage:
    python train_probe_model.py --num_tokens 100000000 --num_steps 5000

Key differences from main training:
- Smaller model (6 layers, 384 dim vs 12 layers, 768 dim)
- Simple Adam optimizer (no Muon/NorMuon)
- Early stopping based on validation loss
- Saves checkpoint for downstream scoring
"""

import argparse
import glob
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Enable TF32 for H100 optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# Constants

BOS_ID = 50256
VOCAB_SIZE = 50257

# -----------------------------------------------------------------------------
# Simple GPT Model (without all the fancy optimizations)


def next_multiple_of_n(x: int, n: int) -> int:
    """Round up x to the next multiple of n"""
    return ((x + n - 1) // n) * n


class SimpleGPT(nn.Module):
    """Simplified GPT model for probe training"""

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int = 2048):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        head_dim = model_dim // num_heads

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=4 * model_dim,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor):
        """
        Args:
            input_ids: [batch_size, seq_len] token ids
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embed(input_ids)  # [B, T, D]

        # Add positional embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos_emb = self.pos_embed(pos)  # [T, D]
        x = x + pos_emb

        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


# -----------------------------------------------------------------------------
# Data Loading


def _load_data_shard(file: Path):
    """Load tokens from a .bin file"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])

    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count mismatch"

    return tokens


def simple_data_loader(filename_pattern: str, num_tokens: int, seq_len: int, batch_size: int):
    """
    Simple data loader that yields batches of (input_ids, targets)

    Args:
        filename_pattern: glob pattern for .bin files
        num_tokens: total tokens to load
        seq_len: sequence length per example
        batch_size: number of sequences per batch
    """
    files = sorted(glob.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    print(f"Loading data from {len(files)} files...")

    # Load all tokens up to num_tokens
    all_tokens = []
    tokens_loaded = 0

    for file_path in files:
        if tokens_loaded >= num_tokens:
            break

        tokens = _load_data_shard(Path(file_path))
        tokens_to_take = min(len(tokens), num_tokens - tokens_loaded)
        all_tokens.append(tokens[:tokens_to_take])
        tokens_loaded += tokens_to_take

        print(f"  Loaded {tokens_loaded:,} / {num_tokens:,} tokens")

    all_tokens = torch.cat(all_tokens)
    print(f"Total tokens loaded: {len(all_tokens):,}")

    # Create batches
    # Each batch needs batch_size * (seq_len + 1) tokens
    tokens_per_batch = batch_size * (seq_len + 1)
    num_batches = len(all_tokens) // tokens_per_batch
    print(f"Total batches: {num_batches}")

    # Trim to fit exact number of batches
    total_length = num_batches * tokens_per_batch
    all_tokens = all_tokens[:total_length]

    for i in range(num_batches):
        start = i * tokens_per_batch
        end = start + tokens_per_batch

        batch_tokens = all_tokens[start:end]

        # Reshape to [batch_size, seq_len + 1]
        batch_tokens = batch_tokens.view(batch_size, seq_len + 1)

        input_ids = batch_tokens[:, :-1].long()  # Convert uint16 -> int64 for embedding
        targets = batch_tokens[:, 1:].long()    # Convert uint16 -> int64 for cross_entropy

        yield input_ids, targets


# -----------------------------------------------------------------------------
# Training


def train_probe_model(
    train_data_pattern: str,
    val_data_pattern: str,
    num_train_tokens: int,
    num_val_tokens: int,
    num_steps: int,
    checkpoint_path: str,
    # Model config
    num_layers: int = 6,
    model_dim: int = 384,
    num_heads: int = 6,
    # Training config
    batch_size: int = 8,
    seq_len: int = 512,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    eval_every: int = 100,
    patience: int = 5,
):
    """Train a small probe model for entropy scoring"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print(f"\nInitializing model:")
    print(f"  Layers: {num_layers}")
    print(f"  Model dim: {model_dim}")
    print(f"  Num heads: {num_heads}")

    model = SimpleGPT(
        vocab_size=VOCAB_SIZE,
        num_layers=num_layers,
        num_heads=num_heads,
        model_dim=model_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate schedule with warmup
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (num_steps - warmup_steps)
            return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    # Training loop
    print(f"\nStarting training for {num_steps} steps...")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}\n")

    # Load training data
    train_loader = simple_data_loader(train_data_pattern, num_train_tokens, seq_len, batch_size)

    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0

    start_time = time.time()

    for input_ids, targets in train_loader:
        if step >= num_steps:
            break

        # Move to device
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)
        optimizer.step()

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * batch_size * seq_len / elapsed
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {get_lr(step):.6f} | Tokens/s: {tokens_per_sec:,.0f}")

        # Validation
        if step % eval_every == 0 and step > 0:
            model.eval()
            val_loss = evaluate(model, val_data_pattern, num_val_tokens, batch_size, seq_len, device)
            print(f"\n{'='*60}")
            print(f"Step {step} | Validation Loss: {val_loss:.4f}")
            print(f"{'='*60}\n")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                print(f"New best validation loss! Saving checkpoint...")
                save_checkpoint(model, optimizer, step, best_val_loss, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {step} steps (patience={patience})")
                    break

            model.train()

        step += 1

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return best_val_loss


def evaluate(model, data_pattern: str, num_tokens: int, batch_size: int, seq_len: int, device):
    """Evaluate model on validation data"""
    loader = simple_data_loader(data_pattern, num_tokens, seq_len, batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def save_checkpoint(model, optimizer, step, val_loss, path: str):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'model_config': {
            'vocab_size': VOCAB_SIZE,
            'num_layers': model.blocks.__len__(),
            'model_dim': model.embed.embedding_dim,
            'num_heads': 6,  # Fixed in this simple version
        }
    }

    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Train probe model for entropy filtering")

    # Data paths
    parser.add_argument("--train_data", type=str, default="data/fineweb10B/fineweb_train_*.bin",
                        help="Training data pattern")
    parser.add_argument("--val_data", type=str, default="data/fineweb10B/fineweb_val_*.bin",
                        help="Validation data pattern")

    # Training config
    parser.add_argument("--num_tokens", type=int, default=100_000_000,
                        help="Number of training tokens (default: 100M)")
    parser.add_argument("--num_val_tokens", type=int, default=10_000_000,
                        help="Number of validation tokens (default: 10M)")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")

    # Model config
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of layers")
    parser.add_argument("--model_dim", type=int, default=384,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Number of attention heads")

    # Output
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/probe_model.pt",
                        help="Path to save checkpoint")

    args = parser.parse_args()

    # Handle DATA_PATH environment variable
    data_path = os.environ.get("DATA_PATH", ".")
    train_data = os.path.join(data_path, args.train_data)
    val_data = os.path.join(data_path, args.val_data)

    print("="*60)
    print("PROBE MODEL TRAINING")
    print("="*60)
    print(f"Training data: {train_data}")
    print(f"Validation data: {val_data}")
    print(f"Number of training tokens: {args.num_tokens:,}")
    print(f"Number of validation tokens: {args.num_val_tokens:,}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print("="*60)

    train_probe_model(
        train_data_pattern=train_data,
        val_data_pattern=val_data,
        num_train_tokens=args.num_tokens,
        num_val_tokens=args.num_val_tokens,
        num_steps=args.num_steps,
        checkpoint_path=args.checkpoint_path,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
