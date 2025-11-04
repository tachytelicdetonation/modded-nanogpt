"""
Entropy Scoring Script for FineWeb Data Filtering

This script scores documents in FineWeb by combining:
1. Negative Log-Likelihood (NLL) from probe model - measures predictive difficulty
2. Inverse Word Frequency - measures term rarity

Higher scores = more informative documents that should be kept for training.

Usage:
    python data/score_fineweb_entropy.py \\
        --data_pattern "data/fineweb10B/fineweb_train_*.bin" \\
        --checkpoint "checkpoints/probe_model.pt" \\
        --output "data/fineweb_scores.json" \\
        --num_tokens 100000000

Based on research:
- https://arxiv.org/html/2406.14124 (Sample Importance in Data Pruning)
- Score = H(W,q) + H(W,f)
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# Add parent directory to path to import probe model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_probe_model import SimpleGPT, _load_data_shard

# Constants
BOS_ID = 50256
VOCAB_SIZE = 50257


# -----------------------------------------------------------------------------
# Document Extraction


def extract_documents(tokens: Tensor) -> List[Tensor]:
    """
    Split token stream into documents using BOS tokens as separators.

    Args:
        tokens: [num_tokens] tensor of token ids

    Returns:
        List of document tensors (each starting with BOS)
    """
    # Find BOS positions
    bos_positions = (tokens == BOS_ID).nonzero(as_tuple=True)[0].tolist()

    if not bos_positions:
        # No BOS found, treat entire stream as one document
        return [tokens]

    documents = []

    for i in range(len(bos_positions)):
        start = bos_positions[i]
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else len(tokens)

        doc = tokens[start:end]

        # Skip very short documents (< 10 tokens)
        if len(doc) >= 10:
            documents.append(doc)

    return documents


# -----------------------------------------------------------------------------
# Scoring Functions


def compute_nll_score(model: SimpleGPT, document: Tensor, device: torch.device, max_seq_len: int = 512) -> float:
    """
    Compute negative log-likelihood (NLL) per token for a document.

    NLL measures how "surprising" the text is to the model.
    Higher NLL = more informative/surprising content.

    Args:
        model: Trained probe model
        document: [doc_len] tensor of token ids
        device: torch device
        max_seq_len: max sequence length for chunking

    Returns:
        Average NLL per token
    """
    model.eval()

    doc_len = len(document)

    # Skip very short documents
    if doc_len < 2:
        return 0.0

    # Chunk long documents to fit in memory
    total_nll = 0.0
    num_tokens = 0

    with torch.no_grad():
        for start in range(0, doc_len - 1, max_seq_len):
            end = min(start + max_seq_len, doc_len - 1)
            chunk_len = end - start

            # Get input and target
            input_ids = document[start:end].unsqueeze(0).to(device)  # [1, chunk_len]
            targets = document[start + 1:end + 1].to(device)  # [chunk_len]

            # Forward pass
            logits = model(input_ids)  # [1, chunk_len, vocab]
            logits = logits.squeeze(0)  # [chunk_len, vocab]

            # Compute NLL
            log_probs = F.log_softmax(logits, dim=-1)
            token_nlls = -log_probs[range(chunk_len), targets]

            total_nll += token_nlls.sum().item()
            num_tokens += chunk_len

    # Return average NLL per token
    return total_nll / num_tokens if num_tokens > 0 else 0.0


def compute_word_freq_score(document: Tensor, word_freq: Dict[int, float]) -> float:
    """
    Compute inverse word frequency score for a document.

    This measures how rare/unusual the words in the document are.
    Higher score = more rare words = more informative.

    Args:
        document: [doc_len] tensor of token ids
        word_freq: Dictionary mapping token_id -> frequency (0 to 1)

    Returns:
        Average inverse word frequency
    """
    if len(document) < 2:
        return 0.0

    # Skip BOS token
    tokens = document[1:] if document[0] == BOS_ID else document

    # Compute inverse frequency for each token
    scores = []
    for token_id in tokens.tolist():
        freq = word_freq.get(token_id, 1e-6)  # Default to very rare if unknown
        # Inverse frequency: log(1/freq)
        inv_freq = math.log(1.0 / max(freq, 1e-6))
        scores.append(inv_freq)

    return sum(scores) / len(scores) if scores else 0.0


def compute_combined_score(nll_score: float, word_freq_score: float, alpha: float = 1.0) -> float:
    """
    Combine NLL and word frequency scores.

    Score = alpha * NLL + (1 - alpha) * word_freq

    Args:
        nll_score: NLL score from probe model
        word_freq_score: Inverse word frequency score
        alpha: Weight for NLL score (0 to 1)

    Returns:
        Combined importance score
    """
    return alpha * nll_score + (1 - alpha) * word_freq_score


# -----------------------------------------------------------------------------
# Word Frequency Computation


def compute_word_frequencies(data_pattern: str, num_tokens: int = None) -> Dict[int, float]:
    """
    Compute word frequencies across the corpus.

    Args:
        data_pattern: Glob pattern for .bin files
        num_tokens: Optional limit on tokens to process

    Returns:
        Dictionary mapping token_id -> frequency (0 to 1)
    """
    print("Computing word frequencies...")

    files = sorted(glob.glob(data_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {data_pattern}")

    token_counts = Counter()
    total_tokens = 0
    tokens_processed = 0

    for file_path in tqdm(files, desc="Processing files"):
        tokens = _load_data_shard(Path(file_path))

        if num_tokens and tokens_processed >= num_tokens:
            break

        tokens_to_process = tokens
        if num_tokens:
            remaining = num_tokens - tokens_processed
            tokens_to_process = tokens[:remaining]

        # Update counts
        token_counts.update(tokens_to_process.tolist())
        total_tokens += len(tokens_to_process)
        tokens_processed += len(tokens_to_process)

    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Unique tokens: {len(token_counts):,}")

    # Convert counts to frequencies
    word_freq = {
        token_id: count / total_tokens
        for token_id, count in token_counts.items()
    }

    return word_freq


# -----------------------------------------------------------------------------
# Main Scoring Pipeline


def score_documents(
    data_pattern: str,
    checkpoint_path: str,
    output_path: str,
    num_tokens: int = None,
    alpha: float = 0.5,
    max_seq_len: int = 512,
):
    """
    Score all documents in FineWeb dataset.

    Args:
        data_pattern: Glob pattern for .bin files
        checkpoint_path: Path to trained probe model checkpoint
        output_path: Path to save scores JSON
        num_tokens: Optional limit on tokens to process
        alpha: Weight for NLL vs word frequency (0=freq only, 1=NLL only)
        max_seq_len: Max sequence length for model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load probe model
    print(f"\nLoading probe model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    model = SimpleGPT(
        vocab_size=model_config['vocab_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        model_dim=model_config['model_dim'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (validation loss: {checkpoint['val_loss']:.4f})")

    # Compute word frequencies
    print("\nStep 1: Computing word frequencies...")
    word_freq = compute_word_frequencies(data_pattern, num_tokens)

    # Load data and extract documents
    print("\nStep 2: Loading data and extracting documents...")
    files = sorted(glob.glob(data_pattern))
    all_documents = []
    document_metadata = []  # Track which file each document came from

    tokens_processed = 0
    for file_idx, file_path in enumerate(tqdm(files, desc="Loading files")):
        if num_tokens and tokens_processed >= num_tokens:
            break

        tokens = _load_data_shard(Path(file_path))

        if num_tokens:
            remaining = num_tokens - tokens_processed
            tokens = tokens[:remaining]

        documents = extract_documents(tokens)

        for doc in documents:
            all_documents.append(doc)
            document_metadata.append({
                'file_idx': file_idx,
                'file_path': file_path,
                'length': len(doc),
            })

        tokens_processed += len(tokens)

    print(f"Total documents extracted: {len(all_documents):,}")
    print(f"Average document length: {sum(len(d) for d in all_documents) / len(all_documents):.1f} tokens")

    # Score documents
    print(f"\nStep 3: Scoring documents (alpha={alpha})...")
    scores = []

    for i, doc in enumerate(tqdm(all_documents, desc="Scoring")):
        # Compute NLL score
        nll_score = compute_nll_score(model, doc, device, max_seq_len)

        # Compute word frequency score
        word_freq_score = compute_word_freq_score(doc, word_freq)

        # Combined score
        combined_score = compute_combined_score(nll_score, word_freq_score, alpha)

        scores.append({
            'doc_idx': i,
            'nll_score': nll_score,
            'word_freq_score': word_freq_score,
            'combined_score': combined_score,
            'length': document_metadata[i]['length'],
            'file_path': document_metadata[i]['file_path'],
        })

    # Sort by combined score (descending)
    scores.sort(key=lambda x: x['combined_score'], reverse=True)

    # Add rank
    for rank, score_dict in enumerate(scores):
        score_dict['rank'] = rank

    # Save results
    print(f"\nStep 4: Saving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'num_documents': len(scores),
                'total_tokens': sum(s['length'] for s in scores),
                'alpha': alpha,
                'checkpoint': checkpoint_path,
                'data_pattern': data_pattern,
            },
            'scores': scores,
        }, f, indent=2)

    print(f"Scores saved! Total documents: {len(scores):,}")

    # Print statistics
    print("\n" + "="*60)
    print("SCORING STATISTICS")
    print("="*60)
    print(f"NLL scores:")
    print(f"  Mean: {sum(s['nll_score'] for s in scores) / len(scores):.4f}")
    print(f"  Min:  {min(s['nll_score'] for s in scores):.4f}")
    print(f"  Max:  {max(s['nll_score'] for s in scores):.4f}")
    print(f"\nWord freq scores:")
    print(f"  Mean: {sum(s['word_freq_score'] for s in scores) / len(scores):.4f}")
    print(f"  Min:  {min(s['word_freq_score'] for s in scores):.4f}")
    print(f"  Max:  {max(s['word_freq_score'] for s in scores):.4f}")
    print(f"\nCombined scores:")
    print(f"  Mean: {sum(s['combined_score'] for s in scores) / len(scores):.4f}")
    print(f"  Min:  {min(s['combined_score'] for s in scores):.4f}")
    print(f"  Max:  {max(s['combined_score'] for s in scores):.4f}")
    print("="*60)


# -----------------------------------------------------------------------------
# CLI


def main():
    parser = argparse.ArgumentParser(description="Score FineWeb documents by entropy")

    parser.add_argument("--data_pattern", type=str, default="data/fineweb10B/fineweb_train_*.bin",
                        help="Glob pattern for input .bin files")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/probe_model.pt",
                        help="Path to probe model checkpoint")
    parser.add_argument("--output", type=str, default="data/fineweb_scores.json",
                        help="Output path for scores JSON")
    parser.add_argument("--num_tokens", type=int, default=None,
                        help="Limit number of tokens to process (default: all)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for NLL vs word frequency (0=freq only, 1=NLL only)")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Max sequence length for model")

    args = parser.parse_args()

    # Handle DATA_PATH environment variable
    data_path = os.environ.get("DATA_PATH", ".")
    data_pattern = os.path.join(data_path, args.data_pattern)

    print("="*60)
    print("ENTROPY-BASED DOCUMENT SCORING")
    print("="*60)
    print(f"Data pattern: {data_pattern}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Num tokens: {args.num_tokens or 'all'}")
    print(f"Alpha (NLL weight): {args.alpha}")
    print("="*60)

    score_documents(
        data_pattern=data_pattern,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_tokens=args.num_tokens,
        alpha=args.alpha,
        max_seq_len=args.max_seq_len,
    )

    print("\n✓ Scoring complete!")


if __name__ == "__main__":
    main()
