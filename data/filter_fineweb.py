"""
Data Filtering Script for FineWeb

Takes scored documents and creates filtered .bin files containing only
the top K% of documents by importance score.

Usage:
    python data/filter_fineweb.py \\
        --scores data/fineweb_scores.json \\
        --input_pattern "data/fineweb10B/fineweb_train_*.bin" \\
        --output_dir "data/fineweb10B_filtered" \\
        --keep_fraction 0.80

This will:
1. Load scores from JSON
2. Select top 80% of documents
3. Extract those documents from original .bin files
4. Write new .bin files in the same format
"""

import argparse
import glob
import json
import os
import struct
from pathlib import Path
from typing import List, Set

import torch
from tqdm import tqdm

# Constants
BOS_ID = 50256
MAGIC_NUMBER = 20240520
VERSION = 1


# -----------------------------------------------------------------------------
# Data Loading


def _load_data_shard(file: Path):
    """Load tokens from a .bin file"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == MAGIC_NUMBER, "magic number mismatch"
    assert header[1] == VERSION, "unsupported version"
    num_tokens = int(header[2])

    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count mismatch"

    return tokens


def extract_documents(tokens: torch.Tensor) -> List[torch.Tensor]:
    """
    Split token stream into documents using BOS tokens as separators.

    Args:
        tokens: [num_tokens] tensor of token ids

    Returns:
        List of document tensors (each starting with BOS)
    """
    # Find BOS positions (use numpy for speed)
    import numpy as np
    bos_mask = tokens.numpy() == BOS_ID
    bos_positions = np.where(bos_mask)[0].tolist()

    if not bos_positions:
        # No BOS found, treat entire stream as one document
        return [tokens]

    documents = []
    num_tokens = len(tokens)

    # Pre-allocate list for speed
    documents = []

    for i in range(len(bos_positions)):
        start = bos_positions[i]
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else num_tokens

        # Skip very short documents (< 10 tokens) without creating tensor
        if end - start >= 10:
            documents.append(tokens[start:end])

    return documents


# -----------------------------------------------------------------------------
# Filtering


def filter_by_scores(scores_path: str, keep_fraction: float) -> Set[int]:
    """
    Load scores and determine which documents to keep.

    Args:
        scores_path: Path to scores JSON file
        keep_fraction: Fraction of documents to keep (0 to 1)

    Returns:
        Set of document indices to keep
    """
    print(f"Loading scores from {scores_path}...")

    with open(scores_path, 'r') as f:
        data = json.load(f)

    scores = data['scores']
    num_docs = len(scores)
    num_keep = int(num_docs * keep_fraction)

    print(f"Total documents: {num_docs:,}")
    print(f"Keeping top {keep_fraction*100:.1f}% = {num_keep:,} documents")

    # Scores are already sorted by combined_score (descending) in scoring script
    # Just take the top num_keep
    keep_indices = {score['doc_idx'] for score in scores[:num_keep]}

    return keep_indices


# -----------------------------------------------------------------------------
# Writing Filtered Data


def write_filtered_bin(output_path: Path, tokens: torch.Tensor):
    """
    Write tokens to a .bin file in FineWeb format.

    Format:
    - 256 int32 header
      [0] = magic number (20240520)
      [1] = version (1)
      [2] = num_tokens
      [3:] = padding (zeros)
    - Followed by tokens as uint16
    """
    num_tokens = len(tokens)

    # Create header
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = MAGIC_NUMBER
    header[1] = VERSION
    header[2] = num_tokens

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('wb') as f:
        # Write header (256 int32 = 1024 bytes)
        f.write(header.numpy().tobytes())

        # Write tokens (uint16)
        f.write(tokens.numpy().tobytes())

    print(f"  Wrote {num_tokens:,} tokens to {output_path.name}")


def create_filtered_dataset(
    input_pattern: str,
    keep_indices: Set[int],
    output_dir: str,
    tokens_per_shard: int = 100_000_000,
):
    """
    Create filtered dataset by extracting selected documents.

    Args:
        input_pattern: Glob pattern for input .bin files
        keep_indices: Set of document indices to keep
        output_dir: Output directory for filtered .bin files
        tokens_per_shard: Approximate tokens per output shard
    """
    print(f"\nCreating filtered dataset...")
    print(f"Output directory: {output_dir}")

    files = sorted(glob.glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_pattern}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine base name (train or val)
    if 'train' in input_pattern:
        base_name = 'fineweb_train'
    elif 'val' in input_pattern:
        base_name = 'fineweb_val'
    else:
        base_name = 'fineweb_filtered'

    # Process all files and extract filtered documents
    all_filtered_tokens = []
    current_shard_tokens = []
    current_shard_size = 0  # Track size without recomputing
    shard_idx = 0
    doc_idx = 0
    total_docs_kept = 0
    total_docs_seen = 0
    total_tokens_processed = 0

    print(f"\nProcessing {len(files)} files with ~{len(keep_indices):,} documents to keep...")

    for file_idx, file_path in enumerate(files):
        print(f"\n[File {file_idx+1}/{len(files)}] Loading {Path(file_path).name}...")
        tokens = _load_data_shard(Path(file_path))
        total_tokens_processed += len(tokens)

        print(f"  Extracting documents from {len(tokens):,} tokens...")
        documents = extract_documents(tokens)

        print(f"  Processing {len(documents):,} documents...")

        # Use tqdm for document-level progress within each file
        for doc in tqdm(documents, desc=f"  File {file_idx+1}/{len(files)}", leave=False):
            if doc_idx in keep_indices:
                current_shard_tokens.append(doc)
                current_shard_size += len(doc)
                total_docs_kept += 1

                # Check if current shard is large enough
                if current_shard_size >= tokens_per_shard:
                    # Write current shard
                    shard_tokens = torch.cat(current_shard_tokens)
                    output_file = output_path / f"{base_name}_{shard_idx:06d}.bin"
                    write_filtered_bin(output_file, shard_tokens)

                    shard_idx += 1
                    current_shard_tokens = []
                    current_shard_size = 0

            doc_idx += 1
            total_docs_seen += 1

        # Print file completion stats
        kept_pct = (total_docs_kept / total_docs_seen * 100) if total_docs_seen > 0 else 0
        print(f"  ✓ File complete: kept {total_docs_kept:,}/{total_docs_seen:,} docs ({kept_pct:.1f}%), {total_tokens_processed:,} tokens processed")

    # Write remaining tokens as final shard
    if current_shard_tokens:
        shard_tokens = torch.cat(current_shard_tokens)
        output_file = output_path / f"{base_name}_{shard_idx:06d}.bin"
        write_filtered_bin(output_file, shard_tokens)
        shard_idx += 1

    print(f"\n{'='*60}")
    print(f"FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total documents seen: {total_docs_seen:,}")
    print(f"Documents kept: {total_docs_kept:,} ({total_docs_kept/total_docs_seen*100:.1f}%)")
    print(f"Total tokens processed: {total_tokens_processed:,}")
    print(f"Output shards: {shard_idx}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Filter FineWeb data by entropy scores")

    parser.add_argument("--scores", type=str, required=True,
                        help="Path to scores JSON file")
    parser.add_argument("--input_pattern", type=str, required=True,
                        help="Glob pattern for input .bin files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for filtered .bin files")
    parser.add_argument("--keep_fraction", type=float, default=0.80,
                        help="Fraction of documents to keep (default: 0.80)")
    parser.add_argument("--tokens_per_shard", type=int, default=100_000_000,
                        help="Approximate tokens per output shard (default: 100M)")

    args = parser.parse_args()

    # Handle DATA_PATH environment variable
    data_path = os.environ.get("DATA_PATH", ".")
    input_pattern = os.path.join(data_path, args.input_pattern)

    print("="*60)
    print("FINEWEB DATA FILTERING")
    print("="*60)
    print(f"Scores: {args.scores}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output directory: {args.output_dir}")
    print(f"Keep fraction: {args.keep_fraction}")
    print("="*60)

    # Step 1: Load scores and determine which documents to keep
    keep_indices = filter_by_scores(args.scores, args.keep_fraction)

    # Step 2: Create filtered dataset
    create_filtered_dataset(
        input_pattern=input_pattern,
        keep_indices=keep_indices,
        output_dir=args.output_dir,
        tokens_per_shard=args.tokens_per_shard,
    )

    print("\n✓ Filtering complete!")
    print(f"\nTo use filtered data, update train_gpt_single.py:")
    print(f'  train_files: str = "{args.output_dir}/{os.path.basename(input_pattern)}"')


if __name__ == "__main__":
    main()
