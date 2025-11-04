#!/bin/bash

# Entropy Filtering Validation Pipeline
# This script runs the complete validation on 100M token subset

set -e  # Exit on error

echo "========================================================================"
echo "ENTROPY-BASED DATA FILTERING VALIDATION PIPELINE"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Train probe model on 100M tokens (~30-45 min)"
echo "  2. Score documents by entropy (~1-2 hours)"
echo "  3. Filter dataset to keep top 80% (~10-15 min)"
echo "  4. Instructions for manual training validation"
echo ""
echo "Total automated time: ~2-3 hours"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Configuration
NUM_TOKENS=100000000
NUM_VAL_TOKENS=10000000
KEEP_FRACTION=0.80
ALPHA=0.5

CHECKPOINT_DIR="checkpoints"
DATA_DIR="data"
PROBE_CHECKPOINT="${CHECKPOINT_DIR}/probe_model_100M.pt"
SCORES_FILE="${DATA_DIR}/fineweb_scores_100M.json"
OUTPUT_DIR="${DATA_DIR}/fineweb10B_filtered_80pct"

echo ""
echo "========================================="
echo "Configuration:"
echo "========================================="
echo "Training tokens: ${NUM_TOKENS}"
echo "Keep fraction: ${KEEP_FRACTION} (prune $(echo "scale=0; (1-${KEEP_FRACTION})*100" | bc)%)"
echo "Alpha (NLL weight): ${ALPHA}"
echo "Probe checkpoint: ${PROBE_CHECKPOINT}"
echo "Scores output: ${SCORES_FILE}"
echo "Filtered data: ${OUTPUT_DIR}"
echo "========================================="
echo ""

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${DATA_DIR}"

# Step 1: Train probe model
echo ""
echo "========================================================================"
echo "STEP 1: Training Probe Model"
echo "========================================================================"
echo "Training a 50M parameter GPT on first 100M tokens..."
echo "Expected time: 30-45 minutes"
echo ""

python train_probe_model.py \
    --num_tokens ${NUM_TOKENS} \
    --num_val_tokens ${NUM_VAL_TOKENS} \
    --num_steps 5000 \
    --batch_size 8 \
    --seq_len 512 \
    --lr 3e-4 \
    --warmup_steps 100 \
    --eval_every 100 \
    --patience 5 \
    --checkpoint_path "${PROBE_CHECKPOINT}"

if [ ! -f "${PROBE_CHECKPOINT}" ]; then
    echo "ERROR: Probe model checkpoint not found at ${PROBE_CHECKPOINT}"
    exit 1
fi

echo ""
echo "✓ Probe model training complete!"
echo ""

# Step 2: Score documents
echo ""
echo "========================================================================"
echo "STEP 2: Scoring Documents"
echo "========================================================================"
echo "Scoring all documents using probe model..."
echo "Expected time: 1-2 hours"
echo ""

python data/score_fineweb_entropy.py \
    --data_pattern "data/fineweb10B/fineweb_train_*.bin" \
    --checkpoint "${PROBE_CHECKPOINT}" \
    --output "${SCORES_FILE}" \
    --num_tokens ${NUM_TOKENS} \
    --alpha ${ALPHA}

if [ ! -f "${SCORES_FILE}" ]; then
    echo "ERROR: Scores file not found at ${SCORES_FILE}"
    exit 1
fi

echo ""
echo "✓ Document scoring complete!"
echo ""

# Step 3: Filter dataset
echo ""
echo "========================================================================"
echo "STEP 3: Filtering Dataset"
echo "========================================================================"
echo "Creating filtered dataset (keeping top ${KEEP_FRACTION})..."
echo "Expected time: 10-15 minutes"
echo ""

python data/filter_fineweb.py \
    --scores "${SCORES_FILE}" \
    --input_pattern "data/fineweb10B/fineweb_train_*.bin" \
    --output_dir "${OUTPUT_DIR}" \
    --keep_fraction ${KEEP_FRACTION}

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Filtered data directory not found at ${OUTPUT_DIR}"
    exit 1
fi

echo ""
echo "✓ Dataset filtering complete!"
echo ""

# Step 4: Manual validation instructions
echo ""
echo "========================================================================"
echo "STEP 4: Manual Training Validation Required"
echo "========================================================================"
echo ""
echo "The automated pipeline is complete! Now you need to manually run two"
echo "training experiments to compare performance:"
echo ""
echo "📊 EXPERIMENT A: Baseline (Raw Data)"
echo "-----------------------------------"
echo "1. Edit train_gpt_single.py:"
echo "   - train_files = 'data/fineweb10B/fineweb_train_*.bin'"
echo "   - num_scheduled_iterations = 1000"
echo "   - num_extension_iterations = 0"
echo ""
echo "2. Run training:"
echo "   python train_gpt_single.py"
echo ""
echo "3. Record final validation loss"
echo ""
echo "📊 EXPERIMENT B: Filtered Data"
echo "-----------------------------------"
echo "1. Edit train_gpt_single.py:"
echo "   - train_files = '${OUTPUT_DIR}/fineweb_train_*.bin'"
echo "   - num_scheduled_iterations = 800  # 20% less"
echo "   - num_extension_iterations = 0"
echo ""
echo "2. Run training:"
echo "   python train_gpt_single.py"
echo ""
echo "3. Record final validation loss"
echo ""
echo "📈 COMPARISON"
echo "-----------------------------------"
echo "Compare the two experiments:"
echo ""
echo "| Metric         | Baseline | Filtered | Change  |"
echo "|----------------|----------|----------|---------|"
echo "| Training time  | ~10 min  | ~8 min   | -20%    |"
echo "| Final val loss | [RECORD] | [RECORD] | [CALC]  |"
echo ""
echo "✅ SUCCESS CRITERIA:"
echo "   - Filtered model matches or beats baseline val loss"
echo "   - Training time reduced by ~20%"
echo ""
echo "If successful, scale to full 900M token dataset!"
echo "See local_reference/ENTROPY_FILTERING_GUIDE.md for details"
echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - Probe model: ${PROBE_CHECKPOINT}"
echo "  - Scores:      ${SCORES_FILE}"
echo "  - Filtered data: ${OUTPUT_DIR}/"
echo ""
echo "Next: Run manual training experiments (see instructions above)"
echo ""
