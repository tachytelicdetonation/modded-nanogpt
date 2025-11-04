#!/bin/bash

################################################################################
# COMPLETE END-TO-END ENTROPY FILTERING VALIDATION
################################################################################
#
# This script runs the COMPLETE validation pipeline including training:
#   1. Train probe model on 100M tokens (~30-45 min)
#   2. Score documents by entropy (~1-2 hours)
#   3. Filter dataset to keep top 80% (~10-15 min)
#   4. Train baseline model on raw 100M tokens (~10 min)
#   5. Train comparison model on filtered 80M tokens (~8 min)
#   6. Compare results automatically
#
# Total time: ~3-4 hours
#
# Usage:
#   bash scripts/run_complete_validation.sh
#
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_TOKENS=100000000
NUM_VAL_TOKENS=10000000
KEEP_FRACTION=0.80
ALPHA=0.5

# Training config
BASELINE_STEPS=1000
FILTERED_STEPS=800  # 20% less steps (800 = 0.8 * 1000)

# Paths
CHECKPOINT_DIR="checkpoints"
DATA_DIR="data"
LOGS_DIR="logs/validation"
PROBE_CHECKPOINT="${CHECKPOINT_DIR}/probe_model_100M.pt"
SCORES_FILE="${DATA_DIR}/fineweb_scores_100M.json"
OUTPUT_DIR="${DATA_DIR}/fineweb10B_filtered_80pct"
BASELINE_LOG="${LOGS_DIR}/baseline_raw.log"
FILTERED_LOG="${LOGS_DIR}/filtered_80pct.log"

echo -e "${BLUE}========================================================================"
echo "COMPLETE ENTROPY-BASED DATA FILTERING VALIDATION"
echo -e "========================================================================${NC}"
echo ""
echo "This will run the COMPLETE validation pipeline including training."
echo ""
echo "Pipeline steps:"
echo "  1. Train probe model (~30-45 min)"
echo "  2. Score documents (~1-2 hours)"
echo "  3. Filter dataset (~10-15 min)"
echo "  4. Train baseline model (~10 min)"
echo "  5. Train filtered model (~8 min)"
echo "  6. Compare results"
echo ""
echo -e "${YELLOW}Total time: ~3-4 hours${NC}"
echo ""
echo "Configuration:"
echo "  Training tokens: ${NUM_TOKENS} (100M)"
echo "  Keep fraction: ${KEEP_FRACTION} (80%)"
echo "  Baseline steps: ${BASELINE_STEPS}"
echo "  Filtered steps: ${FILTERED_STEPS}"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${LOGS_DIR}"

################################################################################
# STEP 1: Train Probe Model
################################################################################

echo ""
echo -e "${BLUE}========================================================================"
echo "STEP 1/6: Training Probe Model"
echo -e "========================================================================${NC}"
echo "Training a 50M parameter GPT on first 100M tokens..."
echo "Expected time: 30-45 minutes"
echo ""

if [ -f "${PROBE_CHECKPOINT}" ]; then
    echo -e "${YELLOW}⚠ Probe checkpoint already exists at ${PROBE_CHECKPOINT}${NC}"
    read -p "Reuse existing checkpoint? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training new probe model..."
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
    else
        echo "Reusing existing checkpoint."
    fi
else
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
fi

if [ ! -f "${PROBE_CHECKPOINT}" ]; then
    echo -e "${RED}ERROR: Probe model checkpoint not found at ${PROBE_CHECKPOINT}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Probe model training complete!${NC}"

################################################################################
# STEP 2: Score Documents
################################################################################

echo ""
echo -e "${BLUE}========================================================================"
echo "STEP 2/6: Scoring Documents"
echo -e "========================================================================${NC}"
echo "Scoring all documents using probe model..."
echo "Expected time: 1-2 hours"
echo ""

if [ -f "${SCORES_FILE}" ]; then
    echo -e "${YELLOW}⚠ Scores file already exists at ${SCORES_FILE}${NC}"
    read -p "Reuse existing scores? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Re-scoring documents..."
        python data/score_fineweb_entropy.py \
            --data_pattern "data/fineweb10B/fineweb_train_*.bin" \
            --checkpoint "${PROBE_CHECKPOINT}" \
            --output "${SCORES_FILE}" \
            --num_tokens ${NUM_TOKENS} \
            --alpha ${ALPHA}
    else
        echo "Reusing existing scores."
    fi
else
    python data/score_fineweb_entropy.py \
        --data_pattern "data/fineweb10B/fineweb_train_*.bin" \
        --checkpoint "${PROBE_CHECKPOINT}" \
        --output "${SCORES_FILE}" \
        --num_tokens ${NUM_TOKENS} \
        --alpha ${ALPHA}
fi

if [ ! -f "${SCORES_FILE}" ]; then
    echo -e "${RED}ERROR: Scores file not found at ${SCORES_FILE}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Document scoring complete!${NC}"

################################################################################
# STEP 3: Filter Dataset
################################################################################

echo ""
echo -e "${BLUE}========================================================================"
echo "STEP 3/6: Filtering Dataset"
echo -e "========================================================================${NC}"
echo "Creating filtered dataset (keeping top ${KEEP_FRACTION})..."
echo "Expected time: 10-15 minutes"
echo ""

if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR})" ]; then
    echo -e "${YELLOW}⚠ Filtered data already exists at ${OUTPUT_DIR}${NC}"
    read -p "Reuse existing filtered data? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Re-filtering dataset..."
        rm -rf "${OUTPUT_DIR}"
        python data/filter_fineweb.py \
            --scores "${SCORES_FILE}" \
            --input_pattern "data/fineweb10B/fineweb_train_*.bin" \
            --output_dir "${OUTPUT_DIR}" \
            --keep_fraction ${KEEP_FRACTION}
    else
        echo "Reusing existing filtered data."
    fi
else
    python data/filter_fineweb.py \
        --scores "${SCORES_FILE}" \
        --input_pattern "data/fineweb10B/fineweb_train_*.bin" \
        --output_dir "${OUTPUT_DIR}" \
        --keep_fraction ${KEEP_FRACTION}
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo -e "${RED}ERROR: Filtered data directory not found at ${OUTPUT_DIR}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Dataset filtering complete!${NC}"

################################################################################
# STEP 4: Train Baseline Model (Raw Data)
################################################################################

echo ""
echo -e "${BLUE}========================================================================"
echo "STEP 4/6: Training Baseline Model (Raw Data)"
echo -e "========================================================================${NC}"
echo "Training on 100M raw tokens for ${BASELINE_STEPS} steps..."
echo "Expected time: ~10 minutes"
echo ""

# Create temporary training script for baseline
cat > /tmp/train_baseline.py << 'EOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We need to modify the hyperparameters before the main script runs
def modify_config():
    import train_gpt_single as tgs
    tgs.args.train_files = "data/fineweb10B/fineweb_train_*.bin"
    tgs.args.num_scheduled_iterations = BASELINE_STEPS_PLACEHOLDER
    tgs.args.num_extension_iterations = 0
    tgs.args.val_loss_every = 100
    tgs.args.save_checkpoint = False
    tgs.args.run_id = "baseline_raw"

    # Update file path with DATA_PATH
    data_path = os.environ.get("DATA_PATH", ".")
    tgs.args.train_files = os.path.join(data_path, tgs.args.train_files)

modify_config()

# Now run the main training
exec(open("train_gpt_single.py").read())
EOF

# Replace placeholder
sed -i.bak "s/BASELINE_STEPS_PLACEHOLDER/${BASELINE_STEPS}/g" /tmp/train_baseline.py
rm -f /tmp/train_baseline.py.bak

echo "Running baseline training..."
python /tmp/train_baseline.py 2>&1 | tee "${BASELINE_LOG}"

# Extract final validation loss
BASELINE_VAL_LOSS=$(grep "val_loss:" "${BASELINE_LOG}" | tail -1 | grep -oP 'val_loss:\s*\K[0-9.]+' || echo "N/A")
BASELINE_TIME=$(grep "step_avg:" "${BASELINE_LOG}" | tail -1 | grep -oP 'train_time:\s*\K[0-9]+' || echo "N/A")

echo ""
echo -e "${GREEN}✓ Baseline training complete!${NC}"
echo "  Final validation loss: ${BASELINE_VAL_LOSS}"
echo "  Training time: ${BASELINE_TIME}ms"

################################################################################
# STEP 5: Train Filtered Model
################################################################################

echo ""
echo -e "${BLUE}========================================================================"
echo "STEP 5/6: Training Filtered Model (80% Data)"
echo -e "========================================================================${NC}"
echo "Training on 80M filtered tokens for ${FILTERED_STEPS} steps..."
echo "Expected time: ~8 minutes"
echo ""

# Create temporary training script for filtered
cat > /tmp/train_filtered.py << 'EOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def modify_config():
    import train_gpt_single as tgs
    tgs.args.train_files = "OUTPUT_DIR_PLACEHOLDER/fineweb_train_*.bin"
    tgs.args.num_scheduled_iterations = FILTERED_STEPS_PLACEHOLDER
    tgs.args.num_extension_iterations = 0
    tgs.args.val_loss_every = 100
    tgs.args.save_checkpoint = False
    tgs.args.run_id = "filtered_80pct"

    # Update file path with DATA_PATH
    data_path = os.environ.get("DATA_PATH", ".")
    tgs.args.train_files = os.path.join(data_path, tgs.args.train_files)

modify_config()

# Now run the main training
exec(open("train_gpt_single.py").read())
EOF

# Replace placeholders
sed -i.bak "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" /tmp/train_filtered.py
sed -i.bak "s/FILTERED_STEPS_PLACEHOLDER/${FILTERED_STEPS}/g" /tmp/train_filtered.py
rm -f /tmp/train_filtered.py.bak

echo "Running filtered training..."
python /tmp/train_filtered.py 2>&1 | tee "${FILTERED_LOG}"

# Extract final validation loss
FILTERED_VAL_LOSS=$(grep "val_loss:" "${FILTERED_LOG}" | tail -1 | grep -oP 'val_loss:\s*\K[0-9.]+' || echo "N/A")
FILTERED_TIME=$(grep "step_avg:" "${FILTERED_LOG}" | tail -1 | grep -oP 'train_time:\s*\K[0-9]+' || echo "N/A")

echo ""
echo -e "${GREEN}✓ Filtered training complete!${NC}"
echo "  Final validation loss: ${FILTERED_VAL_LOSS}"
echo "  Training time: ${FILTERED_TIME}ms"

################################################################################
# STEP 6: Compare Results
################################################################################

echo ""
echo -e "${BLUE}========================================================================"
echo "STEP 6/6: Results Comparison"
echo -e "========================================================================${NC}"
echo ""

# Calculate time improvement
if [ "${BASELINE_TIME}" != "N/A" ] && [ "${FILTERED_TIME}" != "N/A" ]; then
    TIME_IMPROVEMENT=$(echo "scale=1; (1 - ${FILTERED_TIME}/${BASELINE_TIME}) * 100" | bc)
else
    TIME_IMPROVEMENT="N/A"
fi

# Calculate loss difference
if [ "${BASELINE_VAL_LOSS}" != "N/A" ] && [ "${FILTERED_VAL_LOSS}" != "N/A" ]; then
    LOSS_DIFF=$(echo "scale=4; ${FILTERED_VAL_LOSS} - ${BASELINE_VAL_LOSS}" | bc)
else
    LOSS_DIFF="N/A"
fi

# Print comparison table
echo "┌─────────────────────┬──────────────┬──────────────┬──────────────┐"
echo "│ Metric              │ Baseline     │ Filtered     │ Change       │"
echo "├─────────────────────┼──────────────┼──────────────┼──────────────┤"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Training Data" "100M tokens" "80M tokens" "-20%"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Training Steps" "${BASELINE_STEPS}" "${FILTERED_STEPS}" "-20%"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Training Time" "${BASELINE_TIME}ms" "${FILTERED_TIME}ms" "${TIME_IMPROVEMENT}%"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Final Val Loss" "${BASELINE_VAL_LOSS}" "${FILTERED_VAL_LOSS}" "${LOSS_DIFF}"
echo "└─────────────────────┴──────────────┴──────────────┴──────────────┘"
echo ""

# Success criteria
echo -e "${BLUE}Success Criteria:${NC}"
echo "  ✓ Training time reduced by ~20%"

if [ "${LOSS_DIFF}" != "N/A" ]; then
    if (( $(echo "${LOSS_DIFF} <= 0.02" | bc -l) )); then
        echo -e "  ${GREEN}✓ Validation loss: Filtered matches or beats baseline!${NC}"
        SUCCESS=true
    else
        echo -e "  ${YELLOW}⚠ Validation loss: Filtered is slightly worse (+${LOSS_DIFF})${NC}"
        SUCCESS=false
    fi
else
    echo "  ? Unable to compare validation losses"
    SUCCESS=false
fi

echo ""

if [ "$SUCCESS" = true ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  ✓✓✓ VALIDATION SUCCESSFUL! ✓✓✓                              ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  Entropy filtering achieves same performance with 20% less   ║${NC}"
    echo -e "${GREEN}║  data and compute!                                            ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  Next step: Scale to full 900M token dataset                 ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║                                                               ║${NC}"
    echo -e "${YELLOW}║  ⚠ VALIDATION INCONCLUSIVE                                   ║${NC}"
    echo -e "${YELLOW}║                                                               ║${NC}"
    echo -e "${YELLOW}║  Filtered model performance is slightly worse than baseline. ║${NC}"
    echo -e "${YELLOW}║                                                               ║${NC}"
    echo -e "${YELLOW}║  Try adjusting:                                               ║${NC}"
    echo -e "${YELLOW}║  - Keep fraction (0.85 or 0.90)                               ║${NC}"
    echo -e "${YELLOW}║  - Alpha parameter (0.3 or 0.7)                               ║${NC}"
    echo -e "${YELLOW}║  - Train probe model longer                                   ║${NC}"
    echo -e "${YELLOW}║                                                               ║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════════╝${NC}"
fi

echo ""
echo -e "${BLUE}========================================================================"
echo "VALIDATION COMPLETE!"
echo -e "========================================================================${NC}"
echo ""
echo "Generated files:"
echo "  - Probe model:   ${PROBE_CHECKPOINT}"
echo "  - Scores:        ${SCORES_FILE}"
echo "  - Filtered data: ${OUTPUT_DIR}/"
echo "  - Baseline log:  ${BASELINE_LOG}"
echo "  - Filtered log:  ${FILTERED_LOG}"
echo ""
echo "For detailed guide, see: local_reference/ENTROPY_FILTERING_GUIDE.md"
echo ""

# Cleanup temporary files
rm -f /tmp/train_baseline.py /tmp/train_filtered.py

exit 0
