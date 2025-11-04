# Entropy-Based Data Filtering Pipeline

This guide walks through the complete pipeline for filtering FineWeb training data using entropy-based scoring. The goal is to achieve similar or better model performance while using 20-30% less training data.

## 📋 Overview

The pipeline consists of 4 main steps:

1. **Train Probe Model** - Train a small GPT model on a subset of data
2. **Score Documents** - Use probe model to score all documents by "information content"
3. **Filter Dataset** - Keep only top K% of documents by score
4. **Train & Validate** - Compare training on filtered vs raw data

## 🎯 Expected Results

Based on research ([arXiv:2406.14124](https://arxiv.org/html/2406.14124)):
- **20% pruning (keep 80%)**: Should match baseline performance
- **30% pruning (keep 70%)**: Slight performance drop, significant speedup
- **40% pruning (keep 60%)**: Noticeable performance drop

**Conservative approach**: Start with 20% pruning (keep 80%)

---

## 📦 Prerequisites

```bash
# Ensure you have the required packages
pip install torch tqdm transformers  # Already installed for main training
```

---

## 🚀 Phase 1: Validation on 100M Token Subset

### Step 1: Train Probe Model

Train a small 50M parameter model on the first 100M tokens of FineWeb:

```bash
python train_probe_model.py \\
    --num_tokens 100000000 \\
    --num_val_tokens 10000000 \\
    --num_steps 5000 \\
    --batch_size 8 \\
    --seq_len 512 \\
    --lr 3e-4 \\
    --eval_every 100 \\
    --patience 5 \\
    --checkpoint_path checkpoints/probe_model_100M.pt
```

**Expected time**: 30-45 minutes on H100

**Output**: `checkpoints/probe_model_100M.pt` (probe model checkpoint)

**What it does**:
- Trains a simplified 6-layer GPT (50M params)
- Uses Adam optimizer (simpler than NorMuon)
- Early stopping when validation loss plateaus
- Saves checkpoint for scoring

### Step 2: Score Documents

Use the trained probe model to score all documents in the 100M token subset:

```bash
python data/score_fineweb_entropy.py \\
    --data_pattern "data/fineweb10B/fineweb_train_*.bin" \\
    --checkpoint checkpoints/probe_model_100M.pt \\
    --output data/fineweb_scores_100M.json \\
    --num_tokens 100000000 \\
    --alpha 0.5
```

**Expected time**: 1-2 hours on H100

**Output**: `data/fineweb_scores_100M.json` (document scores)

**What it does**:
- Loads probe model
- Splits tokens into documents (using BOS separators)
- Computes two scores per document:
  1. **NLL (Negative Log-Likelihood)**: How surprising the text is to the probe model
  2. **Inverse Word Frequency**: How rare/unusual the words are
- Combines scores: `score = alpha * NLL + (1-alpha) * word_freq`
- Saves ranked list of documents

**Alpha parameter**:
- `alpha=1.0`: Only NLL (model-based scoring)
- `alpha=0.5`: Balanced (recommended)
- `alpha=0.0`: Only word frequency (no model needed)

### Step 3: Filter Dataset

Create filtered dataset keeping top 80% of documents:

```bash
python data/filter_fineweb.py \\
    --scores data/fineweb_scores_100M.json \\
    --input_pattern "data/fineweb10B/fineweb_train_*.bin" \\
    --output_dir data/fineweb10B_filtered_80pct \\
    --keep_fraction 0.80
```

**Expected time**: 10-15 minutes

**Output**: `data/fineweb10B_filtered_80pct/fineweb_train_*.bin` (filtered data)

**What it does**:
- Loads scores JSON
- Selects top 80% of documents by score
- Extracts those documents from original .bin files
- Writes new .bin files in same format (for compatibility)

### Step 4: Validation Training - Baseline

Train a small model on 100M **raw** tokens (baseline):

```bash
# Create a simple validation script
python -c "
import train_gpt_single as tgs
# Modify hyperparameters for quick validation run
# ... (or manually edit train_gpt_single.py)
"
```

**OR manually edit `train_gpt_single.py`** for validation:

```python
# In Hyperparameters class, temporarily change:
train_files: str = "data/fineweb10B/fineweb_train_*.bin"  # Raw data
num_scheduled_iterations: int = 1000  # Shorter run
num_extension_iterations: int = 0
val_loss_every: int = 100
```

Then run:

```bash
python train_gpt_single.py
```

**Expected time**: 8-10 minutes on H100

**Record**: Final validation loss (e.g., `val_loss: 3.65`)

### Step 5: Validation Training - Filtered

Train identical model on 80M **filtered** tokens:

```python
# In Hyperparameters class:
train_files: str = "data/fineweb10B_filtered_80pct/fineweb_train_*.bin"  # Filtered!
num_scheduled_iterations: int = 800  # 20% less (800 = 0.8 * 1000)
num_extension_iterations: int = 0
val_loss_every: int = 100
```

Then run:

```bash
python train_gpt_single.py
```

**Expected time**: 6-8 minutes on H100 (20% faster!)

**Record**: Final validation loss (e.g., `val_loss: 3.63`)

### Step 6: Compare Results

Compare the two runs:

| Metric | Baseline (100M raw) | Filtered (80M) | Change |
|--------|---------------------|----------------|--------|
| Training tokens | 100M | 80M | -20% |
| Training time | ~10 min | ~8 min | -20% |
| Final val loss | 3.65 | 3.63 | **-0.02 (better!)** |

**Success criteria**: Filtered model should match or beat baseline with 20% less data/time.

---

## 🎉 Phase 2: Full Dataset (if validation succeeds)

If Phase 1 validation shows filtering helps, scale to full 900M tokens:

### Step 1: Train Full Probe Model

```bash
python train_probe_model.py \\
    --num_tokens 135000000 \\
    --num_val_tokens 10000000 \\
    --num_steps 10000 \\
    --checkpoint_path checkpoints/probe_model_full.pt
```

**Time**: 1-1.5 hours

### Step 2: Score Full Dataset

```bash
python data/score_fineweb_entropy.py \\
    --data_pattern "data/fineweb10B/fineweb_train_*.bin" \\
    --checkpoint checkpoints/probe_model_full.pt \\
    --output data/fineweb_scores_full.json \\
    --alpha 0.5
```

**Time**: 2-3 hours

### Step 3: Filter Full Dataset

```bash
python data/filter_fineweb.py \\
    --scores data/fineweb_scores_full.json \\
    --input_pattern "data/fineweb10B/fineweb_train_*.bin" \\
    --output_dir data/fineweb10B_filtered_80pct \\
    --keep_fraction 0.80
```

**Time**: 30-45 minutes

### Step 4: Production Training

Update `train_gpt_single.py`:

```python
# In Hyperparameters class:
train_files: str = "data/fineweb10B_filtered_80pct/fineweb_train_*.bin"
# No other changes needed!
```

Run full training:

```bash
python train_gpt_single.py
```

**Expected**: Same validation loss, ~14-16 min training time (vs 18-20 min baseline)

---

## 🔧 Tuning Parameters

### Probe Model Size

**Smaller (faster but less accurate)**:
```bash
python train_probe_model.py --num_layers 4 --model_dim 256 --num_heads 4
```

**Larger (slower but more accurate)**:
```bash
python train_probe_model.py --num_layers 8 --model_dim 512 --num_heads 8
```

### Scoring Alpha

**Model-only (NLL)**:
```bash
python data/score_fineweb_entropy.py --alpha 1.0
```

**Frequency-only (no model)**:
```bash
python data/score_fineweb_entropy.py --alpha 0.0
```

**Balanced (recommended)**:
```bash
python data/score_fineweb_entropy.py --alpha 0.5
```

### Keep Fraction

**Conservative (safest)**:
```bash
python data/filter_fineweb.py --keep_fraction 0.80  # 20% pruning
```

**Moderate**:
```bash
python data/filter_fineweb.py --keep_fraction 0.70  # 30% pruning
```

**Aggressive**:
```bash
python data/filter_fineweb.py --keep_fraction 0.60  # 40% pruning
```

---

## 📊 Understanding Scores

### NLL Score (Negative Log-Likelihood)

- **High NLL** = Model finds text surprising/difficult to predict
- Indicates novel patterns, complex structure, rare combinations
- **Keep documents with high NLL** - they teach the model new things

### Word Frequency Score

- **High score** = Document contains rare/unusual words
- Indicates specialized vocabulary, technical content
- **Keep documents with high word freq score** - they expand vocabulary

### Combined Score

```
combined_score = alpha * NLL + (1 - alpha) * word_freq
```

- Balances model-based (NLL) and statistical (word freq) signals
- Higher combined score = more informative document

---

## ⚠️ Troubleshooting

### Out of Memory during Probe Training

Reduce batch size:
```bash
python train_probe_model.py --batch_size 4 --seq_len 256
```

### Scoring Too Slow

Use CPU for scoring (model is small):
```bash
CUDA_VISIBLE_DEVICES="" python data/score_fineweb_entropy.py ...
```

Or reduce alpha (less model inference):
```bash
python data/score_fineweb_entropy.py --alpha 0.0  # No model, frequency only
```

### Filtered Data Worse than Baseline

Try:
1. **Increase keep fraction**: Use 0.85 or 0.90
2. **Adjust alpha**: Try 0.3 (more frequency) or 0.7 (more NLL)
3. **Train probe model longer**: Increase `--num_steps`

---

## 📚 References

1. **Sample Importance Paper**: https://arxiv.org/html/2406.14124
   - Core methodology: H(W,q) + H(W,f) scoring
   - Shows 50% pruning maintains performance

2. **Ultra-FineWeb**: https://arxiv.org/abs/2505.05427
   - Fast classifier-based filtering approach
   - Shows significant benchmark improvements

3. **Perplexity Pruning**: https://arxiv.org/abs/2405.20541
   - 1.14× faster training with perplexity filtering

4. **Original Reddit post** (from idea.md):
   - 30% less data with entropy filtering
   - 700M → 500M tokens, same performance

---

## 🎯 Quick Commands Summary

**Complete validation pipeline**:

```bash
# 1. Train probe (30-45 min)
python train_probe_model.py \\
    --num_tokens 100000000 \\
    --checkpoint_path checkpoints/probe_model_100M.pt

# 2. Score documents (1-2 hours)
python data/score_fineweb_entropy.py \\
    --checkpoint checkpoints/probe_model_100M.pt \\
    --output data/fineweb_scores_100M.json \\
    --num_tokens 100000000

# 3. Filter dataset (10-15 min)
python data/filter_fineweb.py \\
    --scores data/fineweb_scores_100M.json \\
    --input_pattern "data/fineweb10B/fineweb_train_*.bin" \\
    --output_dir data/fineweb10B_filtered_80pct \\
    --keep_fraction 0.80

# 4. Compare training on raw vs filtered data
# ... (manual training runs)
```

**Total validation time**: ~3-4 hours for complete proof-of-concept

---

## ✅ Success Metrics

| Metric | Target |
|--------|--------|
| Validation loss | ≤ baseline |
| Training time | -20% vs baseline |
| Memory usage | Similar or lower |
| Convergence | No issues |

If all metrics pass → Scale to full dataset!

---

## 🚀 Next Steps

After successful validation:

1. ✅ Scale to full 900M token dataset
2. ✅ Run production training with filtered data
3. 📊 Document final results in README
4. 🔬 Experiment with:
   - Different keep fractions (70%, 60%)
   - Different alpha values (NLL vs frequency weight)
   - Larger probe models
   - Validation on different datasets

---

## 💡 Tips

1. **Save intermediate results**: Probe checkpoints and scores JSON are reusable
2. **Monitor memory**: Scoring loads entire documents into memory
3. **Parallel scoring**: Can process multiple data shards in parallel
4. **Checkpoint often**: Add `--save_checkpoint True` to training runs

---

Happy filtering! 🎉
