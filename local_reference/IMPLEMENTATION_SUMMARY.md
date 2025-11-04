# Entropy-Based Filtering Implementation Summary

## 📁 Files Created

### Core Scripts

1. **`train_probe_model.py`** (Root directory)
   - Trains a small 50M parameter GPT model on subset of data
   - Uses simple Adam optimizer (no Muon complexity)
   - Early stopping based on validation loss
   - Saves checkpoint for downstream scoring
   - ~50M parameters (vs 124M in main model)

2. **`data/score_fineweb_entropy.py`** (Data directory)
   - Loads trained probe model
   - Splits token stream into documents (using BOS separators)
   - Scores each document by:
     * NLL (negative log-likelihood) from probe model
     * Inverse word frequency (statistical measure)
   - Combines scores and ranks documents
   - Outputs JSON with all document scores

3. **`data/filter_fineweb.py`** (Data directory)
   - Loads scores JSON
   - Filters to keep top K% of documents (default 80%)
   - Extracts filtered documents from original .bin files
   - Writes new .bin files in same format
   - Maintains compatibility with existing training code

### Documentation

4. **`local_reference/ENTROPY_FILTERING_GUIDE.md`**
   - Comprehensive usage guide
   - Step-by-step instructions for validation
   - Tuning parameters and troubleshooting
   - References to research papers

5. **`scripts/run_entropy_validation.sh`**
   - Automated validation pipeline script
   - Runs all 3 steps: train → score → filter
   - Provides instructions for manual training comparison

## 🎯 Implementation Approach

### Conservative & Research-Backed

- **Conservative pruning**: 20% (keep 80%) as starting point
- **Advanced scoring**: Probe model + word frequency (matches research)
- **Validation first**: Test on 100M subset before scaling
- **FineWeb-specific**: Optimized for current dataset format

### Key Design Decisions

1. **Probe Model Architecture**
   - Simplified GPT (no fancy optimizations)
   - 6 layers, 384 dim (vs 12 layers, 768 dim in main)
   - ~50M params for fast training (~30-45 min)

2. **Scoring Method**
   - Combined score = `alpha * NLL + (1-alpha) * word_freq`
   - Default alpha=0.5 (balanced)
   - Tunable for different priorities

3. **Filtering Strategy**
   - Document-level (not token-level)
   - Uses BOS_ID=50256 as separator
   - Maintains .bin file format for compatibility

## 📊 Expected Results

Based on research ([arXiv:2406.14124](https://arxiv.org/html/2406.14124)):

| Keep % | Tokens | Training Time | Expected Performance |
|--------|--------|---------------|---------------------|
| 80% | 720M | -20% | Match or beat baseline |
| 70% | 630M | -30% | Slight drop acceptable |
| 60% | 540M | -40% | Noticeable drop |

**Recommendation**: Start with 80% (conservative)

## 🚀 Usage

### Quick Start (Automated)

```bash
# Run complete validation pipeline (~2-3 hours automated)
bash scripts/run_entropy_validation.sh
```

### Manual Steps

See `local_reference/ENTROPY_FILTERING_GUIDE.md` for detailed instructions.

**Summary**:
1. Train probe model (30-45 min)
2. Score documents (1-2 hours)
3. Filter dataset (10-15 min)
4. Compare training on raw vs filtered (manual)

## 🔬 Research Foundation

### Primary Paper
**"Measuring Sample Importance in Data Pruning for LLMs"** (arXiv:2406.14124)
- Combines NLL and word frequency scoring
- Shows 50% pruning maintains performance
- Validated on large-scale LLM training

### Supporting Research
1. **Ultra-FineWeb** (arXiv:2505.05427): FastText filtering
2. **Perplexity Pruning** (arXiv:2405.20541): 1.14× speedup
3. **Reddit "Oren" project**: 30% savings with entropy filtering

## ✅ Implementation Checklist

- [x] Probe model training script with early stopping
- [x] Document scoring with NLL + word frequency
- [x] Filtering script maintaining .bin format
- [x] Comprehensive documentation
- [x] Automated validation pipeline
- [ ] Run validation on 100M subset (manual)
- [ ] Compare baseline vs filtered (manual)
- [ ] Scale to full dataset if successful (manual)

## 🎓 Key Concepts

### Entropy Scoring
- **High entropy** = Surprising/informative content
- **Low entropy** = Predictable/redundant content
- **Goal**: Keep high-entropy samples, prune low-entropy

### NLL (Negative Log-Likelihood)
- Measures how surprising text is to the probe model
- Higher NLL = more informative
- Context-dependent (considers word sequences)

### Word Frequency
- Measures rarity of individual words
- Higher score = more unusual vocabulary
- Context-independent (just word counts)

### Combined Scoring
- Balances model-based and statistical signals
- Alpha parameter controls the mix
- More robust than either alone

## 🛠 Customization Options

### Probe Model Size

**Current (balanced)**:
- 6 layers, 384 dim, 6 heads
- ~50M params
- ~30-45 min training

**Smaller (faster)**:
```bash
--num_layers 4 --model_dim 256 --num_heads 4
```

**Larger (more accurate)**:
```bash
--num_layers 8 --model_dim 512 --num_heads 8
```

### Scoring Alpha

```bash
# Model-only (NLL)
--alpha 1.0

# Balanced (recommended)
--alpha 0.5

# Frequency-only (no model)
--alpha 0.0
```

### Keep Fraction

```bash
# Conservative (safest)
--keep_fraction 0.80

# Moderate
--keep_fraction 0.70

# Aggressive
--keep_fraction 0.60
```

## 📈 Next Steps

1. **Run validation** (follow ENTROPY_FILTERING_GUIDE.md)
2. **Compare results** (baseline vs filtered)
3. **If successful**: Scale to full 900M tokens
4. **Document findings** in README.md
5. **Experiment with**: Different alphas, keep fractions, probe sizes

## 🔗 Quick Links

- **Main Guide**: `local_reference/ENTROPY_FILTERING_GUIDE.md`
- **Validation Script**: `scripts/run_entropy_validation.sh`
- **Research Paper**: https://arxiv.org/html/2406.14124
- **Original Idea**: `local_reference/idea.md`

---

Implementation complete! Ready for validation. 🚀
