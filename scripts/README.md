# Scripts Directory

Automation scripts for the modded-nanogpt project.

## 🚀 Entropy Filtering Validation Scripts

### `run_complete_validation.sh` ⭐ **RECOMMENDED**

**Complete end-to-end validation pipeline** - Fully automated!

Runs everything from start to finish:
1. ✅ Train probe model (~30-45 min)
2. ✅ Score documents (~1-2 hours)
3. ✅ Filter dataset (~10-15 min)
4. ✅ Train baseline model (~10 min)
5. ✅ Train filtered model (~8 min)
6. ✅ Compare results automatically

**Usage:**
```bash
bash scripts/run_complete_validation.sh
```

**Total time:** ~3-4 hours (fully automated, no manual steps!)

**Output:**
- Probe model checkpoint
- Document scores JSON
- Filtered dataset
- Training logs for both experiments
- Automated comparison table

**Features:**
- ✅ Checks for existing files (can resume)
- ✅ Color-coded output
- ✅ Progress indicators
- ✅ Automatic result comparison
- ✅ Success/failure criteria check

---

### `run_entropy_validation.sh`

**Partial automation** - Stops before training experiments.

Runs only the data preparation steps:
1. ✅ Train probe model
2. ✅ Score documents
3. ✅ Filter dataset
4. ❌ Manual training required

**Usage:**
```bash
bash scripts/run_entropy_validation.sh
```

**When to use:**
- If you want to manually control training experiments
- If you want to inspect filtered data before training
- If you need to customize training parameters

---

## 📊 Comparison

| Feature | run_complete_validation.sh | run_entropy_validation.sh |
|---------|---------------------------|---------------------------|
| Probe training | ✅ Auto | ✅ Auto |
| Document scoring | ✅ Auto | ✅ Auto |
| Dataset filtering | ✅ Auto | ✅ Auto |
| Baseline training | ✅ Auto | ❌ Manual |
| Filtered training | ✅ Auto | ❌ Manual |
| Result comparison | ✅ Auto | ❌ Manual |
| **Total automation** | **100%** | **60%** |

---

## 🎯 Quick Start

**For hands-off validation:**
```bash
# Just run this and come back in 3-4 hours!
bash scripts/run_complete_validation.sh
```

**For step-by-step control:**
```bash
# Run data prep only
bash scripts/run_entropy_validation.sh

# Then manually run training experiments
# (see local_reference/ENTROPY_FILTERING_GUIDE.md)
```

---

## 📁 Output Files

Both scripts create:

```
checkpoints/
  └── probe_model_100M.pt              # Trained probe model

data/
  ├── fineweb_scores_100M.json         # Document scores
  └── fineweb10B_filtered_80pct/       # Filtered dataset
      ├── fineweb_train_000000.bin
      ├── fineweb_train_000001.bin
      └── ...

logs/validation/                        # Only in complete script
  ├── baseline_raw.log                  # Baseline training log
  └── filtered_80pct.log                # Filtered training log
```

---

## ⚙️ Configuration

Edit the scripts to customize:

**Data size:**
```bash
NUM_TOKENS=100000000        # 100M for quick validation
NUM_TOKENS=900000000        # 900M for full dataset
```

**Filtering aggressiveness:**
```bash
KEEP_FRACTION=0.80          # Conservative (20% pruning)
KEEP_FRACTION=0.70          # Moderate (30% pruning)
KEEP_FRACTION=0.60          # Aggressive (40% pruning)
```

**Training steps:**
```bash
BASELINE_STEPS=1000         # Quick validation
BASELINE_STEPS=2315         # Full training
```

**Scoring method:**
```bash
ALPHA=0.5                   # Balanced (NLL + word freq)
ALPHA=1.0                   # Model-only (NLL)
ALPHA=0.0                   # Frequency-only (no model)
```

---

## 🔧 Advanced Usage

### Resume from checkpoint

Both scripts check for existing files:
- If probe checkpoint exists: Option to reuse
- If scores exist: Option to reuse
- If filtered data exists: Option to reuse

Just re-run the script!

### Parallel scoring

For faster scoring on multiple GPUs:
```bash
# Edit score_fineweb_entropy.py to process shards in parallel
# Or run multiple scoring jobs on different data shards
```

### Custom parameters

```bash
# Quick test on tiny dataset
NUM_TOKENS=10000000 bash scripts/run_complete_validation.sh

# More aggressive pruning
KEEP_FRACTION=0.60 bash scripts/run_complete_validation.sh
```

---

## 📚 Documentation

For detailed information:
- **Usage guide:** `local_reference/ENTROPY_FILTERING_GUIDE.md`
- **Implementation:** `local_reference/IMPLEMENTATION_SUMMARY.md`
- **Research:** `local_reference/idea.md`

---

## ✅ Expected Results

After running `run_complete_validation.sh`, you should see:

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric              │ Baseline     │ Filtered     │ Change       │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Training Data       │ 100M tokens  │ 80M tokens   │ -20%         │
│ Training Steps      │ 1000         │ 800          │ -20%         │
│ Training Time       │ ~600000ms    │ ~480000ms    │ -20%         │
│ Final Val Loss      │ 3.65         │ 3.63         │ -0.02 ✓      │
└─────────────────────┴──────────────┴──────────────┴──────────────┘

✓✓✓ VALIDATION SUCCESSFUL! ✓✓✓

Entropy filtering achieves same performance with 20% less data and compute!
```

---

## 🐛 Troubleshooting

**Script fails with "file not found":**
- Check that you have FineWeb data: `ls data/fineweb10B/`
- Download data: `python data/cached_fineweb10B.py 9`

**Out of memory:**
- Reduce probe model batch size (edit script)
- Use smaller validation dataset

**Training fails:**
- Check logs in `logs/validation/`
- Ensure train_gpt_single.py works standalone first

---

## 🎉 Success!

If validation succeeds, scale to full dataset:
1. Edit script: `NUM_TOKENS=900000000`
2. Run: `bash scripts/run_complete_validation.sh`
3. Wait ~12 hours for full pipeline
4. Enjoy 20% faster training! 🚀
