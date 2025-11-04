# Training Configuration Files

This directory contains JSON configuration files for different training experiments.

## 📋 Available Configs

### `default.json`
Full production training configuration for GPT-2 Small (124M parameters):
- **Steps:** 2315 (2275 scheduled + 40 extension)
- **Data:** Full FineWeb dataset
- **Use case:** Standard training run

```bash
python train_gpt_single.py --config configs/default.json
```

### `validation_baseline.json`
Baseline validation for entropy filtering experiments:
- **Steps:** 1000
- **Data:** Raw FineWeb (100M tokens, unfiltered)
- **Val frequency:** Every 100 steps
- **Run ID:** baseline_raw
- **Use case:** Baseline comparison for filtered data experiments

```bash
python train_gpt_single.py --config configs/validation_baseline.json
```

### `validation_filtered.json`
Filtered data validation for entropy filtering experiments:
- **Steps:** 800 (20% less than baseline)
- **Data:** Filtered FineWeb (80M tokens, top 80% by entropy score)
- **Val frequency:** Every 100 steps
- **Run ID:** filtered_80pct
- **Use case:** Testing entropy-filtered dataset performance

```bash
python train_gpt_single.py --config configs/validation_filtered.json
```

## 🔧 Config File Format

```json
{
  "_comment": "Optional comment describing this config",
  "data": {
    "train_files": "data/fineweb10B/fineweb_train_*.bin",
    "val_files": "data/fineweb10B/fineweb_val_*.bin",
    "val_tokens": 10485760,
    "train_batch_size": 262144,
    "train_max_seq_len": 2048,
    "val_batch_size": 2097152
  },
  "training": {
    "num_scheduled_iterations": 2275,
    "num_extension_iterations": 40,
    "cooldown_frac": 0.45,
    "grad_accum_steps": 8
  },
  "logging": {
    "run_id": "my-experiment",
    "val_loss_every": 250,
    "save_checkpoint": false
  },
  "attention": {
    "block_size": 128,
    "ws_schedule": [3, 7, 11],
    "ws_validate": 13,
    "ws_validate_post_yarn_ext": 20
  }
}
```

## 📝 Field Descriptions

### Data Section
- **train_files**: Glob pattern for training data .bin files
- **val_files**: Glob pattern for validation data .bin files
- **val_tokens**: Number of validation tokens (keep fixed for fair comparisons)
- **train_batch_size**: Total tokens per training batch
- **train_max_seq_len**: Maximum sequence length for training
- **val_batch_size**: Total tokens per validation batch

### Training Section
- **num_scheduled_iterations**: Steps with learning rate schedule
- **num_extension_iterations**: Additional steps at final learning rate
- **cooldown_frac**: Fraction of scheduled steps spent cooling down LR
- **grad_accum_steps**: Number of gradient accumulation steps

### Logging Section
- **run_id**: Experiment identifier (null = generate UUID)
- **val_loss_every**: Validation frequency in steps (0 = only at end)
- **save_checkpoint**: Whether to save model checkpoints

### Attention Section
- **block_size**: Block size for attention
- **ws_schedule**: Window size schedule as list [early, mid, late]
- **ws_validate**: Window size for validation
- **ws_validate_post_yarn_ext**: Extended window size after YaRN

## ✨ Creating New Configs

To create a custom config:

1. Copy an existing config file:
   ```bash
   cp configs/default.json configs/my_experiment.json
   ```

2. Edit the values:
   ```bash
   vim configs/my_experiment.json
   ```

3. Run with your config:
   ```bash
   python train_gpt_single.py --config configs/my_experiment.json
   ```

## 🔄 Config Merging Behavior

- Missing fields use defaults from `Hyperparameters` dataclass in `train_gpt_single.py`
- `num_iterations` is automatically calculated as `num_scheduled_iterations + num_extension_iterations`
- `run_id: null` will auto-generate a UUID
- `ws_schedule` arrays are converted to tuples internally

## 🎯 Environment Variables

The config system respects environment variables:
- **DATA_PATH**: Prepended to train_files and val_files paths
- Example: `DATA_PATH=/mnt/data python train_gpt_single.py --config configs/default.json`

## 📊 Validation Pipeline

The validation pipeline uses these configs:

```bash
# Automated validation pipeline
bash scripts/run_complete_validation.sh

# Manual validation
python train_gpt_single.py --config configs/validation_baseline.json
python train_gpt_single.py --config configs/validation_filtered.json
```

## 💡 Tips

- **Quick experiments**: Modify num_scheduled_iterations for shorter runs
- **Memory tuning**: Adjust train_batch_size for your GPU
- **Debugging**: Set val_loss_every=10 for frequent validation checks
- **Production**: Use save_checkpoint=true to save model weights

---

For more details, see the main [README.md](../README.md) and [ENTROPY_FILTERING_GUIDE.md](../local_reference/ENTROPY_FILTERING_GUIDE.md).
