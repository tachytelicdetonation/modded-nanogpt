# Modded-NanoGPT (Single H100 Fork)

> **Note**: This is a **personal fork** for quick architecture experimentation and fun! 🚀
> **Not intended for leaderboard submissions.**

This repository contains a single-GPU adaptation of the incredible [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun by [@kellerjordan0](https://x.com/kellerjordan0) and the amazing community of contributors.

## What is this?

The original modded-nanogpt is optimized for **8x H100 GPUs** with distributed training to achieve world-record training speeds (currently **2.345 minutes** to reach 3.28 validation loss on FineWeb).

This fork adapts that code to run on a **single H100 GPU** for:
- 🧪 **Quick architecture testing** without needing 8 GPUs
- 🎯 **Educational purposes** - understanding the optimizations in a simpler setting
- 🎉 **Fun experimentation** with state-of-the-art techniques

## Key Changes

### Removed Distributed Training
- No more `torch.distributed` or `torchrun` required
- Simplified NorMuon and Adam optimizers for single-device operation
- Single-GPU data loading pipeline

### Kept All Optimizations
All the cutting-edge ML innovations are preserved:
- ✅ **NorMuon optimizer** with Polar Express orthogonalization
- ✅ **Flash Attention 3** with sliding window patterns
- ✅ **Modern architecture**: Rotary embeddings (YaRN), QK-Norm, ReLU²
- ✅ **U-net skip connections** and value embeddings
- ✅ **FP8 matmul** for lm_head, bfloat16 activations
- ✅ **Gradient accumulation** (8 steps by default)
- ✅ **H100 optimizations**: TF32 enabled, torch.compile

### Added H100 Single-GPU Optimizations
```python
# Enabled TF32 for faster H100 matrix operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## File Structure

```
modded-nanogpt/
├── train_gpt_single.py      # New: Single H100 training script
├── reference/               # Original distributed training scripts
│   ├── train_gpt.py        # GPT-2 Small (124M, 8xH100, 2.3min)
│   └── train_gpt_medium.py # GPT-2 Medium (350M, 8xH100, 25min)
├── data/                    # FineWeb dataset scripts
├── README.md               # This file
└── README.old.md          # Original README with full speedrun history
```

## Usage

### Requirements
- 1x NVIDIA H100 GPU (80GB)
- PyTorch 2.9+ (nightly recommended)
- CUDA 12.6+

### Installation
```bash
git clone <your-fork-url> && cd modded-nanogpt
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
```

### Download Data
```bash
# Downloads first 900M training tokens (~10GB)
python data/cached_fineweb10B.py 9
```

### Run Training
```bash
# Single H100 - GPT-2 Small (124M params)
python train_gpt_single.py
```

## Expected Performance

### GPT-2 Small (124M parameters)
- **Training time**: ~18-20 minutes on single H100
- **Target**: 3.28 validation loss
- **Memory usage**: ~5-6GB (plenty of room on 80GB H100)
- **Tokens**: 262,144 per batch (configurable)

### Comparison to 8xH100
- Original distributed: **2.3 minutes**
- Single H100: **~18-20 minutes**
- **~8x slower** (linear scaling as expected)

## Tuning for Your H100

The H100 has 80GB of memory - you can experiment with larger batch sizes!

Edit `train_gpt_single.py` line ~1243:
```python
# Default: 262,144 tokens (matches distributed version)
train_batch_size: int = 2048 * 16 * 8

# Experiment: Try up to 512K-2M tokens!
# May require learning rate adjustment
```

## Full Attribution

All credit goes to the original [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) contributors:

**Core Contributors**: [@kellerjordan0](https://x.com/kellerjordan0), [@jxbz](https://x.com/jxbz), [@KoszarskyB](https://x.com/KoszarskyB), [@Grad62304977](https://x.com/Grad62304977), [@YouJiacheng](https://x.com/YouJiacheng), [@leloykun](https://x.com/@leloykun), [@brendanh0gan](https://x.com/brendanh0gan), [@fernbear.bsky.social](https://bsky.app/profile/fernbear.bsky.social), [@varunneal](https://x.com/varunneal), [@classiclarryd](https://x.com/classiclarryd)

**And many more**: [@bozavlado](https://x.com/bozavlado), [@KonstantinWilleke](https://github.com/KonstantinWilleke), [@alexrgilbert](https://github.com/alexrgilbert), [@adricarda](https://github.com/adricarda), [@tuttyfrutyee](https://github.com/tuttyfrutyee), [@vdlad](https://github.com/vdlad), [@ryanyang0](https://x.com/ryanyang0), [@vagrawal](https://github.com/vagrawal), [@byronxu99](https://github.com/byronxu99), [@EmelyanenkoK](https://github.com/EmelyanenkoK), [@bernard24](https://github.com/bernard24), [@GusarichOnX](https://x.com/GusarichOnX), [@jadenj3o](https://x.com/jadenj3o), [@li_zichong](https://github.com/li-zichong)

This single-GPU adaptation simply reorganizes their brilliant work for easier experimentation!

## Key Papers & Techniques

1. **Muon optimizer**: https://kellerjordan.github.io/posts/muon/
2. **NorMuon (adaptive step sizing)**: https://arxiv.org/abs/2510.05491
3. **Polar Express orthogonalization**: https://arxiv.org/pdf/2505.16932
4. **Flash Attention**: https://arxiv.org/abs/2205.14135
5. **YaRN (Yet another RoPE extensioN)**: https://arxiv.org/abs/2309.00071
6. **Value Residual Learning**: https://arxiv.org/abs/2410.17897
7. **Gemma 2 (sliding window, softcapping)**: https://arxiv.org/abs/2408.00118

## Why Not Leaderboard?

This fork:
- ❌ Changes the distributed training setup (single GPU instead of 8x GPUs)
- ❌ Not optimized for absolute speed records
- ✅ **Is for learning and experimenting**
- ✅ **Makes the cutting-edge techniques accessible**

For official records, see the [original repo](https://github.com/KellerJordan/modded-nanogpt)!

## Original Speedrun Records

The original repo holds world records for training speed:
- **Current record**: 2.345 minutes on 8xH100 (down from 45 minutes!)
- **Data efficiency**: 0.73B tokens (down from 10B!)

See [README.old.md](README.old.md) for the complete speedrun history with 41+ records!

## License

Same as the original [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT License).

## Links

- 🏆 **Original Speedrun**: https://github.com/KellerJordan/modded-nanogpt
- 📝 **Muon Optimizer**: https://github.com/KellerJordan/Muon
- 🐦 **Follow the speedrun**: [@kellerjordan0 on X](https://x.com/kellerjordan0)

---
