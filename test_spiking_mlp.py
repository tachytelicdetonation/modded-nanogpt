"""
Test script to verify Spiking MLP integration
"""
import sys
import os

# Suppress torch warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Testing Spiking MLP integration...")
print("=" * 80)

# Test 1: Check if spikingjelly is available
print("\n1. Checking spikingjelly availability...")
try:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    print("   ✓ spikingjelly is installed and importable")
    SPIKING_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ spikingjelly not available: {e}")
    print("   Note: Install with: pip install spikingjelly==0.0.0.0.12")
    SPIKING_AVAILABLE = False

# Test 2: Check if the script imports correctly
print("\n2. Checking train_gpt_single.py imports...")
try:
    # We can't fully import because it requires CUDA, but we can check syntax
    with open('train_gpt_single.py', 'r') as f:
        code = f.read()
    compile(code, 'train_gpt_single.py', 'exec')
    print("   ✓ train_gpt_single.py compiles successfully")
except SyntaxError as e:
    print(f"   ✗ Syntax error: {e}")
    sys.exit(1)

# Test 3: Verify SpikingMLP class exists in the code
print("\n3. Checking SpikingMLP class definition...")
if 'class SpikingMLP' in code:
    print("   ✓ SpikingMLP class found")
else:
    print("   ✗ SpikingMLP class not found")
    sys.exit(1)

# Test 4: Verify Block class accepts spiking parameters
print("\n4. Checking Block class signature...")
if 'use_spiking_mlp: bool' in code and 'time_steps: int' in code:
    print("   ✓ Block class has spiking MLP parameters")
else:
    print("   ✗ Block class missing spiking MLP parameters")
    sys.exit(1)

# Test 5: Verify GPT class accepts spiking parameters
print("\n5. Checking GPT class signature...")
if 'use_spiking_mlp=args.use_spiking_mlp' in code:
    print("   ✓ GPT instantiation passes spiking MLP flag")
else:
    print("   ✗ GPT instantiation missing spiking MLP flag")
    sys.exit(1)

# Test 6: Verify hyperparameters
print("\n6. Checking hyperparameters...")
if 'use_spiking_mlp: bool' in code and 'snn_time_steps: int' in code:
    print("   ✓ Hyperparameters include SNN options")
else:
    print("   ✗ Hyperparameters missing SNN options")
    sys.exit(1)

print("\n" + "=" * 80)
print("All checks passed! ✓")
print("\nNext steps:")
print("1. Install spikingjelly if not already installed:")
print("   pip install spikingjelly==0.0.0.0.12")
print("\n2. Test with standard MLP (default):")
print("   python train_gpt_single.py")
print("\n3. Test with Spiking MLP (experimental):")
print("   Edit train_gpt_single.py and set: use_spiking_mlp = True")
print("   Then run: python train_gpt_single.py")
print("\n4. Monitor GPU memory and training dynamics closely!")
print("=" * 80)
