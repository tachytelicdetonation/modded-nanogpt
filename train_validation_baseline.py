#!/usr/bin/env python3
"""
Baseline training wrapper for validation experiments.

This script runs the main training code with modified hyperparameters
for baseline validation on raw data.
"""

import sys
import os

# Read the main training file
with open('train_gpt_single.py', 'r') as f:
    code = f.read()

# Find and replace the Hyperparameters section
# We'll inject our modifications before the class is instantiated

# Insert our config modifications right after the Hyperparameters class definition
config_override = '''
# VALIDATION CONFIG OVERRIDE - Baseline
args.train_files = "data/fineweb10B/fineweb_train_*.bin"
args.num_scheduled_iterations = 1000
args.num_extension_iterations = 0
args.val_loss_every = 100
args.save_checkpoint = False
args.run_id = "baseline_raw"

# Update file path with DATA_PATH
data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)

print(f"\\n{'='*60}")
print(f"VALIDATION TRAINING: BASELINE (Raw Data)")
print(f"{'='*60}")
print(f"Train files: {args.train_files}")
print(f"Num steps: {args.num_scheduled_iterations}")
print(f"Run ID: {args.run_id}")
print(f"{'='*60}\\n")
'''

# Find where args is instantiated
import_marker = "args = Hyperparameters()"
if import_marker in code:
    # Insert our override right after args instantiation
    code = code.replace(
        import_marker,
        import_marker + "\n\n" + config_override
    )
else:
    print("ERROR: Could not find 'args = Hyperparameters()' in train_gpt_single.py")
    sys.exit(1)

# Execute the modified code
exec(code)
