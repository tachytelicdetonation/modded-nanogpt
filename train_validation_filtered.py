#!/usr/bin/env python3
"""
Filtered training wrapper for validation experiments.

This script runs the main training code with modified hyperparameters
for validation on filtered data (80%).
"""

import sys
import os
import tempfile
import subprocess

# Read the main training file
with open('train_gpt_single.py', 'r') as f:
    code = f.read()

# Get output directory from environment or use default
FILTERED_DATA_DIR = os.environ.get('FILTERED_DATA_DIR', 'data/fineweb10B_filtered_80pct')

# Insert our config modifications right after the Hyperparameters class definition
config_override = f'''
# VALIDATION CONFIG OVERRIDE - Filtered
args.train_files = "{FILTERED_DATA_DIR}/fineweb_train_*.bin"
args.num_scheduled_iterations = 800  # 20% less than baseline
args.num_extension_iterations = 0
args.val_loss_every = 100
args.save_checkpoint = False
args.run_id = "filtered_80pct"

# Update file path with DATA_PATH
data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)

print(f"\\n{{'='*60}}")
print(f"VALIDATION TRAINING: FILTERED (80% Data)")
print(f"{{'='*60}}")
print(f"Train files: {{args.train_files}}")
print(f"Num steps: {{args.num_scheduled_iterations}}")
print(f"Run ID: {{args.run_id}}")
print(f"{{'='*60}}\\n")
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

# Write to a temporary file in the current directory (needed for Triton JIT)
temp_file = '.train_filtered_temp.py'
try:
    with open(temp_file, 'w') as f:
        f.write(code)

    # Run the temporary file
    result = subprocess.run([sys.executable, temp_file], env=os.environ.copy())
    sys.exit(result.returncode)
finally:
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
