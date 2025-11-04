"""
Quick Validation Training Script

Simplified training script for validation experiments.
Accepts command-line arguments for key parameters.

Usage:
    python train_validation_quick.py \\
        --train_files "data/fineweb10B/fineweb_train_*.bin" \\
        --num_steps 1000 \\
        --run_name "baseline"
"""

import argparse
import sys
import os

# Import from main training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll modify the hyperparameters before importing the training code
def main():
    parser = argparse.ArgumentParser(description="Quick validation training")
    parser.add_argument("--train_files", type=str, required=True,
                        help="Training data pattern")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of training steps")
    parser.add_argument("--run_name", type=str, default="validation",
                        help="Run name for logging")

    args = parser.parse_args()

    print(f"="*60)
    print(f"VALIDATION TRAINING: {args.run_name}")
    print(f"="*60)
    print(f"Train files: {args.train_files}")
    print(f"Num steps: {args.num_steps}")
    print(f"="*60)
    print()

    # Import and modify training script
    import train_gpt_single as tgs

    # Override hyperparameters
    original_train_files = tgs.args.train_files
    original_num_scheduled = tgs.args.num_scheduled_iterations
    original_num_extension = tgs.args.num_extension_iterations
    original_val_loss_every = tgs.args.val_loss_every
    original_run_id = tgs.args.run_id

    tgs.args.train_files = args.train_files
    tgs.args.num_scheduled_iterations = args.num_steps
    tgs.args.num_extension_iterations = 0
    tgs.args.val_loss_every = 100
    tgs.args.save_checkpoint = False
    tgs.args.run_id = args.run_name

    # Update file paths
    data_path = os.environ.get("DATA_PATH", ".")
    tgs.args.train_files = os.path.join(data_path, args.train_files)

    print(f"Configured for validation:")
    print(f"  Train files: {tgs.args.train_files}")
    print(f"  Steps: {tgs.args.num_scheduled_iterations}")
    print(f"  Run ID: {tgs.args.run_id}")
    print()

    # Run the training by executing the main training loop from train_gpt_single
    # We need to re-run the initialization with new parameters

    print("Starting training...")
    print()

    # This is a bit hacky but we'll just exec the main training script
    # with modified globals
    exec(open("train_gpt_single.py").read(), {
        "__name__": "__main__",
        "__file__": "train_gpt_single.py"
    })

if __name__ == "__main__":
    main()
