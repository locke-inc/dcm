"""
kaggle_notebook.py — Copy-paste this into a Kaggle notebook cell
================================================================
Prerequisites:
    - GPU T4 x2 accelerator selected in Kaggle Settings
    - Internet enabled (for first run — downloads Qwen2.5-7B)

This script:
    1. Installs dependencies
    2. Clones the DCM repo from GitHub
    3. Runs the sanity check (encoder + diffuser + full pipeline)
    4. Optionally kicks off a training run
"""

# ============================================================
# CELL 1: Install dependencies + clone repo
# ============================================================
# fmt: off
"""
!pip install -q torch transformers accelerate peft bitsandbytes

# Clone the repo (replace with your GitHub URL after pushing)
!git clone https://github.com/YOUR_USERNAME/dcm.git /kaggle/working/dcm

# Or if already cloned, pull latest
# !cd /kaggle/working/dcm && git pull

import sys
sys.path.insert(0, "/kaggle/working/dcm")
"""
# fmt: on

# ============================================================
# CELL 2: Sanity check — encoder + diffuser (no download needed)
# ============================================================
"""
%cd /kaggle/working/dcm
!python sanity_check.py --skip_qwen
"""

# ============================================================
# CELL 3: Full sanity check with Qwen (downloads ~15GB first time)
# ============================================================
"""
!python sanity_check.py
"""

# ============================================================
# CELL 4: Training — synthetic data (quick test)
# ============================================================
"""
!python kaggle_train.py --use_synthetic --max_steps 50 --log_every 5
"""

# ============================================================
# CELL 5: Training — real data with dual T4 DDP
# ============================================================
"""
# First, add a text dataset to your notebook (e.g., Project Gutenberg)
# It will mount at /kaggle/input/<dataset-name>/

# Write accelerate config for dual T4
import os
os.makedirs(os.path.expanduser("~/.cache/huggingface/accelerate"), exist_ok=True)

with open(os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml"), "w") as f:
    f.write('''compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: fp16
num_machines: 1
num_processes: 2
''')

!accelerate launch /kaggle/working/dcm/kaggle_train.py \\
    --data_dir /kaggle/input/YOUR_DATASET/ \\
    --batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --context_len 1024 \\
    --continuation_len 512 \\
    --max_steps 5000 \\
    --save_every 500 \\
    --output_dir /kaggle/working/dcm_checkpoints
"""

# ============================================================
# INLINE VERSION — Run everything in one cell without cloning
# ============================================================
# If you prefer not to use git, just paste this single cell:

def run_inline():
    """Run sanity check directly — useful if you uploaded files manually."""
    import sys, os
    os.chdir("/kaggle/working/dcm")
    sys.path.insert(0, "/kaggle/working/dcm")

    # Quick test (no Qwen download)
    from sanity_check import test_ssm_encoder, test_diffuser, test_full_pipeline_no_qwen
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = test_ssm_encoder(device)
    test_diffuser(device, cfg)
    test_full_pipeline_no_qwen(device)
    print("\nEncoder + Diffuser OK. Run `!python sanity_check.py` for full test with Qwen.")
