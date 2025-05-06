#!/bin/bash
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=10:0:0
#SBATCH --cpus-per-task=32
#SBATCH --mem 64G

echo STARTING AT `date`

module load gcc cuda openmpi python

# Get absolute path to the repo root regardless of where the script is run
REPO_ROOT=$(realpath "$(dirname "$0")/..")
SRC_DIR="$REPO_ROOT/src"
VENV_PATH="$REPO_ROOT/venvs/train"

# Activate virtualenv
source "$VENV_PATH/bin/activate"

# Move to source directory
cd "$SRC_DIR"

nvcc --version

nvidia-smi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="/scratch/barghorn/.cache"

# Verify Torch + CUDA versions
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

export CUDA_VISIBLE_DEVICES=0
trl vllm-serve --model google/gemma-3-1b-it &

sleep 120
echo "Starting GRPO training"


# Training using the second GPU
export CUDA_VISIBLE_DEVICES=1
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes 1 \
    train/train_grpo.py --config reciepes/rule_based_grpo.yaml

