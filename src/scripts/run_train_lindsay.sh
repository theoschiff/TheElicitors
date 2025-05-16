#!/bin/bash
#SBATCH --chdir=/home/bordier/TheElicitors
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=test
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

echo "STARTING AT $(date)"

# Load required system modules (if still needed for CUDA or MPI)
module load gcc cuda openmpi

# >>> Conda environment setup <<<
# Load conda (required if not already in PATH)
source /home/bordier/applications/anaconda3/etc/profile.d/conda.sh

# Activate your environment (replace 'base' if you're using a custom one)
conda activate base

# Environment config
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="/scratch/izar/bordier/.cache"

# Sanity checks
nvcc --version
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# Stage 1: vLLM Serve
export CUDA_VISIBLE_DEVICES=0
trl vllm-serve --model Qwen/Qwen3-1.7B &

# Wait for it to initialize
sleep 120
echo "Starting GRPO training"

# Get in correct src directory
cd
cd TheElicitors/src

# Stage 2: GRPO Training
export CUDA_VISIBLE_DEVICES=1
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes 2 \
    train/rule_based_grpo.py --config receipes/rule_based_grpo.yaml
