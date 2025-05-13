#!/bin/bash
#SBATCH --chdir /home/mellouli/TheElicitors
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:2  # Request 2 GPUs
#SBATCH --partition=test  # Ensure this is a GPU-enabled partition
#SBATCH --time=4:0:0  # Set appropriate time limit
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G  # Adjust memory as needed
#SBATCH --output=/home/mellouli/TheElicitors/slurm-%j.out
#SBATCH --error=/home/mellouli/TheElicitors/slurm-%j.err

echo "Job started at $(date)"

# Set Hugging Face token
#export HF_token="####"

module load gcc cuda openmpi python

source /home/mellouli/TheElicitors/venv/bin/activate

cd /home/mellouli/TheElicitors

nvcc --version
nvidia-smi

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="/scratch/izar/mellouli/.cache"

# Verify PyTorch and CUDA versions
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# Start the vLLM server
export CUDA_VISIBLE_DEVICES=0
trl vllm-serve --model google/gemma-3-1b-it &

# Wait for the server to initialize
sleep 120

echo "Starting GRPO training with rule-based rewards using z-score normalization"

# Set CUDA devices for training
export CUDA_VISIBLE_DEVICES=1

# Run the training script with Accelerate and DeepSpeed
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file /home/mellouli/TheElicitors/src/configs/deepspeed_zero3.yaml --num_processes 1 \
    /home/mellouli/TheElicitors/src/train/rule_based_grpo.py --config /home/mellouli/TheElicitors/src/receipes/rule_based_grpo.yaml

echo "Job finished at $(date)"

