#!/bin/bash
#SBATCH --chdir /home/mellouli/TheElicitors
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=test
#SBATCH --time=1:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --output=/home/mellouli/TheElicitors/slurm-%j.out
#SBATCH --error=/home/mellouli/TheElicitors/slurm-%j.err

echo STARTING AT `date`

module load gcc cuda openmpi python

cd ~/TheElicitors
source venv/bin/activate
cd src

nvcc --version
nvidia-smi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME="/scratch/mellouli/.cache"

python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

export CUDA_VISIBLE_DEVICES=0
trl vllm-serve --model google/gemma-3-1b-it &

sleep 120
echo "Starting GRPO training with normalized logprob-based rewards"

export CUDA_VISIBLE_DEVICES=1,2
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes 2 \
    train/train_grpo.py --config recipes/log_based_grpo.yaml