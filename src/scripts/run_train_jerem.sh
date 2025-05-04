#!/bin/bash
#SBATCH --chdir /home/barghorn/TheElicitors
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --partition h100
#SBATCH --time=10:0:0
#SBATCH --account sma-llm-botafogo
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 128G

echo STARTING AT `date`

module load gcc cuda openmpi python

cd ..
cd MasterProject/
source venvs/train/bin/activate
cd ..
cd TheElicitors/src

nvcc --version

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME="/scratch/barghorn/.cache"

accelerate launch --num_processes 2 --config_file configs/deepspeed_zero3.yaml train/rule_based_grpo.py --config reciepes/rule_based_grpo.yaml
