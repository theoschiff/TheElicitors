#!/bin/bash
#SBATCH --chdir /home/schiffer/MA4/RL_project/TheElicitors
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=10:0:0
# #SBATCH --account master
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 128G

echo STARTING AT `date`

module load gcc cuda  openmpi python

cd ..
cd TheElicitors/src

nvcc --version

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME="/scratch/schifferlearning/.cache"

srun accelerate launch --num_processes 1 --config_file configs/deepspeed_zero3.yaml train/rule_based_grpo.py --config receipes/rule_based_grpo.yaml
