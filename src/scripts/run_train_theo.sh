#!/bin/bash
#SBATCH --chdir /home/schiffer/MA4/RL_project/TheElicitors
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=10:0:0
# #SBATCH --account master
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 128G

echo STARTING AT `date`

module load gcc cuda openmpi


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RL_env

echo "Conda environment after activation: $CONDA_DEFAULT_ENV"

echo "Python version: $(python --version)"

nvcc --version

echo "showing nvidia-smi"
nvidia-smi

echo "Cleaning torch extensions cache"
rm -rf ~/.cache/torch_extensions

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

export MP_START_METHOD=spawn


export HF_HOME="/scratch/izar/schiffer/.cache"


export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

trl vllm-serve --model Qwen/Qwen3-1.7B &


sleep 120
echo "Starting GRPO training"

export CUDA_VISIBLE_DEVICES=1
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file src/configs/deepspeed_zero3.yaml --num_processes 1 \
    src/train/rule_based_grpo.py --config src/receipes/rule_based_grpo.yaml

