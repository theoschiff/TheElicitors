#!/bin/bash
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=10:0:0
#SBATCH --cpus-per-task=32
#SBATCH --mem 64G
#SBTACH --account master

#make sure that you are running from the root directory

echo STARTING AT `date`

module load gcc cuda openmpi python

# Activate virtualenv
#source ~/home/dechilla/TheElicitors/venvs/train/bin/activate
source /venvs/train/bin/activate
cd src

nvcc --version

nvidia-smi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="/scratch/dechilla/.cache"

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

