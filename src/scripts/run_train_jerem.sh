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

nvidia-smi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME="/scratch/barghorn/.cache"

python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

export CUDA_VISIBLE_DEVICES=0
trl vllm-serve --model google/gemma-3-1b-it &

sleep 120
echo "Starting GRPO training"

export CUDA_VISIBLE_DEVICES=1,2
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes 2 \
    train/rule_based_grpo.py --config reciepes/rule_based_grpo.yaml

