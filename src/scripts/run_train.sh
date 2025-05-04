#!/bin/bash
#SBATCH --chdir /home/barghorn/TheElicitors
#SBATCH --ntasks-per-node=1  
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --partition h100
#SBATCH --time=10:0:0
#SBATCH --qos=long
#SBATCH --account master
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 256G

echo STARTING AT `date`

module load gcc cuda  openmpi python hdf5/1.14.3-mpi  py-scipy

cd ..
source ??
cd ??

nvcc --version

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=4
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODES=("${NODELIST[@]}")


echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "TRAIN_NODES=$TRAIN_NODES"

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME="/scratch/barghorn/.cache"

srun accelerate launch --num_processes 3 --config_file configs/accelerate_configs/deepspeed_zero3.yaml scripts/run_r1_grpo.py --config receipes/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
