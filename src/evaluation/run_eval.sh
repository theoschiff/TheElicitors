echo STARTING AT `date`

nvidia-smi

export TOKENIZERS_PARALLELISM=true

export HF_HUB_ENABLE_HF_TRANSFER=1

# export VLLM_WORKER_MULTIPROC_METHOD=spawn

export OMP_NUM_THREADS=16

cd src/evaluation

NUM_GPUS=1

MODELS=(
    google/gemma-3-1b-it
)

GENERATION_SIZES=(
    16384
)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GEN_SIZE="${GENERATION_SIZES[$i]}"
    export GEN_SIZE=$GEN_SIZE

    echo "-----------------------------------------------------"
    echo "MODEL: $MODEL"
    echo "Generation Size: $GEN_SIZE"
    echo "-----------------------------------------------------"

    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$GEN_SIZE,tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$GEN_SIZE,temperature:0.6,top_p:0.95}"
    OUTPUT_DIR=data/evals/$(basename $MODEL)

    for TASK in aime24 aime25 math_500 "gpqa:diamond" U_math; do

        echo "Running evaluation for $TASK with model $MODEL and generation size $GEN_SIZE"
        
        lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
            --custom-tasks evaluate.py \
            --use-chat-template \
            --output-dir $OUTPUT_DIR 
    done
done
