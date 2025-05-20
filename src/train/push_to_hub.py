from transformers import AutoModel
import os

# Run with : 
# export HF_TOKEN="your_key"
# Sinteract -c 4 -t 0:20:0 -m 32G -p gpu -g gpu:1
# export HF_HUB_ENABLE_HF_TRANSFER=1
# python src/train/push_to_hub.py


local_path = "/scratch/izar/bordier/Qwen/Qwen3-1.7B/Qwen-math-normalization-token_level"

model = AutoModel.from_pretrained(local_path, from_tf=False)

model.push_to_hub("lindsaybordier/Qwen-math-normalization-token_level", private=False, token=os.getenv("HF_TOKEN"))
