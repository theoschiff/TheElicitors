from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
from rewards import equation_reward_func, format_reward_func
 
# Load dataset from Hugging Face Hub
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")
# select a random subset of 50k samples
dataset = dataset.shuffle(seed=42).select(range(50000))

model_id = "google/gemma-3-1b-it"
 
# Load tokenizer from Hugging Face Hub to format the dataset to our "r1" prompt 
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# gemerate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_prompt(numbers, target):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
      },
      { 
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
      },
      {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
      }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}
 
# convert our dataset to the r1 prompt
dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))
 
# split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.1)
 
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]


 
# our model we are going to use as policy 
model_config = ModelConfig(
    model_name_or_path=model_id,
    torch_dtype="bfloat16",
    attn_implementation="eager",
    load_in_4bit=True,
)

# Hyperparameters
training_args = GRPOConfig(
    output_dir="gemma_calculator",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    max_prompt_length=256,
    max_completion_length=1024, # max length of the generated output for our solution
    num_generations=2,
    beta=0.001,
    report_to='tensorboard',
)

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model(training_args.output_dir)
