import logging
import os
from dataclasses import dataclass
from datetime import datetime
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, get_peft_config, ModelConfig, TrlParser
from rewards import format_reward_func, global_poetry_reward_func, make_gold_answer_logprob_reward, sentence_similarity_reward_func, make_gold_answer_logprob_reward
from sentence_transformers import SentenceTransformer
from functools import partial
from data_utils import generate_r1_math_prompt, generate_r1_poetry_prompt
from custom_grpo_trainer import GRPOLogProbTrainer 

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jeremmmyyyyy/Math"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    normalization: str = "none"  # Options: none, token-level, z-score, min-max
    task_type: str = "math"
    use_logprob_reward: bool = True
    vllm_api_base: str = "http://0.0.0.0:8000"
   
########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_log_based_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    
    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    #####################
    # Prepare and format dataset
    #####################

    if script_args.task_type == "math":
        dataset = dataset.map(lambda x: generate_r1_math_prompt(tokenizer, x["nums"], x["target"]))
    elif script_args.task_type == "poetry":
        dataset = dataset.map(lambda x: generate_r1_poetry_prompt(tokenizer, x["author"], x["title"], x["poem_start"], x["form"]))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset sample: {dataset[0]}")

    # Split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    logger.info(f"Train example: {train_dataset[0]}")

    #########################
    # Setup rewards
    #########################
    logger.info(f"Using normalization method: {script_args.normalization}")
    
    # Create reward functions with the normalization parameter
    format_reward_with_norm = partial(format_reward_func, normalization=script_args.normalization)

    # Add __name__ attributes to the partial functions
    format_reward_with_norm.__name__ = "format_reward_func"

    reward_functions = []
    
    if script_args.use_logprob_reward:
        gold_logprob_reward = make_gold_answer_logprob_reward(
            model    = model,
            # api_base   = script_args.vllm_api_base,
            tokenizer  = tokenizer,
            batch_size    = training_args.per_device_train_batch_size, # 8 # tune for your GPU / throughput
            normalization = script_args.normalization
        )  
        
        reward_functions.append(gold_logprob_reward)

    if script_args.task_type == "math":
        reward_functions.append(format_reward_with_norm)
        training_args.reward_weights = [0.9, 0.1]
    elif script_args.task_type == "poetry":
        reward_functions.append(format_reward_with_norm)
        training_args.reward_weights = [0.9, 0.1]

    #########################
    # Instantiate GRPO trainer
    #########################
    trainer = GRPOLogProbTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )
    
    #########################
    # Log parameters
    #########################
    if trainer.accelerator.is_main_process:
        logger.info(f"Model parameters: {model_args}")
        logger.info(f"Training/evaluation parameters: {training_args}")
        logger.info(f"Using normalization method: {script_args.normalization}")

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs ***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # Wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "logprob"]})
    # Push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_log_based_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()