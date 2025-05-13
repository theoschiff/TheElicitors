import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from rewards import format_reward_func, equation_reward_func, sentence_similarity_reward_func
from sentence_transformers import SentenceTransformer, util
from functools import partial
from data_utils import generate_r1_math_prompt, generate_r1_poetry_prompt

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    task_type : str = "math"

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


def grpo_function(
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

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # select a random subset of 50k samples
    if script_args.task_type == "math":
        dataset = dataset.shuffle(seed=42).select(range(50000))
    elif script_args.task_type == "poetry":
        dataset = dataset.shuffle(seed=42)

    #####################
    # Prepare and format dataset
    #####################
    if script_args.task_type == "math":
        dataset = dataset.map(lambda x: generate_r1_math_prompt(tokenizer, x["nums"], x["target"]))
    elif script_args.task_type == "poetry":
        dataset = dataset.map(lambda x: generate_r1_poetry_prompt(x["author"], x["title"], x["poem_start"]))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset sample: {dataset[0]}")
    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Sentence_model on device: {sentence_model.device}")

    # Bind the model into the reward function
    
    if script_args.task_type == "math":
        reward_functions = [format_reward_func, equation_reward_func]
    elif script_args.task_type == "poetry":
        similarity_reward = partial(sentence_similarity_reward_func, sentence_model=sentence_model)
        reward_functions = [format_reward_func, similarity_reward]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
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
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Training/evaluation parameters {training_args}")


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    # trainer.log_metrics("train", metrics)
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
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()