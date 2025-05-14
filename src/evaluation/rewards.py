import os
import random
import openai
import torch
from typing import List
import re
from sentence_transformers import util, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial



def format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion in completions:

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)
        
        # Check if the format is correct
        regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def make_gold_answer_logprob_reward(
    api_base: str,
    model_name:    str,
    tokenizer,
    batch_size:    int = 8,
):
    """
    Returns a reward function that takes:
      - prompts: List[str]
      - completions: List[str]    (with <think>…</think><answer>…</answer>)
      - target: List[str]
    and returns List[float] of log-probs for target under the current policy
    served by your vLLM HTTP server.
    """
    # configure OpenAI-compatible client
    openai.api_base = api_base.rstrip("/") + "/v1"
    openai.api_key  = ""
    
    def reward_fn(
        prompts:      List[str],
        completions:  List[str],
        gold_answers:  List[str], #TODO add to dataset
        **kwargs,     # any extra dataset fields are ignored
    ) -> List[float]:
        rewards: List[float] = []
        
        # process in batches
        for i in range(0, len(prompts), batch_size):
            slice_end = i + batch_size
            ctxs    = []
            
            # build each context = prompt + reasoning + "</think>\n"
            for prompt, completion, gold_answer in zip(prompts[i:slice_end], completions[i:slice_end], gold_answers[i:slice_end]):
                m = re.search(r"<think>([\s\S]*?)</think>", completion)
                reasoning = m.group(1) if m else ""
                
                ctxs.append(prompt + reasoning + "</think>\n<answer>" + gold_answer + "</answer>")
                

            # call vLLM in one go
            resp = openai.Completion.create(
                model      = model_name,
                prompt     = ctxs,    # list of strings
                max_tokens = 0,       # no new tokens
                echo       = True,    # return logprobs for all input tokens
                logprobs   = 1,
            )
            
            # extract per-example rewards
            for choice, gold_answer in zip(resp.choices, gold_answers[i:slice_end]):
                token_logprobs = choice.logprobs.token_logprobs # TODO check dimensions
                
                # how many logprobs correspond to the gold_answer?
                ans_ids = tokenizer("<answer>" + gold_answer + "</answer>",
                                    add_special_tokens=False).input_ids
                n = len(ans_ids)
                
                # sum the last n entries
                rewards.append(sum(token_logprobs[-n:]))
        
        return rewards
    
    return reward_fn


# Load the model once globally
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def sentence_similarity_reward_func(completions, targets, sentence_model=None, **kwargs):
    """
    Computes cosine similarity between predicted and target answers using a shared embedding model.
    Assumes format: <think>...</think>\n<answer>...</answer>
    """
    if sentence_model is None:
        raise ValueError("sentence_model must be provided")

    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"

    rewards = []
    to_encode = []

    valid_pairs = []  # Stores indices of valid completions

    for i, (completion, target) in enumerate(zip(completions, targets)):
        try:
            completion = "<think>" + completion
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
                continue

            pred = match.group(2).strip()
            gold = target.strip()
            to_encode.extend([pred, gold])
            valid_pairs.append(i)
            rewards.append(None)  # Placeholder for now

        except Exception as e:
            print(f"[sentence_reward] Error on pair {i}: {e}")
            rewards.append(0.0)

    print(to_encode)
    if to_encode:
        print("to encode")
        embeddings = sentence_model.encode(
            to_encode,
            convert_to_tensor=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        for j, i in enumerate(valid_pairs):
            emb_pred = embeddings[2 * j]
            emb_target = embeddings[2 * j + 1]
            cosine_sim = util.cos_sim(emb_pred, emb_target).item()
            print("Cosine sim:", cosine_sim)
            reward = max(0.0, cosine_sim)
            print("Reward: ", reward)
            rewards[i] = reward

    return rewards


def main_rewards():
    # test on a fixed snippet
    # Reference target text
    test_text = (
        "The moon floats silently above the hills,\n"
        "Casting silver light on sleeping trees."
    )

    print("Fixed Test Snippet ===")
    print(test_text, "\n")

    # Manually constructed completion that matches the regex
    completion = (
        "<think>I want to write about the moon in a peaceful landscape, "
        "using imagery like light, silence, and nature.</think>\n"
        "<answer>The moon was silently floating over the mountain,\n</answer>"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    # Bind the model into the reward function
    similarity_reward = partial(sentence_similarity_reward_func, sentence_model=sentence_model)

    # Compute and print reward
    reward = similarity_reward([completion], [test_text])[0]
    print("\n=== Completion ===\n", completion)
    print("\nReward score:", reward)


if __name__ == "__main__":
    main_rewards()