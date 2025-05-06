import re
import os
import random
import openai
import torch
from typing import List
 
def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

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
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

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
        target:  List[str],
        **kwargs,     # any extra dataset fields are ignored
    ) -> List[float]:
        rewards: List[float] = []
        
        # process in batches
        for i in range(0, len(prompts), batch_size):
            slice_end = i + batch_size
            ctxs    = []
            answers = target[i:slice_end]
            
            # build each context = prompt + reasoning + "</think>\n"
            for prompt, completion in zip(prompts[i:slice_end], completions[i:slice_end]):
                m = re.search(r"<think>([\s\S]*?)</think>", completion)
                reasoning = m.group(1) if m else ""
                ctxs.append(prompt + reasoning + "</think>\n") #TODO reuse the same format as generate_r1_prompt in train_grpo.py
            
            # call vLLM in one go
            resp = openai.Completion.create(
                model      = model_name,
                prompt     = ctxs,    # list of strings
                max_tokens = 0,       # no new tokens
                echo       = True,    # return logprobs for all input tokens
                logprobs   = 0,       # full token_logprobs
            )
            
            # extract per-example rewards
            for choice, ans in zip(resp.choices, answers):
                token_logprobs = choice.logprobs.token_logprobs  # List[float]
                
                # how many logprobs correspond to ans?
                ans_ids = tokenizer(ans,
                                    add_special_tokens=False).input_ids
                n = len(ans_ids)
                
                # sum the last n entries
                rewards.append(sum(token_logprobs[-n:]))
        
        return rewards
    
    return reward_fn