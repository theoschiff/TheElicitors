import re
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import cmudict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

 
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

    import re




def count_syllables(word):
    """Count syllables in a word using cmudict or fallback to simple rule"""
    word = word.lower()
    if word in d:
        return max([len([y for y in x if y[-1].isdigit()]) for x in d[word]])
    else:
        # fallback: estimate syllables by vowel groups
        return len(re.findall(r'[aeiouy]+', word.lower()))

def count_total_syllables(text):
    return sum(count_syllables(w) for w in re.findall(r'\b\w+\b', text))


def rhyme_score(poem):
    """Score based on how many line pairs rhyme"""
    lines = [line.strip() for line in poem.strip().split("\n") if line.strip()]
    rhymes = 0
    for i in range(len(lines) - 1):
        w1 = last_word(lines[i])
        w2 = last_word(lines[i + 1])
        if w1 in d and w2 in d:
            if any(p1[-1] == p2[-1] for p1 in d[w1] for p2 in d[w2]):
                rhymes += 1
    return rhymes / max(1, len(lines) - 1)



def embedding_similarity(text, reference):
    """Use embedding model to compute semantic similarity"""
    emb = embed_model.encode([text, reference])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return sim

def creative_writing_poetry_reward(completions, targets, **kwargs):
    """
    Reward function for poetry generation.
    Returns:
        list[float]: scores between 0 and 1
    """

    nltk.download("cmudict")
    
    # Load once for embedding similarity
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    d = cmudict.dict()

    rewards = []

    for completion, reference in zip(completions, targets):
        try:
            # Extract just the poetic part (optional: assume inside <answer>...</answer>)
            match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if match:
                poem = match.group(1).strip()
            else:
                poem = completion.strip()

            rhyme = rhyme_score(poem)
            syllables = count_total_syllables(poem)

            semantic = embedding_similarity(poem, reference)


            # Combine rewards (tune weights as needed)
            total_reward = (
                0.5 * semantic
                0.25 * rhyme
                0.25 * syllables
            )

            rewards.append(total_reward)
        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(0.0)

    return rewards
