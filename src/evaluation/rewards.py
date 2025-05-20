import os
import random
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
import torch
import re
from sentence_transformers import util

#File containing the same reward functions as in the original train code but slightly modified in order to allow easier running of the evaluation script with the lighteval framework

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

CMU = cmudict.dict()        # keep it global so we load only once
VOWEL_RE = re.compile(r"[aeiouy]+")

def count_syllables(word: str) -> int:
    """Best-effort syllable count with CMU fallback."""
    word = word.lower()
    if word in CMU:
        # a word can have multiple pronunciations; take the max syllable count
        return max(len([p for p in pron if p[-1].isdigit()]) for pron in CMU[word])
    return len(VOWEL_RE.findall(word))

def total_syllables(line: str) -> int:
    return sum(count_syllables(w) for w in re.findall(r"\b\w+\b", line.lower()))

def rhymes(a, b):
    """Simple rhyme test: compare last stressed vowel onward (using cmudict)"""
    def rhyme_key(word):
        phones = CMU.get(word.lower(), [])
        if not phones: return None
        # take the last vowel+stress onward
        for pron in phones:
            for i in range(len(pron)-1, -1, -1):
                if pron[i][-1].isdigit():
                    return tuple(pron[i:])
        return None
    ka, kb = rhyme_key(a), rhyme_key(b)
    return ka is not None and ka == kb

def extract_answer_text(generated_text):
    match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
    return match.group(1).strip() if match else generated_text

def classify_form(poem):
    """Classify a poem's form based on line count and syllable patterns."""
    lines = [line.strip() for line in poem.strip().splitlines() if line.strip()]
    line_count = len(lines)
    syllable_counts = [total_syllables(line) for line in lines]

    # Haiku: 3 lines with 5-7-5 syllable pattern
    if line_count == 3 and syllable_counts == [5, 7, 5]:
        return "haiku"

    # Tanka: 5 lines with 5-7-5-7-7 syllable pattern
    if line_count == 5 and syllable_counts == [5, 7, 5, 7, 7]:
        return "tanka"

    # Limerick: 5 lines with approximate syllable counts
    if line_count == 5 and all(8 <= s <= 9 for s in syllable_counts[:2]) and all(5 <= s <= 6 for s in syllable_counts[2:4]) and 8 <= syllable_counts[4] <= 9:
        return "limerick"

    # Sonnet: 14 lines
    if line_count == 14:
        return "sonnet"

    # Quatrain: 4 lines with similar syllable counts
    if line_count == 4 and max(syllable_counts) - min(syllable_counts) <= 2:
        return "quatrain"

    # Cinquain: 5 lines with specific syllable counts
    if line_count == 5 and syllable_counts == [2, 4, 6, 8, 2]:
        return "cinquain"

    # Octave: 8 lines with similar syllable counts
    if line_count == 8 and max(syllable_counts) - min(syllable_counts) <= 2:
        return "octave"

    # Decastich: 10 lines with similar syllable counts
    if line_count == 10 and max(syllable_counts) - min(syllable_counts) <= 2:
        return "decastich"

    # Sestet: 6 lines with similar syllable counts
    if line_count == 6 and max(syllable_counts) - min(syllable_counts) <= 2:
        return "sestet"

    # Couplet: 2 lines with similar syllable counts
    if line_count == 2 and abs(syllable_counts[0] - syllable_counts[1]) <= 2:
        return "couplet"

    return "free_verse"

def rhyme_key(word: str):
    """Return CMU rhyme key (last stressed vowel → end) or None."""
    phones = CMU.get(word.lower())
    if not phones:
        return None
    for pron in phones:                         # try each pronunciation
        for i in range(len(pron) - 1, -1, -1):
            if pron[i][-1].isdigit():           # first stressed vowel from the end
                return tuple(pron[i:])
    return None

def rhyme_accuracy(completions, targets, **kwargs) -> float:
    """
    A line is 'correct' if its end-word rhymes with the end-word of the
    corresponding reference line (same CMU rhyme key).
    """
    
    rewards = []
    
    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
    
    for i, (completion, target) in enumerate(zip(completions, targets)):
        try:
            completion = "<think>" + completion
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
                continue

            pred = match.group(2).strip()
            gold = target.strip()
            g_lines = [l for l in pred.strip().splitlines() if l.strip()]
            r_lines = [l for l in gold.strip().splitlines() if l.strip()]
            n = min(len(g_lines), len(r_lines))
            if n == 0:
                rewards.append(0.0)
                continue

            hits = 0
            checked = 0
            for g, r in zip(g_lines[:n], r_lines[:n]):
                g_last = re.findall(r"\b\w+\b", g.lower())[-1]
                r_last = re.findall(r"\b\w+\b", r.lower())[-1]
                kg, kr = rhyme_key(g_last), rhyme_key(r_last)
                if kg and kr:                # only score when we have information
                    hits += int(kg == kr)
                    checked += 1
            rewards.append(hits / checked if checked else 0.0)

        except Exception as e:
            print(f"[sentence_reward] Error on pair {i}: {e}")
            rewards.append(0.0)
            
    return rewards


    

def syllable_accuracy(completions, targets, **kwargs) -> float:
    """
    Per-line accuracy = 1 - |Δ syllables| / ref_syllables  (floored at 0).
    Overall score is the mean across comparable lines.
    """
    
    rewards = []
    
    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
    
    for i, (completion, target) in enumerate(zip(completions, targets)):
        try:
            completion = "<think>" + completion
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
                continue

            pred = match.group(2).strip()
            gold = target.strip()
            g_lines = [l for l in pred.strip().splitlines() if l.strip()]
            r_lines = [l for l in gold.strip().splitlines() if l.strip()]
            n = min(len(g_lines), len(r_lines))
            if n == 0:
                rewards.append(0.0)
                continue
            
            scores = []
            for g, r in zip(g_lines[:n], r_lines[:n]):
                rs = total_syllables(r)
                if rs == 0:
                    continue
                diff = abs(total_syllables(g) - rs)
                scores.append(max(0.0, 1.0 - diff / rs))
                
            rewards.append(sum(scores) / len(scores) if scores else 0.0)

        except Exception as e:
            print(f"[sentence_reward] Error on pair {i}: {e}")
            rewards.append(0.0)
            
    return rewards

def reward_poem_form(completions, targets, **kwargs):
    """
    Binary reward: 1 if the *combined* poem (reference beginning + generation)
    has the same detected form as the gold full poem, else 0.
    """
    rewards = []
    
    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
    
    for i, (completion, target) in enumerate(zip(completions, targets)):
        try:
            completion = "<think>" + completion
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
                continue

            pred = match.group(2).strip()
            gold = target.strip()
            gold_form   = classify_form(pred)
            test_form   = classify_form(gold)
            
            rewards.append(int(gold_form == test_form))

        except Exception as e:
            print(f"[sentence_reward] Error on pair {i}: {e}")
            rewards.append(0.0)
            
    return rewards


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

