import os
import random
import re
import torch
import nltk
nltk.download('cmudict')
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import cmudict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from functools import partial
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



 
def normalize_rewards(rewards, normalization_type="none", token_counts=None):
    """
    Apply normalization to a list of rewards.
    
    Args:
        rewards (list[float]): List of reward values
        normalization_type (str): Type of normalization to apply ('none', 'token-level', 'z-score', 'min-max')
        token_counts (list[int], optional): List of token counts (required for token-level normalization)
    
    Returns:
        list[float]: Normalized rewards
    """
    if normalization_type == "none" or not rewards:
        return rewards
    
    # Make a copy of rewards to avoid modifying the original
    normalized_rewards = rewards.copy()
    
    if normalization_type == "token-level":
        if token_counts is None:
            print("Warning: Token counts are required for token-level normalization but not provided.")
            return rewards
        normalized_rewards = [r / max(1, c) for r, c in zip(normalized_rewards, token_counts)]
    
    elif normalization_type == "z-score":
        mean_reward = sum(normalized_rewards) / len(normalized_rewards)
        variance = sum((r - mean_reward) ** 2 for r in normalized_rewards) / len(normalized_rewards)
        std_reward = variance ** 0.5
        if std_reward > 0:  
            normalized_rewards = [(r - mean_reward) / std_reward for r in normalized_rewards]
    
    elif normalization_type == "min-max":
        min_reward = min(normalized_rewards)
        max_reward = max(normalized_rewards)
        if max_reward > min_reward:  
            normalized_rewards = [(r - min_reward) / (max_reward - min_reward) for r in normalized_rewards]
    
    return normalized_rewards


def format_reward_func(completions, target=None, poem_end=None, normalization="none", **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        normalization (str): Type of normalization to apply ('none', 'token-level', 'z-score', 'min-max')
      
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
    
    # Apply normalization (except token-level which requires token counts)
    if normalization != "none" and normalization != "token-level":
        rewards = normalize_rewards(rewards, normalization)
    
    return rewards

def equation_reward_func(completions, target, nums, normalization="none", **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
        normalization (str): Type of normalization to apply ('none', 'token-level', 'z-score', 'min-max')
    
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
    # Apply normalization (except token-level which requires token counts)
    if normalization != "none" and normalization != "token-level":
        rewards = normalize_rewards(rewards, normalization)
    
    return rewards

def make_gold_answer_logprob_reward(
    model,
    tokenizer,
    batch_size: int = 8,
    normalization: str = "none",
):
    """
    Returns a reward function that takes:
      - prompts: List[str]
      - completions: List[str]    (with <think>…</think><answer>…</answer>)
      - gold_answers: List[str]
    and returns List[float] of log-probs for target under the current policy
    served by your vLLM HTTP server.
    
    Args:
        api_base: URL for the vLLM server
        model_name: Name of the model to use
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for processing
        normalization: Type of normalization to apply ('none', 'token-level', 'z-score', 'min-max')
    
    Returns:
        function: Reward function
    """
    # url = f"{api_base}/generate"
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def reward_fn(
        prompts: List[str],
        completions: List[str],
        gold_answers: List[str] = None,
        poem_end: List[str] = None,
        **kwargs,  # any extra dataset fields are ignored
    ) -> List[float]:
        
        rewards = []
        token_counts = []
        full_logprobs = []
        
        model.eval()
        
        gold_labels = gold_answers if gold_answers is not None else poem_end
        
        batch_size = 2

        # Don't append to the list you're iterating over
        batched_prompts_ids = []
        batched_completions_ids = []
        batched_ids = []

        for prompt, completion, gold_answer in zip(prompts, completions, gold_labels):
            m = re.search(r"<think>([\s\S]*?)</think>", completion)
            reasoning = m.group(1) if m else ""
            
            full_prompt = prompt + reasoning + "</think>\n"
            full_completion = f"<answer>{gold_answer}</answer>"

            # Tokenize to get input_ids (not BatchEncoding objects)
            prompt_ids = tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0).to(device)
            completion_ids = tokenizer(full_completion, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0).to(device)

            # Store separately if needed
            batched_prompts_ids.append(prompt_ids)
            batched_completions_ids.append(completion_ids)

            # Concatenate for full input
            full_input = torch.cat([prompt_ids, completion_ids], dim=0)
            batched_ids.append(full_input)
        
        batched_ids_padded = pad_sequence(batched_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

        # Create attention masks (1 for actual tokens, 0 for padding)
        attention_masks = (batched_ids_padded != tokenizer.pad_token_id).long()

        # Move to device
        batched_ids_padded = batched_ids_padded.to(device)
        attention_masks = attention_masks.to(device)
        
        
        # Batched inference with batch size 8
        logits_list = []
        with torch.no_grad():
            for i in range(0, batched_ids_padded.size(0), batch_size):
                input_ids_batch = batched_ids_padded[i:i+batch_size]
                attention_mask_batch = attention_masks[i:i+batch_size]
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                logits_list.append(outputs.logits)
        logits = torch.cat(logits_list, dim=0)
            
        for batch_idx in range(len(batched_ids)):
            prompt_len = batched_prompts_ids[batch_idx].size(0)
            completion_ids = batched_completions_ids[batch_idx]
            example_logprobs = []

            for i, token_id in enumerate(completion_ids):
                pos = prompt_len + i - 1 
                if pos >= logits.size(1):
                    print(f"Warning: Skipping token at pos {pos} (out of bounds) for batch {batch_idx}")
                    continue

                logits_for_token = logits[batch_idx, pos]  # use correct batch index
                log_probs = F.log_softmax(logits_for_token, dim=-1)
                logprob = log_probs[token_id].item()
                example_logprobs.append(logprob)

            full_logprobs.append(np.abs(sum(example_logprobs))) # For min max scaling we need positive values
            
        min_lp = np.min(full_logprobs)
        max_lp = np.max(full_logprobs)
        
        if min_lp == max_lp:
            normalized = np.ones_like(full_logprobs)
        else:
            normalized = (full_logprobs - min_lp) / (max_lp - min_lp)

        print("Full logprobs for all examples:", full_logprobs)
        print("Normalized logprobs:", normalized)
        print("Number of examples:", len(full_logprobs))
        
        del batched_ids_padded
        del attention_masks
        del logits
        del logits_list
        torch.cuda.empty_cache()
        
        return normalized
    
    return reward_fn




# -- Poetry-specific functions --

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

def rhyme_accuracy(completions, target=None,  poem_end=None, normalization="none", **kwargs) -> float:
    """
    A line is 'correct' if its end-word rhymes with the end-word of the
    corresponding reference line (same CMU rhyme key).
    """
    
    rewards = []
    
    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
    
    real_target = target if target is not None else poem_end
    
    for i, (completion, target) in enumerate(zip(completions, real_target)):
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
            
    if normalization != "none" and normalization != "token-level":
        rewards = normalize_rewards(rewards, normalization)
            
    return rewards


    

def syllable_accuracy(completions, target=None,  poem_end=None, normalization="none", **kwargs) -> float:
    """
    Per-line accuracy = 1 - |Δ syllables| / ref_syllables  (floored at 0).
    Overall score is the mean across comparable lines.
    """
    
    rewards = []
    
    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
    
    real_target = target if target is not None else poem_end
    
    for i, (completion, target) in enumerate(zip(completions, real_target)):
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

def reward_poem_form(completions, target=None, poem_end=None, normalization="none", **kwargs):
    """
    Binary reward: 1 if the *combined* poem (reference beginning + generation)
    has the same detected form as the gold full poem, else 0.
    """
    rewards = []
    
    regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
    
    real_target = target if target is not None else poem_end
    
    for i, (completion, target) in enumerate(zip(completions, real_target)):
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
            
    if normalization != "none" and normalization != "token-level":
        rewards = normalize_rewards(rewards, normalization)
            
    return rewards


def sentence_similarity_reward_func(completions, target = None, poem_end=None, sentence_model=None, normalization="none", **kwargs):
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
    
    real_target = target if target is not None else poem_end

    for i, (completion, target) in enumerate(zip(completions, real_target)):
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
            
    if normalization != "none" and normalization != "token-level":
        rewards = normalize_rewards(rewards, normalization)

    return rewards


def global_poetry_reward_func(
        completions: List[str],
        poem_end:   List[str],      # the gold continuation
        ref_start:  List[str],      # pass the given first-third here
    ) -> List[float]:

    rewards = []

    # Weights : tweak as you wish, must sum to 1
    q, r, s, t = 0.30, 0.25, 0.25, 0.20   # sim, form, rhymes, syllables

    for completion, gold_answer, gold_begin in zip(completions, poem_end, ref_start):
        gen = extract_answer_text(completion)

        # 1) semantic similarity on the continuation only
        similarity = sentence_similarity_reward_func(gen, gold_answer)

        # 2) binary structural form (full poem)
        form_ok = reward_poem_form(gold_begin + gold_answer,
                                   gold_begin + gen)

        # 3) rhyme & syllable continuums
        rhyme_score    = rhyme_accuracy(gen, gold_answer)
        syllable_score = syllable_accuracy(gen, gold_answer)

        final_reward = (
              q * similarity
            + r * form_ok
            + s * rhyme_score
            + t * syllable_score
        )
        rewards.append(final_reward)

    return rewards

# -- end of poetry-specific functions --



def main_rewards():
    poem_start_1 = """Young ardent soul, graced with fair Nature's truth,
    Spring warmth of heart, and fervency of mind,
    And still a large late love of all thy kind.
    Spite of the world's cold practice and Time's tooth,"""

    poem_end_1 = """For all these gifts, I know not, in fair sooth,
    Whether to give thee joy, or bid thee blind
    Thine eyes with tears, - that thou hast not resign'd
    The passionate fire and freshness of thy youth:
    For as the current of thy life shall flow,
    Gilded by shine of sun or shadow-stain'd,
    Through flow'ry valley or unwholesome fen,
    Thrice blessed in thy joy, or in thy woe
    Thrice cursed of thy race, - thou art ordain'd
    To share beyond the lot of common men."""

    form_1= "sonnet"

    poem_start_2 = """Now while my lips are living
    Their words must stay unsaid,"""

    poem_end_2 = """And will my soul remember
    To speak when I am dead?
    Yet if my soul remembered
    You would not heed it, dear,
    For now you must not listen,
    And then you could not hear."""

    form_2 = "octave"
    # generate a poem with gemma 1b 
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).eval()

    prompt_1 = f"Write a poem in the form of {form_1} with the following start:\n{poem_start_1}\n"
    prompt_1 += "<think>"
    input_ids_1 = tokenizer(prompt_1, return_tensors="pt").input_ids

    with torch.no_grad():
        gen_output = model.generate(
            input_ids_1,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9
        )
    completion_1 = tokenizer.decode(gen_output[0], skip_special_tokens=True)

    print("=== Generated Poem ===")
    print(completion_1, "\n")
    print("Detected form: ", classify_form(completion_1))

    prompt_2 = f"Write a poem in the form of {form_2} with the following start:\n{poem_start_2}\n"
    prompt_2 += "<think>"
    input_ids_2 = tokenizer(prompt_2, return_tensors="pt").input_ids
    with torch.no_grad():
        gen_output = model.generate(
            input_ids_2,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9
        )
    completion_2 = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print("=== Generated Poem 2===")
    print(completion_2, "\n")
    print("Detected form: ", classify_form(completion_2))
    print("=== Reference Poem ===")
    print(poem_start_1 + poem_end_1, "\n")
    print("Detected form: ", classify_form(poem_start_1 + poem_end_1))


    # Run evaluations
    rewards = global_poetry_reward_func(
        completions=[completion_1, completion_2],
        poem_end=[poem_end_1, poem_end_2],
        ref_start=[poem_start_1, poem_start_2]
    )

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("=== Rewards ===")
    for i, reward in enumerate(rewards):
        print(f"Poem {i+1} Reward: {reward:.4f}")
        print(f"Poem {i+1} Form: {classify_form(completion_1)}")
        print(f"Poem {i+1} Rhyme Accuracy: {rhyme_accuracy(completion_1, poem_end_1):.4f}")
        print(f"Poem {i+1} Syllable Accuracy: {syllable_accuracy(completion_1, poem_end_1):.4f}")
        print(f"Poem {i+1} Embedding Similarity: {embedding_similarity(completion_1, poem_end_1, embed_model):.4f}")
        print(f"Poem {i+1} Syllable Count: {total_syllables(completion_1)}")

if __name__ == "__main__":
    main_rewards()