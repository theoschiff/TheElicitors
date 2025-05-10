import re
import os
import random
import torch 
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import cmudict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
# from sentence_transformers import SentenceTransformer

 
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



def count_syllables(word):
    """Count syllables in a word using cmudict or fallback to simple rule"""
    try:
        nltk.data.find("corpora/cmudict")
    except LookupError:
        nltk.download("cmudict")

    word = word.lower()
    d = cmudict.dict()
    
    if word in d:
        return max([len([y for y in x if y[-1].isdigit()]) for x in d[word]])
    else:
        # fallback: estimate syllables by vowel groups
        return len(re.findall(r'[aeiouy]+', word.lower()))

def total_syllables(text):
    """Count total syllables in a piece of text"""
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(count_syllables(w) for w in words)

def rhymes(a, b):
    """Simple rhyme test: compare last stressed vowel onward (using cmudict)"""
    d = cmudict.dict()
    def rhyme_key(word):
        phones = d.get(word.lower(), [])
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
    """Detect sonnet, haiku, limerick—or free verse otherwise."""

    # ensure poem is a string
    if isinstance(poem, list):
        poem = "\n".join(poem)

    # if generated output is given and <answer></answer> token are given, then only keep what's in between
    poem_only = extract_answer_text(poem)

    lines = [line.strip() for line in poem_only.splitlines() if line.strip()]

    syls = [ total_syllables(line) for line in lines]

    print(syls)

    # Sonnet: 14 lines of exactly 10 syllables (discarding lines for now since we give only a part to generate)
    if all(s == 10 for s in syls):
        return "sonnet"

    # Haiku: 3 lines of 5-7-5 (discarding for now since we give only a part to generate)
    if syls == [5, 7, 5]:
        return "haiku"

    # Limerick: 5 lines, approx [8–9,8–9,5–6,5–6,8–9] + AABBA rhyme
    #if len(lines) == 5: (discarding it for now since we give only a part to generate)
    target = [9, 9, 6, 6, 9]
    if all(abs(s - t) <= 1 for s, t in zip(syls, target)) \
        and rhymes(poem_only[0].split()[-1], poem_only[1].split()[-1]) \
        and rhymes(poem_only[0].split()[-1], poem_only[4].split()[-1]) \
        and rhymes(poem_only[2].split()[-1], poem_only[3].split()[-1]):
        return "limerick"

    # Everything else: free verse
    return "free_verse"

def reward_poem_form(ref_poem, gen_poem: str) -> int:
    """
    Returns 1 if the poem matches its detected form (based on rhymes and syllabes):
      - sonnet, haiku, or limerick structural rules (We should def add more later like Tanka, Sijo, Acrostic etc (found on masterclass website))
      - OR free-verse (no stricter constraint)
    Otherwise returns 0.
    """
    # split into non-blank, stripped lines
    ref_form = classify_form(ref_poem)

    # split into non-blank, stripped lines
    gen_form = classify_form(gen_poem)

    # we only reject if it claims a fixed form but fails its pattern
    return 1 if ref_form == gen_form else 0


def embedding_similarity(text, reference):
    """Use embedding model to compute semantic similarity"""
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = embed_model.encode([text, reference])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return sim


def global_poetry_reward_func(
        completions:  List[str],
        target:  List[str], #TODO add to dataset
        **kwargs,     # any extra dataset fields are ignored
    ) -> List[float]:
        rewards: List[float] = []
        
        for completion, gold_answer in zip(completions, target):
            
            extracted_answer = extract_answer_text(completion)
            
            similarity = embedding_similarity(completion, gold_answer)
            
            form = reward_poem_form(gold_answer, extracted_answer)
            
            reward = 0.4 * similarity + 0.6 * form
            
            rewards.append(reward)
        
        return rewards

def main_rewards():
    # test on a fixed snippet
    test_text = (
        "The fields were bleak and sodden. Not a wing\n"
        "Or note enlivened the depressing wood,\n"
        "A soiled and sullen, stubborn snowdrift stood\n"
        "Beside the roadway. Winds came muttering"
    )

    print("Fixed Test Snippet ===")
    print(test_text, "\n")
    print("Detected form: ", classify_form(test_text))

    # generate a poem with gemma 1b 
    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).eval()

    prompt = "Write a short poem about the moon:\n<answer>"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        gen_output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9
        )
    generated = tokenizer.decode(gen_output[0], skip_special_tokens=True)

    print("=== Generated Poem ===")
    print(generated, "\n")
    print("Detected form: ", classify_form(generated))
    print("Reward (0/1):", reward_poem_form(test_text, generated))


if __name__ == "__main__":
    main_rewards()