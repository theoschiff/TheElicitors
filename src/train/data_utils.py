from fractions import Fraction

def generate_r1_prompt(tokenizer, numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
          },
          { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."
          },
          {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
          }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": numbers}

def find_countdown_expression(nums, target):
    """
    Find an arithmetic expression using each number in nums exactly once that evaluates to target.
    Operations allowed: +, -, *, /, with parentheses.
    Uses Fraction for exact arithmetic to avoid floating-point issues.

    Args:
        nums (list of int or float): input numbers
        target (int or float): desired target value

    Returns:
        str: a string representation of the expression that evaluates to target

    Raises:
        ValueError: if no expression can be found
    """
    # Prepare initial list of (value, expression) pairs
    initial = [(Fraction(n), str(n)) for n in nums]
    target_frac = Fraction(target)

    def helper(pairs):
        # If only one value remains, check if it equals the target
        if len(pairs) == 1:
            val, expr = pairs[0]
            if val == target_frac:
                return expr
            return None

        # Try all pairs of numbers
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                a_val, a_expr = pairs[i]
                b_val, b_expr = pairs[j]

                # Build a new list of remaining numbers
                rest = [pairs[k] for k in range(len(pairs)) if k not in (i, j)]

                # Generate all possible operations
                operations = [
                    (a_val + b_val, f"({a_expr}+{b_expr})"),
                    (a_val - b_val, f"({a_expr}-{b_expr})"),
                    (b_val - a_val, f"({b_expr}-{a_expr})"),
                    (a_val * b_val, f"({a_expr}*{b_expr})"),
                ]

                # Division operations, avoid division by zero
                if b_val != 0:
                    operations.append((a_val / b_val, f"({a_expr}/{b_expr})"))
                if a_val != 0:
                    operations.append((b_val / a_val, f"({b_expr}/{a_expr})"))

                # Recurse on each possibility
                for new_val, new_expr in operations:
                    result = helper(rest + [(new_val, new_expr)])
                    if result:
                        val = eval(result, {"__builtins__": None}, {})
                        assert abs(float(val) - float(target)) < 1e-5, f"Invalid expression: {result}"
                        return result
        return None

    expression = helper(initial)
    if expression is None:
        raise ValueError(f"No solution found for nums={nums} target={target}")
    return expression

def add_gold_answer_to_dataset(dataset):
    """
    Load the HF dataset, compute the countdown solution for each example, and return a new Dataset with 'gold_answer'.

    Args:
        dataset : Hugging Face dataset

    Returns:
        datasets.Dataset: with additional 'gold_answer' column
    """
    def solve(example):
        nums = example['nums']
        target = example['target']
        try:
            example['gold_answer'] = find_countdown_expression(nums, target)
        except ValueError:
            example['gold_answer'] = None
        return example

    new_ds = dataset.map(solve)
    return new_ds