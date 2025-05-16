def generate_r1_math_prompt(tokenizer, numbers, target):
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
      

def generate_r1_poetry_prompt(tokenizer, author, title, poem_start):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    },
    { 
        "role": "user",
        "content": f"Using the author name : {author}, the title : {title} and the start of an existing poem that is as follows : {poem_start}, create a new ending for this poem. You need to take into account the form, the rhymes, the syllables and theme to create a new realistic ending. Show your work in <think> </think> where you are allowed to thinka bout structure, rhymes, etc. And return the final poem ending in <answer> </answer> tags. Think step by step inside <think> tags."
    },
    {
        "role": "assistant",
        "content": "Let me write a poem step by step.\n<think>"
    }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "author": author, "title": title, "poem_start": poem_start}