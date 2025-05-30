"""Custom evaluation tasks for LightEval."""

from aenum import extend_enum
from sentence_transformers import SentenceTransformer
from rewards import (
    format_reward_func,
    equation_reward_func,
    sentence_similarity_reward_func,
    reward_poem_form,
    rhyme_accuracy,
    syllable_accuracy
)
from functools import partial
from lighteval.metrics.metrics import SampleLevelMetric, MetricCategory, MetricUseCase, Metrics
import numpy as np
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import os


MATH_CALCULATION_QUERY_TEMPLATE = """
Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.
""".strip()

POETRY_QUERY_TEMPLATE = """
Using the author name : {author}, the title : {title}, the poem form : {form} and the start of an existing poem that is as follows : {poem_start}, create a new ending for this poem. You need to take into account the form the rhymes, the syllables and theme to create a new realistic ending. Show your work in <think> </think> where you are allowed to thinka bout structure, rhymes, etc. And return the final poem ending in <answer> </answer> tags. Think step by step inside <think> tags.
""".strip()
    
    
def math_calculation_metric(
    predictions: list[str],
    formatted_doc: Doc,
    **kwargs
) -> dict:
    
    fmt_score = format_reward_func(predictions)
    
    eq_score = equation_reward_func(
        completions=predictions,
        target=[formatted_doc.choices[1]],
        nums=[formatted_doc.choices[0]],
        )
        
    return {
        "format_score": fmt_score,
        "equation_score": eq_score,
        "avg_score": [(fmt_score[0] + eq_score[0]) / 2],
    }
    
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
similarity_reward_with_norm = partial(sentence_similarity_reward_func, sentence_model=sentence_model)
similarity_reward_with_norm.__name__ = "sentence_similarity_reward_func"


def poetry_calculation_metric(
    predictions: list[str], 
    formatted_doc: Doc, 
    **kwargs 
) -> dict:
    """
    Computes two scores:
    - format_score: how well the answer is wrapped in <answer> tags and uses the required format
    - equation_score: correctness of the equation versus the gold answer
    Returns a dict with both scores.
    """
    fmt_score = format_reward_func(predictions)
    
    
    similarity_score = similarity_reward_with_norm(
        completions=predictions,
        targets=[formatted_doc.choices[0]],
    )
    
    reward_poem_form_score = reward_poem_form(
        completions=predictions,
        targets=[formatted_doc.choices[0]]
    )
    
    rhyme_accuracy_score = rhyme_accuracy(
        completions=predictions,
        targets=[formatted_doc.choices[0]]
    )
     
    syllable_accuracy_score = syllable_accuracy(
        completions=predictions,
        targets=[formatted_doc.choices[0]]
    )
       
    return {
        "format_score": fmt_score,
        "similarity_score": similarity_score,
        "reward_poem_form_score": reward_poem_form_score,
        "rhyme_accuracy_score": rhyme_accuracy_score,
        "syllable_accuracy_score": syllable_accuracy_score,
        "agg_score": [(0.1 * fmt_score[0] 
                       + 0.25 * similarity_score[0] 
                       + 0.25 * reward_poem_form_score[0] 
                       + 0.1 * rhyme_accuracy_score[0] 
                       + 0.3 * syllable_accuracy_score[0])],
    }
    
    
def corpus_level_fn_math(values: list[dict[str, list[float]]]) -> dict[str, float]:
    return {
        "format_score": np.mean([v["format_score"][0] for v in values]),
        "equation_score": np.mean([v["equation_score"][0] for v in values]),
        "avg_score": np.mean([v["avg_score"][0] for v in values]),
    }
    
def corpus_level_fn_poetry(values: list[dict[str, list[float]]]) -> dict[str, float]:
    return {
        "format_score": np.mean([v["format_score"][0] for v in values]),
        "similarity_score": np.mean([v["similarity_score"][0] for v in values]),
        "reward_poem_form_score": np.mean([v["reward_poem_form_score"][0] for v in values]),
        "rhyme_accuracy_score": np.mean([v["rhyme_accuracy_score"][0] for v in values]),
        "syllable_accuracy_score": np.mean([v["syllable_accuracy_score"][0] for v in values]),
        "agg_score": np.mean([v["agg_score"][0] for v in values]),
    }

math_calculation_metric = SampleLevelMetric(
    metric_name="math_calculation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=math_calculation_metric,
    corpus_level_fn=corpus_level_fn_math,
)

poetry_calculation_metric = SampleLevelMetric(
    metric_name="poetry_calculation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=poetry_calculation_metric,
    corpus_level_fn=corpus_level_fn_poetry,
)

extend_enum(Metrics, "math_calculation", math_calculation_metric)

extend_enum(Metrics, "poetry_calculation", poetry_calculation_metric)


def math_calculation_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_CALCULATION_QUERY_TEMPLATE.format(numbers=line["nums"], target=line["target"]),
        choices=[line["nums"], line["target"], line["gold_answer"]],
        gold_index=0,
    )

def poetry_calculation_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=POETRY_QUERY_TEMPLATE.format(author=line["author"], title=line["title"], poem_start=line["poem_start"], form=line["form"]),
        choices=[line["poem_end"], line["poem_start"], line["author"], line["title"],  line["form"]],
        gold_index=0,
    )

math_calulation = LightevalTaskConfig(
    name="math_calculation",
    suite=["custom"],
    prompt_function=math_calculation_prompt_fn,
    hf_repo="Jeremmmyyyyy/math",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=int(os.getenv("GEN_SIZE", 32768)),
    metric=[math_calculation_metric],
    version=1,
)

poetry_calulation = LightevalTaskConfig(
    name="poetry_calculation",
    suite=["custom"],
    prompt_function=poetry_calculation_prompt_fn,
    hf_repo="schifferlearning/Poetry-Categorized",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=int(os.getenv("GEN_SIZE", 32768)),
    metric=[poetry_calculation_metric],
    version=1,
)


TASKS_TABLE = []
TASKS_TABLE.append(math_calulation)
TASKS_TABLE.append(poetry_calulation)

if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))