"""Custom evaluation tasks for LightEval."""

from aenum import extend_enum

from rewards import (
    format_reward_func,
    equation_reward_func,   
)
from lighteval.metrics.metrics import SampleLevelMetric, MetricCategory, MetricUseCase, Metrics
import numpy as np
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import os


MATH_CALCULATION_QUERY_TEMPLATE = """
Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.
""".strip()


def math_calculation_metric(
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
    
def corpus_level_fn(values: list[dict[str, list[float]]]) -> dict[str, float]:
    return {
        "format_score": np.mean([v["format_score"][0] for v in values]),
        "equation_score": np.mean([v["equation_score"][0] for v in values]),
        "avg_score": np.mean([v["avg_score"][0] for v in values]),
    }

math_calculation_metric = SampleLevelMetric(
    metric_name="math_calculation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=math_calculation_metric,
    corpus_level_fn=corpus_level_fn,
)

extend_enum(Metrics, "math_calculation", math_calculation_metric)

    

def math_calculation_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_CALCULATION_QUERY_TEMPLATE.format(numbers=line["nums"], target=line["target"]),
        choices=[line["nums"], line["target"], line["gold_answer"]],
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


TASKS_TABLE = []
TASKS_TABLE.append(math_calulation)

if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))