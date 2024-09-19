import os
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.solver import generate, prompt_template


from veval.tasks.template import Task as _Task
from veval.utils.io_utils import load_from_yaml
from veval.metrics.template import get_inspect_scorer

from copy import deepcopy


# Read OpenAI API key
try:
    f = open(Path.home() / ".openai.key", "r")
    os.environ["OPENAI_API_KEY"] = f.read().rstrip("\n")
    f.close()
except Exception as err:
    print(f"Could not read your OpenAI API key: {err}")


def create_few_shot_exemplars(dataset, num_exemplars=5, seed=42):
    dataset.shuffle(seed=seed)
    exemplars = dataset[:num_exemplars]
    exemplars_str = []
    for ex_elm in exemplars:
        exemplars_str.append(f"QUESTION: {ex_elm.input}\nANSWER: {ex_elm.target}")
    return "\n\n".join(exemplars_str)


limit = 88

NUM_EXEMPLARS = 5

# ZERO_SHOT_PROMPT_TEMPLATE = {
#     "base": """QUESTION: {prompt}\nANSWER:""",
#     "chat": """Provide an answer to the following QUESTION.\n\nQUESTION: {prompt}\nANSWER:""",
# }

FEW_SHOT_PROMPT_TEMPLATE = {
    "base": """{exemplars}\n\nQUESTION: {prompt}\nANSWER:""",
    "chat": """Provide an answer to the following QUESTION. Some examples are provided below.\n\n{exemplars}\n\nQUESTION: {prompt}\nANSWER:""",
}

multihop_rag_dataset = hf_dataset(
    "vector-institute/MultiHopRAG-syn-data-ctx_len-4096-100",
    split="train",  # "train" is the only split in the dataset.
    sample_fields=FieldSpec(
        input="question",
        target="ground_truth",
        metadata=["contexts", "evolution_type", "metadata"],
    ),
    limit=limit,
)

FEW_SHOT_PROMPT_TEMPLATE = {
    k: v.format(
        exemplars=create_few_shot_exemplars(
            deepcopy(multihop_rag_dataset), num_exemplars=NUM_EXEMPLARS
        ),
        prompt="{prompt}",
    )
    for k, v in FEW_SHOT_PROMPT_TEMPLATE.items()
}

task_cfg = load_from_yaml("tasks/multihop-rag-syn-ctx-4096-100/multihop-rag-syn.yaml")
task_obj = _Task(config=task_cfg, limit=limit)
task_obj.build()
assert len(task_obj.doc_store.documents) > 0

ragas_scorer = get_inspect_scorer(
    "openai-gpt-4o-2024-08-06",
    ragas_feature_names=["correctness_answer"],
)


@task
def multihop_rag():
    return Task(
        dataset=multihop_rag_dataset,
        plan=[
            prompt_template(template=FEW_SHOT_PROMPT_TEMPLATE["chat"]),
            generate(),
        ],
        scorer=ragas_scorer(),
    )
