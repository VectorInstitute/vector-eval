import os
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset, csv_dataset
from inspect_ai.solver import generate

from veval.systems.basic_rag import BasicRag

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


# def create_few_shot_exemplars(dataset, retriever_kwargs, num_exemplars=5, seed=42):
#     dataset.shuffle(seed=seed)
#     exemplars = dataset[:num_exemplars]
#     # retriever = retriever_kwargs["retriever"]
#     # docs = retriever_kwargs["docs"]
#     exemplars_str = []
#     for ex_elm in exemplars:
#         # context = retriever.invoke(ex_elm.input, docs)
#         # context_str = "\n".join(
#         #     [
#         #         (f"{(idx+1)}. " + elm.strip("\n"))
#         #         for idx, elm in enumerate(context.context["vector_retriever"])
#         #     ]
#         # )
#         # exemplars_str.append(
#         #     f"QUESTION: {ex_elm.input}\nCONTEXT:\n{context_str}\nANSWER: {ex_elm.target}"
#         # )
#         exemplars_str.append(
#             f"QUESTION: {ex_elm.input}\nANSWER: {ex_elm.target}"
#         )
#     return "\n\n".join(exemplars_str)


limit = 67
max_concurrency = 1

# NUM_EXEMPLARS = 5

RAG_ZERO_SHOT_PROMPT_TEMPLATE = {
    "base": """QUESTION: {query}\nCONTEXT:\n{context}\nANSWER:""",
    "chat": """Provide an answer to the following QUESTION. You are allowed to use the CONTEXT given below for answering the QUESTION.\n\nQUESTION: {query}\nCONTEXT:\n{context}\nANSWER:""",
}

# RAG_FEW_SHOT_PROMPT_TEMPLATE = {
#     "base": """{exemplars}\n\nQUESTION: {query}\nCONTEXT:\n{context}\nANSWER:""",
#     "chat": """Provide an answer to the following QUESTION. You are allowed to use the CONTEXT given below for answering the QUESTION. Some examples are provided below.\n\n{exemplars}\n\nQUESTION: {query}\nCONTEXT:\n{context}\nANSWER:""",
# }

task_cfg = load_from_yaml("tasks/legal-data-syn-120-v1/config.yaml")

# qa_dataset = hf_dataset(
#     "vector-institute/MultiHopRAG-syn-data-ctx_len-4096-100",
#     split="train",  # "train" is the only split in the dataset.
#     sample_fields=FieldSpec(
#         input="question",
#         target="ground_truth",
#         metadata=["contexts", "evolution_type", "metadata"],
#     ),
#     limit=limit,
# )
qa_dataset = csv_dataset(
    task_cfg["dataset_path"],
    sample_fields=FieldSpec(
        input="question",
        target="ground_truth",
        metadata=["contexts", "evolution_type", "metadata"],
    ),
    limit=limit,
)
print(len(qa_dataset))

task_obj = _Task(config=task_cfg, limit=limit)
task_obj.build()
assert len(task_obj.doc_store.documents) > 0

retrieval_system = BasicRag(
    sys_name="basic_rag",
    llm_name="openai-gpt-3.5-turbo",  # NOTE: Not used since retriever_only is True
    embed_model_name="/fs01/projects/opt_test/rag-knowledge-cutoff-embed-models/bge-large-en-v1.5",  # bge-large-en-v1.5 # NV-Embed-v1
    retriever_only=True,
)

# RAG_FEW_SHOT_PROMPT_TEMPLATE = {
#     k: v.format(
#         exemplars=create_few_shot_exemplars(
#             deepcopy(multihop_rag_dataset),
#             retriever_kwargs={
#                 "retriever": retrieval_system,
#                 "docs": task_obj.doc_store.documents,
#             },
#             num_exemplars=NUM_EXEMPLARS,
#         ),
#         query="{query}",
#         context="{context}",
#     )
#     for k, v in RAG_FEW_SHOT_PROMPT_TEMPLATE.items()
# }

document_search_solver = retrieval_system.get_inspect_solver(
    documents=task_obj.doc_store.documents,
    rag_prompt_template=RAG_ZERO_SHOT_PROMPT_TEMPLATE["chat"],
    max_concurrency=max_concurrency,
)

ragas_scorer = get_inspect_scorer(
    "openai-gpt-4o-2024-08-06",
    ragas_feature_names=["correctness_answer"],
    max_concurrency=max_concurrency,
)


@task
def multihop_rag():
    return Task(
        dataset=qa_dataset,
        plan=[
            document_search_solver(),
            generate(),
        ],
        scorer=ragas_scorer(),
    )
