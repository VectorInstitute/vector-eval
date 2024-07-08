import os
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.solver import generate

from veval.systems.basic_rag import BasicRag

from veval.tasks.template import Task as _Task
from veval.utils.io_utils import load_from_yaml
from veval.metrics.template import get_inspect_scorer


# Read OpenAI API key
try:
    f = open(Path.home() / ".openai.key", "r")
    os.environ["OPENAI_API_KEY"] = f.read().rstrip("\n")
    f.close()
except Exception as err:
    print(f"Could not read your OpenAI API key: {err}")


limit = None

RAG_PROMPT_TEMPLATE = """
Provide an answer to the following QUESTION. You are allowed to use the CONTEXT given below for answering the QUESTION.

QUESTION: {query}

CONTEXT:
{context}
"""

multihop_rag_dataset = hf_dataset(
    "yixuantt/MultiHopRAG",
    split="train",  # "train" is the only split in the dataset.
    name="MultiHopRAG",
    sample_fields=FieldSpec(
        input="query", target="answer", metadata=["evidence_list", "question_type"]
    ),
    limit=limit,
)

task_cfg = load_from_yaml("tasks/multihop-rag/multihop-rag.yaml")
task_obj = _Task(config=task_cfg, limit=limit)
task_obj.build()
assert len(task_obj.doc_store.documents) > 0

retrieval_system = BasicRag(
    sys_name="basic_rag",
    llm_name="openai-gpt-3.5-turbo", # NOTE: Not used since retriever_only is True
    embed_model_name="openai-text-embedding-3-small",
    retriever_only=True,
)
document_search_solver = retrieval_system.get_inspect_solver(
    documents=task_obj.doc_store.documents,
    rag_prompt_template=RAG_PROMPT_TEMPLATE,
    max_concurrency=20,
)
ragas_scorer = get_inspect_scorer(
    "openai-gpt-4o",
    ragas_feature_names=[row["metric"] for row in task_cfg["metric_list"]],
)

print("task_obj.doc_store.documents", len(task_obj.doc_store.documents))
print("retrieval_system.faiss_dim", retrieval_system.faiss_dim)


@task
def multihop_rag():
    return Task(
        dataset=multihop_rag_dataset,
        plan=[
            document_search_solver(),
            generate(),
        ],
        scorer=ragas_scorer(),
    )
