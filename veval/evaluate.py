from tqdm import tqdm
from typing import Dict, Any

from .systems.template import SystemResponse, System
from .tasks.template import Task


class Evaluator():

    def __init__(self, system: System, task: Task):
        self._system = system
        self._task = task

    def evaluate(self) -> Dict[str, Any]:

        task_docs = self._task.doc_store.documents

        # TODO: Implement the logic to read from cached responses
        resps = []
        for instance in tqdm(
            self._task.instances, 
            desc="Obtaining system response"
        ):
            sys_resp = self._system.invoke(
                query=instance.query, docs=task_docs)
            # sys_resp.context = sys_resp.context["vector_retriever"]
            sys_resp.context = sys_resp.context["reranker"]
            resps.append(sys_resp)

        result = self._task.process_results(
            insts=self._task.instances,
            resps=resps,
        )

        return result