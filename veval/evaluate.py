from tqdm import tqdm
from typing import Dict, Any

from .systems.template import System
from .tasks.template import Task


class Evaluator():
    """Class to evaluate a system on a task."""
    def __init__(self, system: System, task: Task):
        """
        Initializes a Evaluator object.

        Args:
            system (System): The RAG system to be evaluated.
            task (Task): The task to evaluate the RAG system on.
        """
        self._system = system
        self._task = task

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the system's response for each instance in the task.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results.
        """
        task_docs = self._task.doc_store.documents

        # TODO: Implement the logic to read from cached responses
        resps = []
        for instance in tqdm(
            self._task.instances, 
            desc="Obtaining system response"
        ):
            sys_resp = self._system.invoke(
                query=instance.query, docs=task_docs)
            resps.append(sys_resp)
        self._system.cleanup()

        result = self._task.process_results(
            insts=self._task.instances,
            resps=resps,
        )

        return result