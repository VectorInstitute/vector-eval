import os

from pydantic.utils import deep_update
from tqdm import tqdm
from typing import Dict, Any

from .systems.template import System
from .tasks.template import Task
from .utils.io_utils import read_from_json, write_to_json


class Evaluator():
    """Class to evaluate a system on a task."""
    def __init__(self, system: System, task: Task, log_file: str = None):
        """
        Initializes a Evaluator object.

        Args:
            system (System): The RAG system to be evaluated.
            task (Task): The task to evaluate the RAG system on.
        """
        self._system = system
        self._task = task
        self.log_file = log_file

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

        result = {
            "num_samples": len(self._task.instances),
            "scores": result,
        }

        # Log responses if log_file provided
        if self.log_file is not None:
            self._log_responses(
                insts=self._task.instances,
                resps=resps,
                result=result,
            )

        return result

    def _log_responses(self, insts, resps, result) -> None:

        responses = []
        for inst, resp in zip(insts, resps):
            responses.append(
                {
                    "query": inst.query,
                    "context": resp.context,
                    "answer": resp.answer,
                    "gt_answer": inst.gt_answer,
                    "gt_context": inst.gt_context,
                }
            )
        
        log_data = {
            self._task.config.task_name: {
                self._system.name: {
                    self._system.get_cfg()["llm_name"]: {
                        "responses": responses,
                        "result": result,
                    }
                }
            }
        }

        if os.path.exists(self.log_file):
            prev_log_data = read_from_json(self.log_file)
            log_data = deep_update(prev_log_data, log_data)

        write_to_json(log_data, self.log_file)
