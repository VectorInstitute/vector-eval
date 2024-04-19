import abc
import datasets
import os

from collections import defaultdict
from dataclasses import dataclass
from inspect import getsource
from tqdm import tqdm
from typing import (
    List, Optional, Dict, Any
)

from veval.registry import get_metric
from veval.systems.template import SystemResponse


@dataclass
class Instance:
    """
    A class to represents an instance of a query and its 
    corresponding ground truth answer and context (if available).
    
    Attributes:
        query (str): The query string.
        gt_answer (str): The ground truth answer string.
        gt_context (Optional[List[str]]): The ground truth context 
            as a list of strings, or None if not available.
    """
    query: str
    gt_answer: str
    gt_context: Optional[List[str]]


@dataclass
class DocumentStore:
    """
    A class representing a document store.

    Attributes:
        documents (List[str]): A list of documents stored in the document store.
        docs_path (str): The path to the document store.
    """
    documents: List[str]
    docs_path: str


@dataclass
class TaskConfig(dict):
    """
    Represents the configuration for a task.

    Attributes:
        task_name (Optional[str]): The name of the task.
        dataset_path (Optional[str]): The path to the dataset.
        dataset_name (Optional[str]): The name of a specific subset of the dataset.
        dataset_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the dataset.
        validation_split (Optional[str]): The validation split alias for the dataset.
        data_instance_map (Optional[Dict[str, Any]]): A mapping of fields for the data instance.
        docs_path (Optional[str]): The path to the document store for the dataset.
        metric_list (Optional[list]): A list of metrics to be computed.

    Methods:
        __getitem__(self, key: Any) -> Any: Returns the value of the specified attribute.
        __setitem__(self, key: Any, value: Any) -> None: Sets the value of the specified attribute.
    """
    task_name: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_kwargs: Optional[Dict[str, Any]] = None
    validation_split: Optional[str] = None
    data_instance_map: Optional[Dict[str, Any]] = None
    docs_path: Optional[str] = None
    metric_list: Optional[list] = None

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        return setattr(self, key, value)
    

class Task(abc.ABC):
    """
    Class to represent a task. Implements an interface for loading 
    dataset and documents as well as calculating metrics.
    """
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        """
        Initializes a Task object.

        Args:
            config (Optional[dict]): A dictionary containing the configuration 
                for the task. If None, a ValueError is raised.
        """
        if config is None:
            raise ValueError("Task configuration must be provided in `config` kwarg.")
        self.config = TaskConfig(**config)
        self.data_instance_map = self.config.data_instance_map

        self._metric_fn_list = {}
        self._metric_fn_args_list = {}
        for metric_cfg in self.config.metric_list:
            metric_name = metric_cfg["metric"]
            self._metric_fn_list[metric_name] = get_metric(metric_name)
            self._metric_fn_args_list[metric_name] = metric_cfg["args"]

        self.download(self.config.dataset_kwargs)

        self.instances: List[Instance] = None
        self.doc_store: DocumentStore = None

        self.limit = kwargs.get("limit", None)

    def build(self) -> None:
        """Loads the dataset and constructs the documents store."""
        eval_data = self.validation_dataset()

        if not os.path.exists(self.config.docs_path):
            self._create_doc_store(eval_data)
        
        self.doc_store = DocumentStore(
            documents=self.read_docs(),
            docs_path=self.config.docs_path
        )

        instances = []
        for elm_idx, elm in tqdm(enumerate(eval_data), desc="Building instances"):
            instances.append(self.construct_instance(elm))
        if self.limit is not None:
            self.instances = instances[:self.limit]

    # Code borrowed from: 
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L870C5-L875C10
    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Downloads the dataset from the specified path.

        Args:
            dataset_kwargs (Optional[Dict[str, Any]]): 
                Additional keyword arguments to be passed.
        """
        self.dataset = datasets.load_dataset(
            path=self.config.dataset_path,
            name=self.config.dataset_name,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )

    def construct_instance(self, elm: Dict) -> Instance:
        """
        Constructs an instance of the `Instance` class using the provided input.

        Args:
            elm (dict): The input dictionary containing an element of the dataset.

        Returns:
            Instance: An instance of the `Instance` class.
        """
        return Instance(
            query=elm[self.data_instance_map.get("query")],
            gt_answer=elm[self.data_instance_map.get("gt_answer")],
            gt_context=(
                elm[self.data_instance_map.get("gt_context")] 
                if "gt_context" in self.data_instance_map else None
            ),
        )

    # TODO - How to generalize this? Needs more brainstorming.
    def _create_doc_store(self, data: datasets.Dataset) -> None:
        """
        Creates a document store by writing the ground truth context 
        of each element in the given dataset to separate text files.

        Args:
            data (datasets.Dataset): The dataset containing the elements.
        """
        os.makedirs(self.config.docs_path, exist_ok=True)
        for idx, elm in tqdm(enumerate(data), desc="Creating document store"):
            filepath = os.path.join(self.config.docs_path, f'doc_{idx+1}.txt')
            context = elm[self.data_instance_map.get("gt_context")]
            if isinstance(context, list):
                context = "\n".join(context)
            with open(filepath, 'w') as f:
                f.write(context)

    def read_docs(self) -> List[str]:
        """
        Reads and returns a list of documents from the document store.

        Returns:
            List[str]: A list of document strings.
        """
        docs = []
        for filename in tqdm(
            os.listdir(self.config.docs_path), 
            desc="Reading documents"
        ):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.config.docs_path, filename)
                with open(filepath, 'r') as f:
                    docs.append(f.read().strip().strip('\n'))
        return docs

    def process_results(self, insts: List[Instance], resps: List[SystemResponse]):
        """
        Calculate metrics associated with the task.

        Args:
            insts (List[Instance]): A list of instances representing the evaluation queries.
            resps (List[SystemResponse]): A list of system responses corresponding to the evaluation queries.

        Returns:
            dict: A dictionary containing the evaluation results, 
                where the keys are the metric names and the values are the computed metric scores.
        """
        assert len(insts) == len(resps), f"Mismatch between number of instances ({len(insts)}) and responses ({len(resps)})."

        # Handle multiple intermediate contexts from the system. Evaluate all metrics for all contexts.
        # TODO: Think of a less redundant logic.
        context_keys = resps[0].context.keys()
        inputs = defaultdict()
        for ctx_k in context_keys:
            inputs[ctx_k] = {
                "query": [],
                "context": [],
                "answer": [],
                "gt_answer": [],
                "gt_context": [],
            }
        for inst, resp in zip(insts, resps):
            for ctx_k in context_keys:
                inputs[ctx_k]["query"].append(inst.query)
                inputs[ctx_k]["context"].append(resp.context[ctx_k])
                inputs[ctx_k]["answer"].append(resp.answer)
                inputs[ctx_k]["gt_answer"].append(inst.gt_answer)
                inputs[ctx_k]["gt_context"].append(inst.gt_context)

        result_dict = {k: defaultdict() for k in context_keys}
        for ctx_k in context_keys:
            for metric_name, metric_fn in self._metric_fn_list.items():
                result_dict[ctx_k][metric_name] = metric_fn(
                    **{k: v for k, v in inputs[ctx_k].items() if k in self._metric_fn_args_list[metric_name]}
                )

        return result_dict
    

    def validation_dataset(self) -> datasets.Dataset:
        return self.dataset[self.config.validation_split]