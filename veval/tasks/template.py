import abc
import datasets
import os

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
    query: str
    gt_answer: str
    gt_context: Optional[List[str]]


@dataclass
class DocumentStore:
    documents: List[str]
    docs_path: str


@dataclass
class TaskConfig(dict):
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
    def __init__(self, config: Optional[dict] = None):
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

        self.limit = 2

    def build(self):
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
        self.instances = instances[:self.limit]

    # Code borrowed from: 
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L870C5-L875C10
    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.config.dataset_path,
            name=self.config.dataset_name,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )

    def construct_instance(self, elm) -> Instance:
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
        os.makedirs(self.config.docs_path, exist_ok=True)
        for idx, elm in tqdm(enumerate(data), desc="Creating document store"):
            filepath = os.path.join(self.config.docs_path, f'doc_{idx+1}.txt')
            context = elm[self.data_instance_map.get("gt_context")]
            if isinstance(context, list):
                context = "\n".join(context)
            with open(filepath, 'w') as f:
                f.write(context)

    def read_docs(self) -> List[str]:
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

        inputs = {
            "query": [inst.query for inst in insts],
            "context": [resp.context for resp in resps],
            "answer": [resp.answer for resp in resps],
            "gt_answer": [inst.gt_answer for inst in insts],
            "gt_context": [inst.gt_context for inst in insts],
        }

        result_dict = {}
        for metric_name, metric_fn in self._metric_fn_list.items():
            result_dict[metric_name] = metric_fn(
                **{k: v for k, v in inputs.items() if k in self._metric_fn_args_list[metric_name]}
            )

        return result_dict
    

    def validation_dataset(self) -> datasets.Dataset:
        return self.dataset[self.config.validation_split]