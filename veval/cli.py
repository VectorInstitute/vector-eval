import argparse
import os

from veval.evaluate import Evaluator
from veval.systems.basic_rag import BasicRag
from veval.tasks.template import Task
from veval.utils.io_utils import load_from_yaml


def main(args):
    task = args.task
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tasks/{task}/{task}.yaml")
    cfg = load_from_yaml(cfg_path)
    
    task_obj = Task(config=cfg)
    task_obj.build()
    
    rag_sys_obj = BasicRag()

    eval_obj = Evaluator(system=rag_sys_obj, task=task_obj)
    output = eval_obj.evaluate()

    print(output)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pubmedqa", help="Specify the task")
    args = parser.parse_args()

    main(args)