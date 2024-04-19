import argparse
import os
from pathlib import Path

from veval.evaluate import Evaluator
from veval.systems.basic_rag import BasicRag
from veval.systems.rerank_rag import RerankRag
from veval.tasks.template import Task
from veval.utils.io_utils import load_from_yaml


def main(args):
    task = args.task
    system = args.sys

    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tasks/{task}/{task}.yaml")
        cfg = load_from_yaml(cfg_path)
    except FileNotFoundError:
        raise Exception(f"Task {task} not supported or the configuration file does not exists.")
        
    task_obj = Task(config=cfg)
    task_obj.build()
    
    if system == "basic_rag":
        rag_sys_obj = BasicRag()
    elif system == "rerank_rag":
        rag_sys_obj = RerankRag()
    else:
        raise ValueError(f"System {system} not supported.")

    eval_obj = Evaluator(system=rag_sys_obj, task=task_obj)
    output = eval_obj.evaluate()

    print(output)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="pubmedqa", help="Specify the task.")
    parser.add_argument("--sys", type=str, default="basic_rag", help="Specify the system.")

    args = parser.parse_args()

    # Read OpenAI API key
    try:
        f = open(Path.home() / ".openai.key", "r")
        os.environ["OPENAI_API_KEY"] = f.read().rstrip("\n")
        f.close()
    except Exception as err:
        print(f"Could not read your OpenAI API key: {err}")

    main(args)