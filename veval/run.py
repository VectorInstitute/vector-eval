import argparse
import os
from pathlib import Path

from veval.evaluate import Evaluator
from veval.systems.basic_rag import BasicRag
from veval.systems.rerank_rag import RerankRag
from veval.tasks.template import Task
from veval.utils.io_utils import load_from_yaml, write_to_json, read_from_json


def main(args):
    task = args.task
    system = args.sys
    limit = int(args.limit) if args.limit != -1 else None

    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tasks/{task}/{task}.yaml")
        cfg = load_from_yaml(cfg_path)
    except FileNotFoundError:
        raise Exception(f"Task {task} not supported or the configuration file does not exists.")
        
    task_obj = Task(config=cfg, limit=limit)
    task_obj.build()
    
    if system == "basic_rag":
        rag_sys_obj = BasicRag(llm_name="cohere-command")
    elif system == "rerank_rag":
        rag_sys_obj = RerankRag(llm_name="cohere-command")
    else:
        raise ValueError(f"System {system} not supported.")

    eval_obj = Evaluator(system=rag_sys_obj, task=task_obj)
    output = eval_obj.evaluate()
    output = {
        f"{system}": output
    }

    out_path = f"tasks/{task}/results{(('_' + str(limit)) if limit is not None else '')}.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    if os.path.exists(out_path):
        results = read_from_json(out_path)
        results.update(output)
    else:
        results = output
    write_to_json(results, out_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="pubmedqa", help="Specify the task.")
    parser.add_argument("--sys", type=str, default="basic_rag", help="Specify the system.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of instances.")

    args = parser.parse_args()

    # Read OpenAI API key
    try:
        f = open(Path.home() / ".openai.key", "r")
        os.environ["OPENAI_API_KEY"] = f.read().rstrip("\n")
        f.close()
    except Exception as err:
        print(f"Could not read your OpenAI API key: {err}")

    # Read Cohere API key
    try:
        f = open(Path.home() / ".cohere.key", "r")
        os.environ["COHERE_API_KEY"] = f.read().rstrip("\n")
        f.close()
    except Exception as err:
        print(f"Could not read your Cohere API key: {err}")

    main(args)