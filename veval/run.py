import argparse
import datetime
import os
from pathlib import Path
from typing import Optional

from veval.evaluate import Evaluator
from veval.systems.basic_rag import BasicRag
from veval.systems.rerank_rag import RerankRag
from veval.tasks.template import Task
from veval.utils.io_utils import load_from_yaml, write_to_json, read_from_json
from collections import defaultdict


def run_evaluation(
    task: str,
    system: str,
    model: str,
    limit: Optional[int] = None,
    log_file: Optional[str] = None,
):  
    log_msg = f"""
    Evaluating the following configuration:
    Task: {task}
    System: {system}
    Model: {model}\n
    """
    print(log_msg)

    parent_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        task_cfg_path = os.path.join(parent_dir, f"tasks/{task}/{task}.yaml")
        task_cfg = load_from_yaml(task_cfg_path)
    except FileNotFoundError:
        raise Exception(f"Task {task} not supported or the configuration file does not exists.")
        
    task_obj = Task(config=task_cfg, limit=limit)
    task_obj.build()
    
    if system == "basic_rag":
        rag_sys_obj = BasicRag(sys_name=system, llm_name=model)
    elif system == "rerank_rag":
        rag_sys_obj = RerankRag(sys_name=system, llm_name=model)
    else:
        raise ValueError(f"System {system} not supported.")

    eval_obj = Evaluator(system=rag_sys_obj, task=task_obj, log_file=log_file)
    result = eval_obj.evaluate()

    output = {
        "task_cfg": task_obj.config.as_dict(),
        "system_cfg": rag_sys_obj.get_cfg(),
        "result": result,
        "metadata": {
            "limit": limit,
            "log_file": log_file.lstrip(parent_dir),
        }
    }
    
    return output


def main(args):
    tasks = args.tasks
    systems = args.systems
    models = args.models
    limit = int(args.limit) if args.limit != -1 else None
    log_dir = args.log_dir

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(parent_dir, log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_file = os.path.join(log_dir, f"{timestamp}.json")

    for task in tasks:
        results = defaultdict()
        for system in systems:
            results[system] = defaultdict()
            for model in models:
                output = run_evaluation(
                    task=task,
                    system=system,
                    model=model,
                    limit=limit,
                    log_file=log_file
                )
                results[system][model] = output

        out_path = f"tasks/{task}/results.json"
        out_path = os.path.join(parent_dir, out_path)
        if os.path.exists(out_path):
            prev_results = read_from_json(out_path)
            results.update(prev_results)
            
        write_to_json(results, out_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", type=str, default="pubmedqa", nargs="+", help="Specify the tasks to evaluate.")
    parser.add_argument("--systems", type=str, default="basic_rag", nargs="+", help="Specify the systems to evaluate the tasks.")
    parser.add_argument("--models", type=str, default="openai-gpt-3.5", nargs="+", help="Specify the models used for generation.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of instances.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Specify the log directory.")

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