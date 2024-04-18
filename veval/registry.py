from typing import Callable

from veval.metrics.template import (
    relevance_query_answer,
    groundedness_context_answer,
    relevance_query_context,
    answer_correctness,
)


METRIC_REGISTRY = {
    "relevance_query_answer": relevance_query_answer,
    "groundedness_context_answer": groundedness_context_answer,
    "relevance_query_context": relevance_query_context,
    "answer_correctness": answer_correctness,
}
HIGHER_IS_BETTER_REGISTRY = {}
REQUIRES_GROUND_TRUTH = {}


def get_metric(name: str) -> Callable:
    if name in METRIC_REGISTRY:
        return METRIC_REGISTRY[name]
    else:
        raise Exception(f"Could not find registered metric '{name}'.")


# Code borrowed from: 
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/registry.py#L93C1-L119C20
def register_metric(**args):
    def decorate(fn):
        assert "metric" in args
        name = args["metric"]

        for key, registry in [
            ("metric", METRIC_REGISTRY),
            ("higher_is_better", HIGHER_IS_BETTER_REGISTRY),
            ("requires_ground_truth", REQUIRES_GROUND_TRUTH),
        ]:
            if key in args:
                value = args[key]
                assert (
                    value not in registry
                ), f"{key} named '{value}' conflicts with existing registered {key}!"

                if key == "metric":
                    registry[name] = fn
                else:
                    registry[name] = value

        return fn

    return decorate