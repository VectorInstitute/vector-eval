from typing import TYPE_CHECKING, Callable, List

from datasets import Dataset
from inspect_ai.scorer import Score, Scorer, Target, bootstrap_std, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import concurrency
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_relevancy,
    faithfulness,
)

from utils.model_utils import LangChainLLM

if TYPE_CHECKING:
    from systems.template import SystemResponse


def relevance_query_answer(
        query: List[str], 
        context: List[List[str]], 
        answer: List[str]
    ) -> float:
    return LLMJudgeMetrics().relevance_query_answer(
            query=query,
            context=context,
            answer=answer
        )

def groundedness_context_answer(
        query: List[str], 
        context: List[List[str]], 
        answer: List[str]
    ) -> float:
    return LLMJudgeMetrics().groundedness_context_answer(
            query=query,
            context=context,
            answer=answer
        )

def relevance_query_context(
        query: List[str], 
        context: List[List[str]] 
    ) -> float:
    return LLMJudgeMetrics().relevance_query_context(
            query=query,
            context=context,
        )

def correctness_answer(
        query: List[str],
        answer: List[str],
        gt_answer: List[str],
    ) -> float:
    return LLMJudgeMetrics().correctness_answer(
            query=query,
            answer=answer,
            gt_answer=gt_answer
        )


# Define class for metrics which use LLM as a judge
class LLMJudgeMetrics:
    """Class for metrics which use LLM as a judge."""
    def __init__(self, llm_name: str = "openai-gpt-3.5-turbo") -> None:
        self._judge_llm = LangChainLLM(
            lm_name=llm_name,
        )

    def relevance_query_answer(
            self, 
            query: List[str], 
            context: List[List[str]], 
            answer: List[str]
        ) -> float:
        """
        Calculate score based on relevancy of answer to the query.
        Reference: https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html

        Args:
            query (List[str]): List of query/question, one for each sample.
            context (List[List[str]]): List of retrieved contexts, one/many for each sample.
            answer (List[str]): List of answer, one for each sample.

        Returns:
            float: The query-answer relevancy score.
        """
        data = Dataset.from_dict({
            "question": query,
            "answer": answer,
            "contexts": context
        })
        score = evaluate(
            dataset=data,
            metrics=[answer_relevancy],
            llm=self._judge_llm,
        )
        return score.get("answer_relevancy")
    
    def groundedness_context_answer(
            self, 
            query: List[str], 
            context: List[List[str]], 
            answer: List[str]
        ) -> float:
        """
        Calculate score based on groundedness of the answer on the retrieved contexts.
        Reference: https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html

        Args:
            query (List[str]): List of query/question, one for each sample.
            context (List[List[str]]): List of retrieved contexts, one/many for each sample.
            answer (List[str]): List of answer, one for each sample.

        Returns:
            float: The context-answer groundedness score.
        """
        data = Dataset.from_dict({
            "question": query,
            "answer": answer,
            "contexts": context
        })
        score = evaluate(
            dataset=data,
            metrics=[faithfulness],
            llm=self._judge_llm,
        )
        return score.get("faithfulness")
    
    def relevance_query_context(
            self, 
            query: List[str], 
            context: List[List[str]] 
        ) -> float:
        """
        Calculate score based on relevancy of retrieved context to the query.
        Reference: https://docs.ragas.io/en/stable/concepts/metrics/context_relevancy.html

        Args:
            query (List[str]): List of query/question, one for each sample.
            context (List[List[str]]): List of retrieved contexts, one/many for each sample.

        Returns:
            float: The query-context relevancy score.
        """
        data = Dataset.from_dict({
            "question": query,
            "contexts": context
        })
        score = evaluate(
            dataset=data,
            metrics=[context_relevancy],
            llm=self._judge_llm,
        )
        return score.get("context_relevancy")
    
    def correctness_answer(
            self, 
            query: List[str], 
            answer: List[str], 
            gt_answer: List[str],
        ) -> float:
        """
        Calculate answer correctness score based on both factual and semantic similarity.
        Reference: 
        1. https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html
        2. https://docs.ragas.io/en/stable/concepts/metrics/semantic_similarity.html

        Args:
            query (List[str]): List of query/question, one for each sample.
            answer (List[str]): List of answer, one for each sample.
            gt_answer (List[str]): List of ground truth answer, one for each sample.

        Returns:
            float: The answer correctness score.
        """
        data = Dataset.from_dict({
            "question": query,
            "answer": answer,
            "ground_truth": gt_answer
        })
        score = evaluate(
            dataset=data,
            metrics=[answer_correctness],
            llm=self._judge_llm,
        )
        return score.get("answer_correctness")


def get_inspect_scorer(
    judge_llm_name: str,
    max_concurrency: int = 1,
) -> Callable[..., Scorer]:
    """
    Return scorer compatible with AISI Inspect.

    This scorer runs on one example at a time and does not support batching.
    """

    _judge_llm = LangChainLLM(lm_name=judge_llm_name)

    @scorer(metrics=[bootstrap_std()])
    def _scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            rag_response: SystemResponse | None = state.metadata.get(
                "document_search", {}
            ).get("response")
            context = (
                list(rag_response.context.values()) if rag_response is not None else []
            )

            data = Dataset.from_dict(
                {
                    "question": [state.input_text],
                    "answer": [state.output.completion],
                    "ground_truth": [target.text],
                    "contexts": context,
                }
            )

            async with concurrency("ragas", max_concurrency):
                metrics = evaluate(dataset=data, llm=_judge_llm)

            return Score(value=metrics)

        return score

    return _scorer
