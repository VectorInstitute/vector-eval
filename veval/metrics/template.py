import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, faithfulness, context_relevancy, 
    answer_correctness, answer_similarity,
)

# from veval.registry import register_metric


# Read OpenAI API key
try:
    f = open(Path.home() / ".openai.key", "r")
    os.environ["OPENAI_API_KEY"] = f.read().rstrip("\n")
    f.close()
except Exception as err:
    print(f"Could not read your OpenAI API key: {err}")


# @register_metric(
#     metric="relevance_query_answer",
#     higher_is_better=True,
#     requires_ground_truth=False,
# )
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

def answer_correctness(
        query: List[str],
        answer: List[str],
        gt_answer: List[str],
    ) -> float:
    return LLMJudgeMetrics().answer_correctness(
            query=query,
            answer=answer,
            gt_answer=gt_answer
        )


# Define class for metrics which use LLM as a judge
class LLMJudgeMetrics:
    def __init__(self, openai_model: str = "gpt-3.5-turbo-0125") -> None:
        self._openai_model = ChatOpenAI(
            model_name=openai_model, 
            temperature=0.0,
            max_tokens=128
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
            Dict[str, float]: A single element dict with the query-answer relevancy score.
        """
        data = Dataset.from_dict({
            "question": query,
            "answer": answer,
            "contexts": context
        })
        score = evaluate(
            dataset=data,
            metrics=[answer_relevancy],
            llm=self._openai_model,
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
            Dict[str, float]: A single element dict with the context-answer groundedness score.
        """
        data = Dataset.from_dict({
            "question": query,
            "answer": answer,
            "contexts": context
        })
        score = evaluate(
            dataset=data,
            metrics=[faithfulness],
            llm=self._openai_model,
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
            Dict[str, float]: A single element dict with the query-context relevancy score.
        """
        data = Dataset.from_dict({
            "question": query,
            "contexts": context
        })
        score = evaluate(
            dataset=data,
            metrics=[context_relevancy],
            llm=self._openai_model,
        )
        return score.get("context_relevancy")
    
    def answer_correctness(
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
            Dict[str, float]: A single element dict with the query-context relevancy score.
        """
        data = Dataset.from_dict({
            "question": query,
            "answer": answer,
            "ground_truth": gt_answer
        })
        score = evaluate(
            dataset=data,
            metrics=[answer_correctness],
            llm=self._openai_model,
        )
        return score.get("answer_correctness")