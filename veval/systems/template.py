import abc

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SystemResponse:
    """
    A class to represent a system response.

    Attributes:
        query (str): The query that was asked.
        answer (str): The answer that was returned.
        context (Dict[str, List[str]]): Key value contexts that are returned.
    """
    query: str
    answer: str
    context: Dict[str, List[str]]


# TODO: 
# 1. Make system configurable for swapping models and other parameters.
# 2. Replace local models with API calls.
class System(abc.ABC):
    """
    Abstract base class for RAG systems. Implements an interface that accepts a query 
    with list of documents (optional) and returns a (query, answer, constext) tuple.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def invoke(self, query: str, docs: Optional[List[str]]) -> SystemResponse:
        """
        Query the system and return the query, answer, and contexts.

        Args:
            query (str): The query string.
            docs (List[str]): The list of document strings.

        Returns:
            SystemResponse: A tuple containing the query, 
                generated answer, and all contexts.
        """
        pass