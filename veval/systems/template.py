import abc
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SystemResponse:
    """A class to represent a system response.

    Attributes:
        query (str): The query that was asked.
        answer (str): The answer that was returned.
        context (Dict[str, List[str]]): Key value contexts that are returned.
    """

    query: str
    answer: str
    context: Dict[str, List[str]]


class System(abc.ABC):
    """Abstract base class for systems. 

        Implements an interface that accepts a query and returns a (query, answer, context) tuple.

    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def invoke(self, query: str) -> SystemResponse:
        """Query the system and return the query, answer, and context.

        Args:
            query (str): The query to ask the system.

        Returns:
            SystemResponse: A tuple containing the query, answer, and context.
        """
        pass