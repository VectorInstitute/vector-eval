import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from inspect_ai.solver import Generate, Solver, TaskState, Tool, solver, tool
from inspect_ai.solver._tool.tool import ToolResult
from inspect_ai.util import concurrency

from .config import DEFAULT_SYSTEM_CONFIG

RAG_SOLVER_TEMPLATE = """\
{previous_prompt}

Context: {rag_context}
"""


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


@dataclass
class SystemConfig:
    """
    Represents the configuration for an abstract system.

    Attributes:
        llm_name (Optional[str]): The name of the llm used for generation.
        llm_gen_args (Optional[Dict[str, Any]]): The generating arguments for the llm.
        prompt_template (Optional[str]): The template for the generation prompt.

    Methods:
        __getitem__(self, key: Any) -> Any: Returns the value of the specified attribute.
        __setitem__(self, key: Any, value: Any) -> None: Sets the value of the specified attribute.
        as_dict(self) -> Dict[str, Any]: Convert the SystemConfig object to a dictionary.
    """

    sys_name: Optional[str] = None
    llm_name: Optional[str] = None
    llm_gen_args: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        return setattr(self, key, value)
    
    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class System(abc.ABC):
    """
    Abstract base class for RAG systems. Implements an interface that accepts a query 
    with list of documents (optional) and returns a (query, answer, constext) tuple.
    """

    _cfg = SystemConfig()

    def __init__(self, **kwargs):
        # Fetch and update system config
        self._set_default_cfg()
        self._update_cfg(**kwargs)
        
        self.name = self._cfg.sys_name

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

    def _set_default_cfg(self) -> None:
        try:
            default_cfg = DEFAULT_SYSTEM_CONFIG[type(self).__name__]
        except KeyError:
            print(f"Default config not found for system {type(self).__name__}, loading `BasicRag` config instead.")
            default_cfg = DEFAULT_SYSTEM_CONFIG["BasicRag"]
        for k, v in default_cfg.items():
            self._cfg[k] = v

    def _update_cfg(self, **kwargs) -> None:
        if self._cfg is None:
            raise ValueError("Load default config first by calling `self._get_default_cfg()`.")
        if kwargs is not None:
            for k, v in kwargs.items():
                self._cfg[k] = v

    def get_cfg(self):
        if self._cfg is None:
            raise ValueError("System config not set.")
        return self._cfg.as_dict()

    def get_inspect_tool(
        self,
        documents: list[str],
        max_concurrency: int = 1,
    ) -> Callable[..., Tool]:
        """
        Return tool interface compatible with inspect_ai.

        Adapted from the inspect_ai web search tool.
        """

        @tool(
            prompt="""Please use retrieval-augmented generation to assist in answering the question."""
        )
        def document_search():
            async def execute(query: str) -> tuple[ToolResult, dict[str, Any]]:
                """
                Tool for searching the local knowledgebase.

                Args:
                    query (str): Search query.
                """
                async with concurrency("document_search", max_concurrency):
                    response = self.invoke(query, documents)

                return response.answer, {
                    "document_search": {
                        "query": query,
                        "answer": response.answer,
                        "results": response.context,
                    }
                }

            return execute

        return document_search

    def get_inspect_solver(
        self,
        documents: list[str],
        max_concurrency: int = 1,
    ) -> Callable[..., Solver]:
        """
        Return solver compatible with inspect_ai.

        Example: insert RAG context into prompt before invoking model.

        Entire input would be sent to RAG pipeline as query.
        """

        @solver("RAG")
        def _rag_solver() -> Solver:
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                query = state.user_prompt.text
                async with concurrency("document_search", max_concurrency):
                    response = self.invoke(query, documents)

                state.user_prompt.text = RAG_SOLVER_TEMPLATE.format(
                    previous_prompt=query,
                    rag_context=response.answer,
                )
                state.metadata["rag"] = response
                return state

            return solve

        return _rag_solver
