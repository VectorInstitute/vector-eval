import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core import (
    Document, VectorStoreIndex, PromptTemplate, StorageContext, 
    get_response_synthesizer, load_index_from_storage
    )
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

from veval.utils.io_utils import delete_directory
from veval.utils.model_utils import LlamaIndexLLM

from .basic_rag import BasicRag, BasicRagConfig
from .template import SystemResponse


@dataclass
class RerankRagConfig(BasicRagConfig):
    rerank_llm_name: Optional[str] = None
    rerank_llm_gen_args: Optional[Dict[str, Any]] = None
    rerank_top_k: Optional[int] = None


class RerankRag(BasicRag):
    """A linear RAG system using a reranker model."""

    _cfg = RerankRagConfig()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rerank_llm = LlamaIndexLLM(
            lm_name=self._cfg.rerank_llm_name,
            **self._cfg.rerank_llm_gen_args,
        )
        self.rerank_top_k = self._cfg.rerank_top_k

    def invoke(self, query: str, docs: List[str]) -> SystemResponse:
        all_docs = [Document(text=doc) for doc in docs]

        # Load or create vector index
        if os.path.exists(self._index_dir):
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=self._index_dir)
            vector_index = load_index_from_storage(storage_context)
        else:
            os.makedirs(self._index_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store)
            vector_index = VectorStoreIndex.from_documents(
                documents=all_docs, 
                storage_context=storage_context,
                service_context=self.service_context,
            )
            vector_index.storage_context.persist(
                persist_dir=self._index_dir
            )

        # Build query engine
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self.similarity_top_k,
            service_context=self.service_context,
        )
        node_postprocessor = [
            LLMRerank(
                llm=self.rerank_llm,
                top_n=self.rerank_top_k,
            )
        ]
        response_synthesizer = get_response_synthesizer(
            response_mode=self.response_mode,
            text_qa_template=PromptTemplate(self.prompt_template),
            service_context=self.service_context,
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessor,
            response_synthesizer=response_synthesizer,
        )

        try:
            result = query_engine.query(query)
            # Obtain raw retrieved context
            retrieved_context = query_engine.retriever.retrieve(query)
            retrieved_context = [elm.node.get_content() for elm in retrieved_context]
            # Obtain re-ranked context
            reranked_context = [elm.node.get_content() for elm in result.source_nodes]
            result = result.response
        except ValueError as e:
            print(f"Cannot obtain response: {e}")
            result = "I don't know"
            retrieved_context = ['']
            reranked_context = ['']
        
        try:
            # Extract the answer from the generated text.
            answer = result.split("### Answer\n")[-1]
        except IndexError:
            # If the model fails to generate an answer, return a default response.
            answer = "I don't know"

        sys_response = SystemResponse(
            query=query,
            answer=answer,
            context={
                "vector_retriever": retrieved_context,
                "reranker": reranked_context,
            },
        )
        
        return sys_response
    
    def cleanup(self):
        # Delete index store
        if os.path.exists(self._index_dir):
            delete_directory(self._index_dir)