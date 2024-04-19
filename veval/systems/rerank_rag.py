from typing import List

from llama_index.core import (
    Document, VectorStoreIndex, PromptTemplate, 
    get_response_synthesizer,
    )
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI

from veval.utils.model_utils import trim_predictions_to_max_token_length

from .basic_rag import BasicRag
from .template import SystemResponse


class RerankRag(BasicRag):
    """A linear RAG system using a reranker model."""
    def __init__(self):
        super().__init__()

        self.rerank_llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125")
        self.rerank_top_k = 3 

    def invoke(self, query: str, docs: List[str]) -> SystemResponse:
        all_docs = [Document(text=doc) for doc in docs]

        # Create vector index
        vector_index = VectorStoreIndex.from_documents(
            documents=all_docs, 
            storage_context=self.storage_context,
            service_context=self.service_context,
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
            # Obtain re-ranked context
            reranked_context = [elm.node.get_content() for elm in result.source_nodes]
            result = result.response
        except Exception as e:
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
                                
        # Trim the prediction to a maximum of 128 (default) tokens.
        trimmed_answer = trim_predictions_to_max_token_length(
            tokenizer=self.tokenizer, prediction=answer
        )

        sys_response = SystemResponse(
            query=query,
            answer=trimmed_answer,
            context={
                "vector_retriever": retrieved_context,
                "reranker": reranked_context,
            },
        )
        
        return sys_response