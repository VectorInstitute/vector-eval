import os
from typing import List

import faiss

from llama_index.core import (
    Document, ServiceContext, StorageContext, VectorStoreIndex, PromptTemplate, 
    get_response_synthesizer, load_index_from_storage
    )
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore

from veval.utils.io_utils import delete_directory
from veval.utils.model_utils import LlamaIndexLLM, get_embedding_model

from .template import System, SystemResponse


def get_embed_model_dim(embed_model):
    embed_out = embed_model.get_text_embedding("Dummy Text")
    return len(embed_out)


class BasicRag(System):
    """A basic RAG system with a linear pipeline."""
    def __init__(self, llm_name: str):
        super().__init__()

        # Define chunking vars for node parser
        self.chunk_size = 256
        self.chunk_overlap = 0

        # Load node parser
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

        # Load embedding model
        embed_model_name = "openai-text-embedding-3-small" 
        # "openai-text-embedding-3-small" # "cohere-embed-english-v3.0" # "bge-small-en-v1.5"
        # TODO: Debug dimension mismatch issue for local and cohere embeddings in FAISS
        self.embed_model = get_embedding_model(embed_model_name)
        
        # Load LLM
        # Specify the large language model to be used.
        self.llm = LlamaIndexLLM(lm_name=llm_name, temperature=0, max_tokens=128)

        # Configure service context
        self.service_context = ServiceContext.from_defaults(
            node_parser=self.node_parser,
            embed_model=self.embed_model,
            llm=self.llm,
        )

        # Configure vector store
        self.faiss_dim = get_embed_model_dim(self.embed_model)
        faiss_index = faiss.IndexFlatL2(self.faiss_dim)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        self._index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".index_store")

        # Retriever vars
        self.similarity_top_k = 5

        # Generation vars
        self.response_mode = "compact"

        # Template for formatting the input to the language model, including placeholders for the question and references.
        self.prompt_template = """
        ### Question
        {query_str}

        ### References 
        {context_str}

        ### Answer
        """

    # TODO - Think of docs from a general framework perspective
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
        node_postprocessor = None
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
            retrieved_context = [elm.node.get_content() for elm in result.source_nodes]
            result = result.response
        except IndexError as e:
            print(f"Cannot obtain response: {e}")
            result = "I don't know"
            retrieved_context = ['']

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
            },
        )
        
        return sys_response
    
    def cleanup(self):
        # Delete index store
        if os.path.exists(self._index_dir):
            delete_directory(self._index_dir)