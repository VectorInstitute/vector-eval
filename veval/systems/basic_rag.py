import faiss
import os

from dataclasses import dataclass
from typing import List, Optional

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

from .template import System, SystemConfig, SystemResponse


def get_embed_model_dim(embed_model):
    embed_out = embed_model.get_text_embedding("Dummy Text")
    return len(embed_out)


@dataclass
class BasicRagConfig(SystemConfig):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embed_model_name: Optional[str] = None
    similarity_top_k: Optional[int] = None
    response_mode: Optional[str] = None


class BasicRag(System):
    """A basic RAG system with a linear pipeline."""

    _cfg = BasicRagConfig()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define chunking vars for node parser
        self.chunk_size = self._cfg.chunk_size
        self.chunk_overlap = self._cfg.chunk_overlap
        # Load node parser
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

        # Load embedding model
        # TODO: Debug dimension mismatch issue for local and cohere embeddings in FAISS
        self.embed_model = get_embedding_model(self._cfg.embed_model_name)
        
        # Load LLM - Specify the large language model to be used.
        self.llm = LlamaIndexLLM(
            lm_name=self._cfg.llm_name,
            **self._cfg.llm_gen_args,
        )

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
        self._index_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".index_store"
        )

        # Retriever vars
        self.similarity_top_k = self._cfg.similarity_top_k

        # Generation vars
        self.response_mode = self._cfg.response_mode
        # Template for formatting the input to the language model, including placeholders for the question and references.
        self.prompt_template = self._cfg.prompt_template

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