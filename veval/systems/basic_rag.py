import os
from typing import List

import faiss
import numpy as np
import torch

from bs4 import BeautifulSoup
from llama_index.core import (
    Document, ServiceContext, StorageContext, VectorStoreIndex, PromptTemplate, 
    get_response_synthesizer,
    )
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.faiss import FaissVectorStore
from models.utils import trim_predictions_to_max_token_length
from transformers import (
    BitsAndBytesConfig,
)

from .template import System, SystemResponse


def get_embed_model_dim(embed_model):
    embed_out = embed_model.get_text_embedding("Dummy Text")
    return len(embed_out)


class BasicRag(System):
    def __init__(self):
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
        embed_model_name='models/embedding-model/bge-small-en-v1.5'
        self.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name
        )
        
        # Load LLM
        # Specify the large language model to be used.
        model_name = "models/meta-llama/Llama-2-7b-chat-hf"
        # Configuration for model quantization to improve performance, using 4-bit precision.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
        # Load the large language model with the specified quantization configuration.
        self.llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            context_window=4096,
            max_new_tokens=75,
            model_kwargs={
                "quantization_config": bnb_config, 
                "torch_dtype": torch.float16
            },
            device_map='auto',
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
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

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
        except Exception as e:
            print(f"Cannot obtain response: {e}")
            result = ''
            retrieved_context = ['']
        
        try:
            # Extract the answer from the generated text.
            answer = result.split("### Answer\n")[-1]
        except IndexError:
            # If the model fails to generate an answer, return a default response.
            answer = "I don't know"
                                
        # Trim the prediction to a maximum of 75 tokens (this function needs to be defined).
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        sys_response = SystemResponse(
            query=query,
            answer=trimmed_answer,
            context={
                "vector_retriever": retrieved_context,
            },
        )
        
        return sys_response