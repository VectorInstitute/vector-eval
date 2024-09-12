import tiktoken
import numpy as np
import pandas as pd

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


GEN_DATA_SIZE = 100


def get_token_stats(docs: list[str], model_name: str = "gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model_name)
    token_len_list = []
    for doc in docs:
        token_len_list.append(len(enc.encode(doc)))
    return {
        "mean": np.mean(token_len_list),
        "median": np.median(token_len_list),
        "max": max(token_len_list),
        "min": min(token_len_list),
    }


# Load MultiHopRAG corpus
mhr_corpus_loader = HuggingFaceDatasetLoader(
    path="yixuantt/MultiHopRAG",
    name="corpus",
    page_content_column="body",
)
mhr_corpus_docs = mhr_corpus_loader.load()
print(len(mhr_corpus_docs))

# # Get statitics for token count per doc
# mhr_corpus_content_txt = [doc.page_content for doc in mhr_corpus_docs]
# mhr_corpus_token_stats = get_token_stats(mhr_corpus_content_txt)
# print(mhr_corpus_token_stats)
mhr_corpus_token_stats = {'mean': 2514.193760262726, 'median': 1885.0, 'max': 17167, 'min': 1142}


# # Generate synthetic Q&A data using Ragas
# generator_llm = ChatOpenAI(model="gpt-4o-mini")
# critic_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# generator = TestsetGenerator.from_langchain(
#     generator_llm=generator_llm,
#     critic_llm=critic_llm,
#     embeddings=embeddings,
#     chunk_size=4096,
# )

# distributions = {
#     simple: 0.1,
#     multi_context: 0.7,
#     reasoning: 0.2,
# }

# mhr_qa_data = generator.generate_with_langchain_docs(
#     documents=mhr_corpus_docs,
#     test_size=GEN_DATA_SIZE,
#     distributions=distributions,
# )
# mhr_qa_data_df = mhr_qa_data.to_pandas()
# mhr_qa_data_df = mhr_qa_data_df.reset_index().rename(columns={"index": "uid"})
# mhr_qa_data_df.to_csv(f"./data/mhr_qa_data_{GEN_DATA_SIZE}.csv", index=False)


# mhr_qa_data_df.to_csv(zip_file_name, compression=dict(method="zip", archive_name=zip_file_name), index=False)
