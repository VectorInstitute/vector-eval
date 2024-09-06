from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


GEN_DATA_SIZE = 50

# Load MultiHopRAG corpus
mhr_corpus_loader = HuggingFaceDatasetLoader(
    path="yixuantt/MultiHopRAG",
    name="corpus",
    page_content_column="body",
)
mhr_corpus_docs = mhr_corpus_loader.load()


# Generate synthetic Q&A data using Ragas
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
    chunk_size=1024,
)

distributions = {
    simple: 0.3,
    multi_context: 0.7,
    reasoning: 0.0,
}

mhr_qa_data = generator.generate_with_langchain_docs(
    documents=mhr_corpus_docs,
    test_size=GEN_DATA_SIZE,
    distributions=distributions,
)
# mhr_qa_data.to_pandas().to_csv(f"./data/mhr_qa_data_{GEN_DATA_SIZE}.csv")


# # Save as zip file
# import pandas as pd
# mhr_qa_data_df = pd.read_csv(f"./data/mhr_qa_data_{GEN_DATA_SIZE}.csv")
# mhr_qa_data_df.rename(columns={"Unnamed: 0": "uid"}, inplace=True)
# csv_file_name = f"./data/mhr_qa_data_{GEN_DATA_SIZE}.csv"
# zip_file_name = f"./data/mhr_qa_data_{GEN_DATA_SIZE}.zip"
# mhr_qa_data_df.to_csv(csv_file_name, index=False)
# mhr_qa_data_df.to_csv(zip_file_name, compression=dict(method="zip", archive_name=zip_file_name), index=False)
