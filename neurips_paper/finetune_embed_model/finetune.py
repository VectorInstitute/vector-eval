import os
import copy
import random

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI
from llama_index.finetuning import SentenceTransformersFinetuneEngine


def train_test_split(documents, test_ratio=0.2, seed=42):
    n = len(documents)
    test_size = int(n * test_ratio)
    docs = copy.deepcopy(documents)
    random.seed(seed)
    random.shuffle(docs)
    return docs[test_size:], docs[:test_size]


def split_into_nodes(train_docs, test_docs, chunk_size=512, chunk_overlap=0):
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    train_nodes = parser.get_nodes_from_documents(train_docs)
    test_nodes = parser.get_nodes_from_documents(test_docs)
    return train_nodes, test_nodes


TEST_RUN = True

RAW_DATA_DIR = "/fs01/projects/aieng/public/vector-eval/LMOps/adaptllm/data_samples/multihoprag-raw-texts"
LLM_MODEL = "gpt-4o-mini"
BASE_MODEL_PATH = "/fs01/projects/opt_test/rag-knowledge-cutoff-embed-models"
EMBED_MODEL_PATH = f"{BASE_MODEL_PATH}/bge-large-en-v1.5"
EMBED_MODEL = EMBED_MODEL_PATH.split("/")[-1]
OUTPUT_PATH = f"{BASE_MODEL_PATH}/finetuned-models"


EXP_NAME = f"{LLM_MODEL}-{EMBED_MODEL}-finetune"
FT_DATA_DIR = f"./data/{EXP_NAME}"

if (not os.path.exists(FT_DATA_DIR)) or (len(os.listdir(FT_DATA_DIR)) == 0):
    os.makedirs(FT_DATA_DIR, exist_ok=True)

    loader = SimpleDirectoryReader(input_dir=RAW_DATA_DIR)
    documents = loader.load_data()
    if TEST_RUN:
        documents = documents[:10]

    train_docs, test_docs = train_test_split(documents, test_ratio=0.1)

    train_nodes, test_nodes = split_into_nodes(
        train_docs, test_docs, chunk_size=1024, chunk_overlap=0
    )
    print(len(train_nodes), len(test_nodes))

    train_dataset = generate_qa_embedding_pairs(
        nodes=train_nodes, llm=OpenAI(model=LLM_MODEL)
    )
    test_dataset = generate_qa_embedding_pairs(
        nodes=test_nodes, llm=OpenAI(model=LLM_MODEL)
    )
    # DEBUG: Why same datasets??

    train_dataset.save_json(f"{FT_DATA_DIR}/train.json")
    test_dataset.save_json(f"{FT_DATA_DIR}/test.json")

else:
    train_dataset = EmbeddingQAFinetuneDataset.from_json(f"{FT_DATA_DIR}/train.json")
    test_dataset = EmbeddingQAFinetuneDataset.from_json(f"{FT_DATA_DIR}/test.json")

print(f"Train dataset size: {len(train_dataset.queries)}")
print(f"Test dataset size: {len(test_dataset.queries)}")
# DEBUG: Why same datasets??


final_output_path = f"{OUTPUT_PATH}/{EXP_NAME}"
os.makedirs(final_output_path, exist_ok=True)

finetune_engine = SentenceTransformersFinetuneEngine(
    dataset=train_dataset,
    model_id=EMBED_MODEL_PATH,
    model_output_path=final_output_path,
    val_dataset=test_dataset,
)
finetune_engine.finetune()

ft_embed_model = finetune_engine.get_finetuned_model()
ft_embed_model
