DEFAULT_PROMPT_TEMPLATE = """
    ### Question
    {query_str}

    ### References 
    {context_str}

    ### Answer
    """


DEFAULT_SYSTEM_CONFIG = {
    "BasicRag": {
        "chunk_size": 256,
        "chunk_overlap": 0,
        "embed_model_name": "openai-text-embedding-3-small",
        "llm_name": "openai-gpt-3.5-turbo",
        "llm_gen_args": {
            "temperature": 0.8,
            "max_tokens": 128,
        },
        "similarity_top_k": 5,
        "response_mode": "compact",
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    },
    "RerankRag": {
        "chunk_size": 256,
        "chunk_overlap": 0,
        "embed_model_name": "openai-text-embedding-3-small",
        "llm_name": "openai-gpt-3.5-turbo",
        "llm_gen_args": {
            "temperature": 0.8,
            "max_tokens": 128,
        },
        "similarity_top_k": 5,
        "rerank_llm_name": "openai-gpt-3.5-turbo",
        "rerank_llm_gen_args": {
            "temperature": 0,
        },
        "rerank_top_k": 3,
        "response_mode": "compact",
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    },
}