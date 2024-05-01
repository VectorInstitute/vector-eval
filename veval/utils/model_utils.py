from typing import Any, Dict, Optional

from openai import OpenAI

from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import (
   CustomLLM,
   CompletionResponse,
   CompletionResponseGen,
   LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL = "openai-gpt-3.5-turbo"
DEFAULT_CONTEXT_WINDOW = 16385 # TODO: Verify this
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 37

LM_MODEL_CONFIG = {
   "gpt-3.5-turbo": {
      "context_window": 16385,
      },
   "gpt-4": {
      "context_window": 8192,
      },
   }


def trim_predictions_to_max_token_length(tokenizer, prediction, max_token_length=128):
   """Trims prediction output to `max_token_length` tokens"""
   tokenized_prediction = tokenizer.encode(prediction)
   trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
   trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
   return trimmed_prediction


class LlamaIndexLLM(CustomLLM):
   lm_type: str = None
   lm_name: str = None
   context_window: int = None

   lm_client: OpenAI = None # TODO: Generalize
   is_chat_model: bool = None
   temperature: float = None
   max_tokens: Optional[int] = None
   random_seed: int = None
   additional_gen_kwargs: Dict[str, Any] = None

   sys_prompt: str = None

   def __init__(
      self, 
      lm_name: str = DEFAULT_MODEL,
      temperature: float = DEFAULT_TEMPERATURE,
      seed: int = DEFAULT_SEED,
      max_tokens: Optional[int] = None,
      additional_kwargs: Optional[Dict[str, Any]] = None,
   ) -> None:
      super().__init__()
      
      _lm_type = lm_name.split("-")[0]
      if _lm_type in ["openai", "cohere"]:
         self.lm_type = _lm_type
         self.lm_name = lm_name.split(f"{_lm_type}-")[-1]
      else:
         self.lm_type = "local"
         self.lm_name = lm_name

      if self.lm_type == "openai":
         self.lm_client = OpenAI()

      # lm model config
      self.context_window = LM_MODEL_CONFIG.get(self.lm_name, {}).get("context_window", DEFAULT_CONTEXT_WINDOW)
      self.is_chat_model = True if self.lm_type == "openai" else False

      # lm generation config
      self.temperature = temperature
      self.max_tokens = max_tokens
      self.random_seed = seed
      self.additional_gen_kwargs = additional_kwargs or {}

      # TODO: Make configurable
      self.sys_prompt = DEFAULT_SYSTEM_PROMPT

   @property
   def metadata(self) -> LLMMetadata:
      if self.lm_type == "openai":
         # TODO: What about system_role?
         return LLMMetadata(
            model_name=self.lm_name,
            context_window=self.context_window,
            is_chat_model=self.is_chat_model,
         )
   
   @llm_completion_callback()
   def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
      
      if self.lm_type == "openai":
         response = self.lm_client.chat.completions.create(
            model=self.lm_name,
            messages=[
               {"role": "system", "content": self.sys_prompt},
               {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.random_seed,
            **self.additional_gen_kwargs,
         )
         response_text = response.choices[0].message.content
      else:
         raise NotImplementedError("Only OpenAI models are supported")

      return CompletionResponse(
         text=response_text
      )

   @llm_completion_callback()
   def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
      
      if self.lm_type == "openai":
         response = self.lm_client.chat.completions.create(
            model=self.lm_name,
            messages=[
               {"role": "system", "content": self.sys_prompt},
               {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.random_seed,
            **self.additional_gen_kwargs,
         )
      else:
         raise NotImplementedError("Only OpenAI models are supported")

      response_text = ""
      for chunk in response:
         chunk_text = chunk.choices[0].delta.content
         response_text += chunk_text
         return CompletionResponse(
            text=response_text,
            delta=chunk_text,
         )
