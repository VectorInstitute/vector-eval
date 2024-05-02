import cohere
import subprocess
import time

from functools import wraps
from typing import Any, Dict, Optional, Callable

from llama_index.core.llms import (
   CustomLLM,
   CompletionResponse,
   CompletionResponseGen,
   LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
import os
import os


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL = "openai-gpt-3.5-turbo"
DEFAULT_CONTEXT_WINDOW = 16385 # TODO: Verify this
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 37
DEFAULT_RL_LIMIT = 50
DEFAULT_RL_INTERVAL = 60.0

LM_MODEL_CONFIG = {
   "gpt-3.5-turbo": {
      "context_window": 16385,
   },
   "gpt-4": {
      "context_window": 8192,
   },
   "command": {
      "context_window": 4096,
   },
   "llama2-7b": {
      "context_window": 4096,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 1,
   },
   "llama2-7b-chat": {
      "context_window": 4096,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 1,
   },
   "llama2-13b": {
      "context_window": 4096,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 2,
   },
   "llama2-13b-chat": {
      "context_window": 4096,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 2,
   },
   "llama2-70b": {
      "context_window": 4096,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 4,
   },
   "llama2-70b-chat": {
      "context_window": 4096,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 4,
   },
   "llama3-8b": {
      "context_window": 8192,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 1,
   },
   "llama3-8b-instruct": {
      "context_window": 8192,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 1,
   },
   "llama3-70b": {
      "context_window": 8192,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 4,
   },
   "llama3-70b-instruct": {
      "context_window": 8192,
      # launch config
      "gpu_partition": "a40",
      "num_gpus": 4,
   },
   # "mixtral-7b": {
   #    "context_window": 32768,
   #    "moe": False,
   #    # launch config
   #    "gpu_partition": "a40",
   #    "num_nodes": 1,
   #    "num_gpus": 1,
   # },
   # "mixtral-7b-instruct": {
   #    "context_window": 32768,
   #    "moe": False,
   #    # launch config
   #    "gpu_partition": "a40",
   #    "num_nodes": 1,
   #    "num_gpus": 1,
   # },
   "mixtral-8x7b-instruct": {
      "context_window": 32768,
      "moe": True,
      # launch config
      "gpu_partition": "a40",
      "num_nodes": 1,
      "num_gpus": 4,
   },
   "mixtral-8x22b": {
      "context_window": 32768,
      "moe": True,
      # launch config
      "gpu_partition": "a40",
      "num_nodes": 2,
      "num_gpus": 4,
   },
   "mixtral-8x22b-instruct": {
      "context_window": 32768,
      "moe": True,
      # launch config
      "gpu_partition": "a40",
      "num_nodes": 2,
      "num_gpus": 4,
   },
   }


def rate_limiter(limit: int, interval: float) -> Callable:
   def decorator(func: Callable) -> Callable:
      calls = 0
      last_reset = time.time()

      @wraps(func)
      def wrapper(*args, **kwargs):
         nonlocal calls, last_reset
         calls += 1
         if calls > limit:
            elapsed_time = time.time() - last_reset
            if elapsed_time < interval:
               time.sleep(interval - elapsed_time)
               last_reset = time.time()
               calls = 1
            else:
               last_reset = time.time()
               calls = 1
         return func(*args, **kwargs)

      return wrapper

   return decorator


def trim_predictions_to_max_token_length(tokenizer, prediction, max_token_length=128):
   """Trims prediction output to `max_token_length` tokens"""
   tokenized_prediction = tokenizer.encode(prediction)
   trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
   trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
   return trimmed_prediction


class LlamaIndexLLM(CustomLLM):
   lm_type: str = None
   lm_name: str = None
   lm_alias: str = None
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

      elif self.lm_type == "cohere":
         self.lm_client = cohere.Client()

      elif self.lm_type == "local":
         if not os.path.exists("veval/utils/vector-inference"):
            error_msg = "Directory not found: veval/utils/vector-inference." + \
               " Ensure you are on the Vector cluster to use local models and" + \
               " clone the `vector-inference` repo under `veval/utils`."
            raise FileNotFoundError(error_msg)

         lm_family = self.lm_name.split("-")[0]
         lm_variant = self.lm_name.split(f"{lm_family}-")[-1]

         if lm_family == "llama2":
            self.lm_alias = f"Llama-2-{lm_variant}"
         elif lm_family == "llama3":
            if "instruct" in lm_variant:
               lm_variant = lm_variant.split("-")
               lm_variant = "-".join([lm_variant[0].upper(), lm_variant[-1].capitalize()])
            else:
               lm_variant = lm_variant.upper()
            self.lm_alias = f"Meta-Llama-3-{lm_variant}"
         elif lm_family == "mixtral":
            if "instruct" in lm_variant:
               lm_variant = lm_variant.split("-")
               lm_variant = "-".join([lm_variant[0].upper(), lm_variant[-1].capitalize()])
            else:
               lm_variant = lm_variant.upper()
            lm_variant += "-v0.1"
            lm_alias_head = "Mixtral" if LM_MODEL_CONFIG.get(self.lm_name, {}).get("moe", False) else "Mistral"
            self.lm_alias = f"{lm_alias_head}-{lm_variant}"
         else:
            raise ValueError(f"Local model family {lm_family} not supported.")

         url_file = f"veval/utils/vector-inference/models/{lm_family}/.vLLM_{self.lm_alias}_url"
         # TODO: Use different signal for checking whether server is up, instead of presence of the url file
         if not os.path.exists(url_file):
            # Start a new instance if not already running
            gpu_partition = LM_MODEL_CONFIG.get(self.lm_name, {}).get("gpu_partition", "a40")
            num_gpus = LM_MODEL_CONFIG.get(self.lm_name, {}).get("num_gpus", 1)
            # qos = LM_MODEL_CONFIG.get(self.lm_name, {}).get("qos", "m3")
            print(f"Starting local {lm_name} server...")
            launch_cmd = f"bash veval/utils/vector-inference/models/{lm_family}/launch_server.sh" + \
               f" -v {lm_variant}" + \
               f" -p {gpu_partition}" + \
               f" -n {num_gpus}"
               # f" -q {qos}"
            if lm_family == "mixtral":
               num_nodes = LM_MODEL_CONFIG.get(self.lm_name, {}).get("num_nodes", 1)
               launch_cmd += f" -N {num_nodes}"
            try:
               result = subprocess.run(
                  launch_cmd,
                  shell=True,
                  capture_output=True,
                  text=True,
               )
               print(result.stdout)
               print(result.stderr)
            except Exception as e:
               print(f"Failed to launch local {lm_name} server: {e}")

            while not os.path.exists(url_file):
               time.sleep(5)

         # Fetch endpoint url from file, keep trying until its populated
         local_endpoint = ""
         while local_endpoint == "":
            with open(url_file, "r") as f:
               local_endpoint = f.read().strip()
            time.sleep(1)

         self.lm_client = OpenAI(base_url=local_endpoint, api_key="LOCAL")

      # lm model config
      self.context_window = LM_MODEL_CONFIG.get(self.lm_name, {}).get("context_window", DEFAULT_CONTEXT_WINDOW)
      self.is_chat_model = True if self.lm_type in ["openai", "cohere"] else False

      # lm generation config
      self.temperature = temperature
      self.max_tokens = max_tokens
      self.random_seed = seed
      self.additional_gen_kwargs = additional_kwargs or {}

      # TODO: Make configurable
      self.sys_prompt = DEFAULT_SYSTEM_PROMPT

   @property
   def metadata(self) -> LLMMetadata:
      # TODO: What about system_role?
      return LLMMetadata(
         model_name=self.lm_name,
         context_window=self.context_window,
         is_chat_model=self.is_chat_model,
      )
   
   @rate_limiter(
      limit=DEFAULT_RL_LIMIT, 
      interval=DEFAULT_RL_INTERVAL,
   )
   @llm_completion_callback()
   def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
      
      if self.lm_type in ["openai", "local"]:
         model_name = self.lm_name
         if self.lm_type == "local":
            model_name = f"/model-weights/{self.lm_alias}"
            if "Llama-2" in self.lm_alias:
               model_name += "-hf"
         response = self.lm_client.chat.completions.create(
            model=model_name,
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
      elif self.lm_type == "cohere":
         # TODO: Use preamble?
         response = self.lm_client.chat(
            model=self.lm_name,
            chat_history=[],
            message=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.random_seed,
            **self.additional_gen_kwargs,
         )
         response_text = response.text
      else:
         raise NotImplementedError(f"Only supports `openai` and `cohere` LLMs. `{self.lm_type}` is not supported.")

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
      elif self.lm_type == "cohere":
         # TODO: Use preamble?
         response = self.lm_client.chat_stream(
            model=self.lm_name,
            chat_history=[],
            message=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.random_seed,
            **self.additional_gen_kwargs,
         )
      else:
         raise NotImplementedError(f"Only supports `openai` and `cohere` LLMs. `{self.lm_type}` is not supported.")

      response_text = ""
      for chunk in response:
         
         if self.lm_type == "openai":
            chunk_text = chunk.choices[0].delta.content
         elif self.lm_type == "cohere":
            if chunk.event_type == "text-generation":
               chunk_text = chunk.text
            elif chunk.event_type == "stream-end":
               chunk_text = None
         
         if chunk_text is not None:
            response_text += chunk_text
            yield CompletionResponse(
               text=response_text,
               delta=chunk_text,
            )
