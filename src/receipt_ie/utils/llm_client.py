# """
# llm_client.py
# --------------
# Unified interface for calling different Large Language Models (LLMs)
# such as OpenAI GPT, Hugging Face Hub (Llama/Mistral), or local inference endpoints.

# This abstraction ensures the reasoning agent can use any backend
# without changing its core logic.
# """

# import os
# import time
# import json
# import logging
# from typing import Literal, Optional, Dict, Any
# from dotenv import load_dotenv
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# try:
#     import openai
# except ImportError:
#     openai = None

# try:
#     from huggingface_hub import InferenceClient
# except ImportError:
#     InferenceClient = None

# load_dotenv() 
# class LLMClient:
#     """
#     A flexible LLM interface supporting:
#     - OpenAI API models (gpt-3.5, gpt-4, etc.)
#     - Hugging Face Inference API models (Mistral, Llama-2, etc.)
#     """

#     def __init__(
#         self,
#         provider: Literal["openai", "huggingface"] = "openai",
#         model: Optional[str] = "gpt-4-turbo",
#         temperature: float = 0.2,
#         max_tokens: int = 512,
#         retry: int = 3,
#     ):
#         self.provider = provider
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.retry = retry

#         # --- Provider setup ---
#         if provider == "openai":
#             if openai is None:
#                 raise ImportError("OpenAI package not installed.")
#             openai.api_key = os.getenv("OPENAI_API_KEY")
#             self.model = model or "gpt-4-turbo"

#         elif provider == "huggingface":
#             if InferenceClient is None:
#                 raise ImportError("huggingface_hub not installed.")
#             hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#             if not hf_token:
#                 raise EnvironmentError("Missing HUGGINGFACEHUB_API_TOKEN.")
#             self.client = InferenceClient(model or "mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
#         else:
#             raise ValueError(f"Unsupported provider: {provider}")

#         logger.info(f"LLM client initialized → provider={self.provider}, model={self.model if provider=='openai' else model}")

#     # ------------------------------------------------------------------ #
#     def generate(self, prompt: str) -> str:
#         """Generate text using the configured provider."""
#         for attempt in range(self.retry):
#             try:
#                 if self.provider == "openai":
#                     return self._generate_openai(prompt)
#                 elif self.provider == "huggingface":
#                     return self._generate_hf(prompt)
#             except Exception as e:
#                 logger.warning(f"Attempt {attempt+1}/{self.retry} failed: {e}")
#                 time.sleep(2)
#         raise RuntimeError("All LLM generation attempts failed.")

#     # ------------------------------------------------------------------ #
#     def _generate_openai(self, prompt: str) -> str:
#         """Generate response from OpenAI chat models."""
#         response = openai.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#         )
#         msg = response.choices[0].message.content.strip()
#         logger.debug(f"OpenAI response: {msg[:200]}...")
#         return msg

#     # ------------------------------------------------------------------ #
#     def _generate_hf(self, prompt: str) -> str:
#         """Generate response from Hugging Face Inference API models."""
#         output = self.client.text_generation(
#             prompt,
#             max_new_tokens=self.max_tokens,
#             temperature=self.temperature,
#             do_sample=True,
#         )
#         if isinstance(output, str):
#             msg = output.strip()
#         elif isinstance(output, list):
#             msg = output[0]["generated_text"]
#         else:
#             msg = str(output)
#         logger.debug(f"HuggingFace response: {msg[:200]}...")
#         return msg

"""
llm_client.py
--------------
Unified interface for calling different Large Language Models (LLMs):
- OpenAI GPT models (gpt-3.5, gpt-4, etc.)
- Hugging Face Hub (Llama/Mistral, etc.)
- Groq API (free, ultra-fast inference)
"""

import os
import time
import json
import logging
from typing import Literal, Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Load API keys ---
load_dotenv()

# --- Optional imports for providers ---
try:
    import openai
except ImportError:
    openai = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    from groq import Groq
except ImportError:
    Groq = None


class LLMClient:
    """
    A flexible LLM interface supporting:
    - OpenAI API models (gpt-3.5, gpt-4, etc.)
    - Hugging Face Inference API (Mistral, Llama-2)
    - Groq API (Gemma, Llama-3, Mixtral)
    """

    def __init__(
        self,
        provider: Literal["openai", "huggingface", "groq"] = "groq",
        model: Optional[str] = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: int = 512,
        retry: int = 3,
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry = retry

        # --- Provider setup ---
        if provider == "openai":
            if openai is None:
                raise ImportError("❌ OpenAI package not installed.")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4-turbo"

        elif provider == "huggingface":
            if InferenceClient is None:
                raise ImportError("❌ huggingface_hub not installed.")
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                raise EnvironmentError("❌ Missing HUGGINGFACEHUB_API_TOKEN.")
            self.model = model or "mistralai/Mistral-7B-Instruct-v0.2"
            self.client = InferenceClient(self.model, token=hf_token)

        elif provider == "groq":
            if Groq is None:
                raise ImportError("❌ groq package not installed. Run `pip install groq`.")
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise EnvironmentError("❌ Missing GROQ_API_KEY.")
            self.client = Groq(api_key=groq_key)
            self.model = model or "llama3-70b-8192"

        else:
            raise ValueError(f"❌ Unsupported provider: {provider}")

        logger.info(f"✅ LLM initialized → provider={self.provider}, model={self.model}")

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        """Generate text using the configured provider."""
        for attempt in range(self.retry):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt)
                elif self.provider == "huggingface":
                    return self._generate_hf(prompt)
                elif self.provider == "groq":
                    return self._generate_groq(prompt)
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt+1}/{self.retry} failed: {e}")
                time.sleep(2)
        raise RuntimeError("❌ All LLM generation attempts failed.")

    # ------------------------------------------------------------------ #
    def _generate_openai(self, prompt: str) -> str:
        """Generate from OpenAI GPT models."""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response: {msg[:200]}...")
        return msg

    # ------------------------------------------------------------------ #
    def _generate_hf(self, prompt: str) -> str:
        """Generate from Hugging Face Inference API."""
        output = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
        )
        if isinstance(output, str):
            msg = output.strip()
        elif isinstance(output, list):
            msg = output[0]["generated_text"]
        else:
            msg = str(output)
        logger.debug(f"HuggingFace response: {msg[:200]}...")
        return msg

    # ------------------------------------------------------------------ #
    def _generate_groq(self, prompt: str) -> str:
        """Generate from Groq’s ChatGPT-compatible API."""
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = chat_completion.choices[0].message.content.strip()
        logger.debug(f"Groq response: {msg[:200]}...")
        return msg
