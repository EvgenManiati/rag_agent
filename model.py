from email import message
import os
from secrets import choice
from urllib import response
from datasets import config
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from openai import OpenAI
from dataclasses import dataclass


load_dotenv()
@dataclass

class ModelLoader:
    name: str
    provider: str
    model_id : str
    max_new_tokens: int

models = {
    "krikri": ModelLoader(
        name = "krikri",
        provider =  "huggingface",
        model_id = "ilsp/Llama-Krikri-8B-Instruct",
        max_new_tokens = 300
    ),
    "llama": ModelLoader(
        name = "llama",
        provider = "ollama",
        model_id = "llama3.2:3b",
        max_new_tokens = 300
    ),

   "qwen": ModelLoader(
        name="Qwen3 14B",
        provider="openrouter",
        model_id="qwen/qwen3-14b",
        max_new_tokens=500,
    ),

    "gpt41_mini": ModelLoader(
        name="GPT-4.1 Mini",
        provider="openrouter",
        model_id="openai/gpt-4.1-mini",
        max_new_tokens=1200
    ),

    "gemini_flash": ModelLoader(
        name="Gemini 2.5 Flash",
        provider="openrouter",
        model_id="google/gemini-2.5-flash",
        max_new_tokens=1200
    ), #meh

    "claude_haiku": ModelLoader(
        name="Claude Haiku 4.5",
        provider="openrouter",
        model_id="anthropic/claude-haiku-4.5",
        max_new_tokens=500
),

}
class OpenRouterLLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 1000,
    ):
        self.model_id = model_name
        self.max_new_tokens = max_new_tokens

        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Δεν βρέθηκε OPENROUTER_API_KEY στο .env"
            )

    def _create_client(self):
        """
        Δημιουργεί νέο OpenRouter client.

        Έτσι, αν ένας προηγούμενος HTTP client έχει κλείσει
        από το DeepEval, η επόμενη κλήση χρησιμοποιεί νέο client.
        """
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def invoke(self, prompt: str):
        client = self._create_client()

        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=self.max_new_tokens,
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    f"Το {self.model_id} δεν επέστρεψε απάντηση."
                )

            return content

        finally:
            client.close()
    
def load_hf_model(config: ModelLoader):
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    return HuggingFacePipeline(pipeline=pipe)

def load_ollama_model(config: ModelLoader):
    print(f"Φόρτωση Ollama μοντέλου: {config.name}")

    return ChatOllama(
        model=config.model_id,
        temperature=0,
        num_predict=config.max_new_tokens
    )

def load_openrouter_model(config: ModelLoader):
    print(f"Φόρτωση OpenRouter μοντέλου: {config.name}")

    return OpenRouterLLM(
        model_name=config.model_id,
        max_new_tokens=config.max_new_tokens
    )

def load_llm(model_key: str = "llama"):
    if model_key not in models:
        raise ValueError(f"Άγνωστο μοντέλο: {model_key}")

    config = models[model_key]

    if config.provider == "huggingface":
        return load_hf_model(config)

    if config.provider == "ollama":
        return load_ollama_model(config)

    if config.provider == "openrouter":
        return load_openrouter_model(config)

    raise ValueError(f"Άγνωστος provider: {config.provider}")


def list_available_models():
    return models