import os
from dataclasses import dataclass

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from openai import OpenAI, api_key


load_dotenv()

@dataclass
class ModelLoader:
    name: str
    provider: str
    model_id : str
    max_new_tokens: int
    no_think: bool = False

models = {

   "qwen": ModelLoader(
        name="Qwen3 14B",
        provider="openrouter",
        model_id="qwen/qwen3-14b",
        max_new_tokens=500,
        no_think=True,
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
        no_think: bool = False,
        api_key: str | None = None,
    ):
        self.model_id = model_name
        self.max_new_tokens = max_new_tokens
        self.no_think = no_think
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Δεν δόθηκε OpenRouter API key."
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
        for attempt in range(2):
            client = self._create_client()

            try:
                final_prompt = f"{prompt}\n\n/no_think" if self.no_think else prompt

                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": final_prompt,
                        }
                    ],
                    temperature=0,
                    max_tokens=self.max_new_tokens,
                )

                content = response.choices[0].message.content

                if content and content.strip():
                    return content.strip()

                if attempt == 0:
                    print(f"Κενή απάντηση από {self.model_id}. Επανάληψη...")

            finally:
                client.close()

        raise RuntimeError(
            f"Το {self.model_id} δεν επέστρεψε απάντηση μετά από 2 προσπάθειες."
        )
        
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

def load_openrouter_model(config: ModelLoader, api_key: str | None = None):
    print(f"Φόρτωση OpenRouter μοντέλου: {config.name}")

    return OpenRouterLLM(
        model_name=config.model_id,
        max_new_tokens=config.max_new_tokens,
        no_think=config.no_think,
        api_key=api_key,
    )

def load_llm(model_key: str = "qwen", api_key: str | None = None):
    if model_key not in models:
        raise ValueError(f"Άγνωστο μοντέλο: {model_key}")

    config = models[model_key]

    if config.provider == "openrouter":
        return load_openrouter_model(config, api_key=api_key)

    raise ValueError(f"Άγνωστος provider: {config.provider}")

