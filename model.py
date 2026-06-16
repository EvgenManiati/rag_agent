from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import DEFAULT_MODEL, LLM_MODELS



def load_model(model_name=DEFAULT_MODEL):
    model_id = LLM_MODELS.get(model_name, LLM_MODELS[DEFAULT_MODEL])

    print(f"Φόρτωση μοντέλου: {model_name} -> {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32
    )




    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        max_length=2048,
        temperature=0.1,
        do_sample=True,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)

