from xml.parsers.expat import model

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import DEFAULT_MODEL, LLM_MODELS



def load_model(model_name=DEFAULT_MODEL):
    model_id = LLM_MODELS.get(model_name, LLM_MODELS[DEFAULT_MODEL])

    print(f"Φόρτωση μοντέλου: {model_name} -> {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage= False
        #generation_config=model.generation_config
    )
    
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False
    model.generation_config.max_length = 2048



    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        #temperature= 0.1 
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    return HuggingFacePipeline(pipeline=pipe)

