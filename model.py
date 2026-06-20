#from xml.parsers.expat import model

#from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
#from transformers import pipeline
#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#from config import DEFAULT_MODEL, LLM_MODELS
from langchain_ollama import ChatOllama

'''
def load_model(model_name=DEFAULT_MODEL, max_new_tokens=200):
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
'''

def load_ollama_model(model_name="llama3.2:1b", max_new_tokens=120):
    print(f"Φόρτωση Ollama μοντέλου: {model_name}")

    return ChatOllama(
        model=model_name,
        temperature=0,
        num_predict=max_new_tokens
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        #temperature= 0.1 
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    return HuggingFacePipeline(pipeline=pipe)



