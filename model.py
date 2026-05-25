from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline
import torch
from confing import model_id




def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cpu"
    )



    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)

#llm = HuggingFacePipeline(pipeline=pipe)


