PDF_FOLDER = "C:/Users/user/Desktop/fek"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "krikri"
LLM_MODELS= {"krikri": "ilsp/Llama-Krikri-8B-Instruct", "llama": "meta-llama/Llama-3.2-1B-Instruct"}
  #"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"} i tried to use this as well but it wasnt really succesful 
#TOP_K = 2
RETRIEVER_TOP_K = 2
BGE_MODEL = "BAAI/bge-m3"
MINILM_WEIGHT = 0.5
BGE_WEIGHT = 0.5

# Google Drive configuration

GOOGLE_DRIVE_FOLDER_IDS = [
    "1wsj-Hr8FlVk2CcVqbUS019tiFl9vRJBs",
    "1uj-rTyFzAwAXT737AX2HqaGqlbfprUyu",
    "146gzLrJB25Ndke7u8pFfo3Fef_idhzPV",
    "1c-zBlp-rTLO-tRnpfBrdCe00QyIOh_4R",
]

GOOGLE_CREDENTIALS_FILE = "credentials.json"
GOOGLE_TOKEN_FILE = "token.json"

# Αν είναι True, διαβάζονται και υποφάκελοι.
DRIVE_RECURSIVE = True

