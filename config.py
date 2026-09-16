PDF_FOLDER = "C:/Users/user/Desktop/fek"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "krikri"
LLM_MODELS= {"krikri": "ilsp/Llama-Krikri-8B-Instruct", "llama": "meta-llama/Llama-3.2-1B-Instruct"}
  #"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"} i tried to use this as well but it wasnt really succesful 
#TOP_K = 2
RETRIEVER_TOP_K = 2
BGE_MODEL = "BAAI/bge-m3"
MINILM_WEIGHT = 0.3
BGE_WEIGHT = 0.7
SEARCH_K = 5
ENSEMBLE_K = 5

# Google Drive configuration

from pathlib import Path

DIAVGEIA_DATASET_FILE = Path(
    "data/diavgeia/final_dataset.jsonl"
)

VECTOR_STORE_DIR = Path("data/vectorstores")

# Local indexes
MINILM_INDEX_DIR = VECTOR_STORE_DIR / "minilm_index"
BGE_INDEX_DIR = VECTOR_STORE_DIR / "bge_index"

# Google Drive indexes
DRIVE_BGE_INDEX_DIR = VECTOR_STORE_DIR / "bge_drive"
DRIVE_MINILM_INDEX_DIR = VECTOR_STORE_DIR / "minilm_drive"


# Google Drive folders
GOOGLE_DRIVE_ROOT_FOLDER_ID = "1lHB8v0IV67KNP3RJh2QBkgfr2TO3wrtl"
GOOGLE_DRIVE_DIAVGEIA_FOLDER_ID = "1qTU0ilkzZNKIlCh_ZEx4ffuV_F5qlHk7"
GOOGLE_DRIVE_EXTERNAL_FOLDER_ID = "1T73G502HHkA486zOEfr8vnUYat4Wb4L0"


GOOGLE_CREDENTIALS_FILE = "credentials.json"
GOOGLE_TOKEN_FILE = "token.json"

DRIVE_RECURSIVE = True