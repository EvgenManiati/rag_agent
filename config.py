from pathlib import Path

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE_MODEL = "BAAI/bge-m3"

SEARCH_K = 5
ENSEMBLE_K = 15

MINILM_WEIGHT = 0.3
BGE_WEIGHT = 0.7

DIAVGEIA_DATASET_FILE = Path("data/diavgeia/final_dataset.jsonl")

VECTOR_STORE_DIR = Path("data/vectorstores")

MINILM_INDEX_DIR = VECTOR_STORE_DIR / "minilm_index"
BGE_INDEX_DIR = VECTOR_STORE_DIR / "bge_index"

DRIVE_BGE_INDEX_DIR = VECTOR_STORE_DIR / "bge_drive"
DRIVE_MINILM_INDEX_DIR = VECTOR_STORE_DIR / "minilm_drive"

DIAVGEIA_5000_FOLDER_ID = "1lHB8v0IV67KNP3RJh2QBkgfr2TO3wrtl"

GOOGLE_CREDENTIALS_FILE = "credentials.json"
GOOGLE_TOKEN_FILE = "token.json"

DRIVE_RECURSIVE = True