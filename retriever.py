from functools import lru_cache
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DIAVGEIA_DATASET_FILE,
    SEARCH_K,
    ENSEMBLE_K,
    MINILM_INDEX_DIR,
    BGE_INDEX_DIR,
)
#rom google_drive_loader import load_documents_from_drive_folders


# Ρυθμίσεις embedding models


MINILM_MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

BGE_MODEL_ID = "BAAI/bge-m3"

SEARCH_K = 5
ENSEMBLE_K = 15


# Απλό ensemble retriever

class SimpleEnsembleRetriever:
    """
    Combine multiple retrievers using Weighted Reciprocal Rank Fusion.

    Each retriever contributes to the final ranking according
    to its assigned weight.

    Example:
        MiniLM -> 0.3
        BGE-M3 -> 0.7
    """

    def __init__(
        self,
        retrievers,
        weights=None,
        k=5,
        rrf_constant=60,
    ):
        self.retrievers = retrievers
        self.k = k
        self.rrf_constant = rrf_constant

        # If no weights are provided,
        # use equal weights.
        if weights is None:
            weights = [
                1.0 / len(retrievers)
                for _ in retrievers
            ]

        if len(weights) != len(retrievers):
            raise ValueError(
                "The number of weights must match "
                "the number of retrievers."
            )

        if sum(weights) <= 0:
            raise ValueError(
                "Retriever weights must have "
                "a positive total."
            )

        # Normalize weights so that they sum to 1.
        total_weight = sum(weights)

        self.weights = [
            weight / total_weight
            for weight in weights
        ]

    def invoke(
        self,
        query,
        config=None,
        **kwargs,
    ):
        """
        Retrieve documents from all retrievers and combine
        their rankings using Weighted Reciprocal Rank Fusion.
        """

        scores = {}
        documents_by_key = {}

        for retriever, weight in zip(
            self.retrievers,
            self.weights,
        ):
            documents = retriever.invoke(
                query
            )

            for rank, document in enumerate(
                documents,
                start=1,
            ):
                # Each chunk must have a unique key.
                
                document_key = str(
                document.metadata.get(
                    "ada"
                )
                or document.metadata.get(
                    "source_id"
                )
                or document.metadata.get(
                    "source"
                )
                or ""
            )    

                chunk_id = str(
                    document.metadata.get(
                        "chunk_id",
                        "",
                    )
                )

                key = (
                    document_key,
                    chunk_id,
                )

                # Weighted Reciprocal Rank Fusion
                rrf_score = (
                    weight
                    / (
                        self.rrf_constant
                        + rank
                    )
                )

                scores[key] = (
                    scores.get(
                        key,
                        0.0,
                    )
                    + rrf_score
                )

                documents_by_key[key] = (
                    document
                )

        # Sort from highest score to lowest.
        ranked_keys = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        # Return only the top-k documents.
        return [
            documents_by_key[key]
            for key in ranked_keys[:self.k]
        ]


# Φόρτωση εγγράφων από Google Drive

@lru_cache(maxsize=1)
def load_source_documents() -> tuple[Document, ...]:
    """
    Load all documents from the final corpus.

    Supports:
    - Diavgeia decisions
    - external PDF documents
    """

    if not DIAVGEIA_DATASET_FILE.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε το dataset: {DIAVGEIA_DATASET_FILE.resolve()}")

    documents = []

    diavgeia_count = 0
    external_count = 0
    skipped_count = 0

    with DIAVGEIA_DATASET_FILE.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                skipped_count += 1
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                print(f"Παράλειψη μη έγκυρου JSON στη γραμμή {line_number}: {error}")
                skipped_count += 1
                continue

            text = str(record.get("text", "")).strip()

            if not text:
                print(f"Παράλειψη γραμμής {line_number}: κενό text")
                skipped_count += 1
                continue

            source = str(record.get("source", "diavgeia")).strip()

            metadata = {
                "ada": record.get("ada"),
                "source_id": record.get("source_id"),
                "subject": record.get("subject"),
                "document_title": record.get("document_title"),
                "issue_date": record.get("issue_date"),
                "organization_id": record.get("organization_id"),
                "organization": record.get("organization"),
                "decision_type_id": record.get("decision_type_id"),
                "document_url": record.get("document_url"),
                "source": source,
            }

            subject = str(record.get("subject", "")).strip()

            if subject:
                content = f"Θέμα: {subject}\n\nΚείμενο:\n{text}"
            else:
                content = text

            document = Document(page_content=content, metadata=metadata)
            documents.append(document)

            if source == "external_pdf":
                external_count += 1
            else:
                diavgeia_count += 1

    if not documents:
        raise RuntimeError("Το dataset φορτώθηκε αλλά δεν περιείχε αξιοποιήσιμα documents.")

    print(f"Συνολικά documents: {len(documents)}")
    print(f"Diavgeia documents: {diavgeia_count}")
    print(f"External PDF documents: {external_count}")
    print(f"Skipped records: {skipped_count}")

    return tuple(documents)


# Chunking


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Document, ...]:
    """
    Split Diavgeia decisions into overlapping chunks while
    preserving the metadata of the original decision.
    """

    documents = list(
        load_source_documents()
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    # Προσθέτουμε ένα μοναδικό index ανά chunk.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(
        f"Δημιουργήθηκαν {len(chunks)} chunks "
        f"από {len(documents)} αποφάσεις."
    )

    return tuple(chunks)


# Embedding models

@lru_cache(maxsize=1)
def load_minilm_embeddings():
    """
    Load the multilingual MiniLM embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


@lru_cache(maxsize=1)
def load_bge_embeddings() -> HuggingFaceEmbeddings:
    """
    Φορτώνει το BGE-M3 μία φορά από την τοπική cache.
    """

    print("Φόρτωση BGE-M3 embeddings...")

    return HuggingFaceEmbeddings(
        model_name=BGE_MODEL_ID,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 4,
        },
    )


# FAISS vector stores

@lru_cache(maxsize=1)
def build_minilm_vectorstore():
    """
    Load an existing MiniLM FAISS index or create
    and persist it if it does not exist.
    """

    embeddings = load_minilm_embeddings()

    
    # Αν υπάρχει ήδη αποθηκευμένο index,
    # το φορτώνουμε χωρίς να ξανακάνουμε embeddings.
    

    if MINILM_INDEX_DIR.exists():

        print(
            "Φόρτωση υπάρχοντος FAISS index "
            "για MiniLM..."
        )

        return FAISS.load_local(
            folder_path=str(
                MINILM_INDEX_DIR
            ),
            embeddings=embeddings,

            # Το LangChain αποθηκεύει και metadata/docstore
            # σε pickle αρχείο.
            allow_dangerous_deserialization=True,
        )

    
    # Διαφορετικά χτίζουμε νέο index.
    

    print(
        "Δεν βρέθηκε MiniLM FAISS index."
    )

    print(
        "Δημιουργία νέου index..."
    )

    chunks = list(
        load_chunks()
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Δημιουργούμε τον parent φάκελο.
    MINILM_INDEX_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Αποθήκευση στον δίσκο.
    vectorstore.save_local(
        str(
            MINILM_INDEX_DIR
        )
    )

    print(
        "MiniLM FAISS index αποθηκεύτηκε."
    )

    return vectorstore



@lru_cache(maxsize=1)
def build_bge_vectorstore():
    """
    Load an existing BGE-M3 FAISS index or create
    and persist it if it does not exist.
    """

    embeddings = load_bge_embeddings()

    
    # Υπάρχει ήδη index;
    

    if BGE_INDEX_DIR.exists():

        print(
            "Φόρτωση υπάρχοντος FAISS index "
            "για BGE-M3..."
        )

        return FAISS.load_local(
            folder_path=str(
                BGE_INDEX_DIR
            ),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    
    # Δεν υπάρχει → χτίσιμο.
    

    print(
        "Δεν βρέθηκε BGE-M3 FAISS index."
    )

    print(
        "Δημιουργία FAISS index με BGE-M3..."
    )

    chunks = list(
        load_chunks()
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    BGE_INDEX_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(
            BGE_INDEX_DIR
        )
    )

    print(
        "BGE-M3 FAISS index αποθηκεύτηκε."
    )

    return vectorstore

# Δημιουργία retrievers

@lru_cache(maxsize=3)

def load_retriever(
    mode: str = "bge",
):
    from config import (
    MINILM_WEIGHT,
    BGE_WEIGHT,
)

    mode = mode.lower().strip()

    if mode == "minilm":

        return (
            build_minilm_vectorstore()
            .as_retriever(
                search_kwargs={
                    "k": SEARCH_K,
                }
            )
        )

    if mode == "bge":

        return (
            build_bge_vectorstore()
            .as_retriever(
                search_kwargs={
                    "k": SEARCH_K,
                }
            )
        )

    if mode == "ensemble":

        minilm_retriever = (
            build_minilm_vectorstore()
            .as_retriever(
                search_kwargs={
                    "k": SEARCH_K,
                }
            )
        )

        bge_retriever = (
            build_bge_vectorstore()
            .as_retriever(
                search_kwargs={
                    "k": SEARCH_K,
                }
            )
        )

        return SimpleEnsembleRetriever(
            retrievers=[
                minilm_retriever,
                bge_retriever,
            ],
            weights=[
            MINILM_WEIGHT,
            BGE_WEIGHT,
            ],
            k=ENSEMBLE_K,
        )

    raise ValueError(
        f"Άγνωστος retriever mode: {mode}"
    )


# Χειροκίνητη ανανέωση δεδομένων


def clear_retriever_cache() -> None:
    """
    Καθαρίζει όλο το cache.

    Χρησιμοποίησέ το όταν αλλάζουν αρχεία στο Google Drive
    και θέλεις να ξαναφορτωθούν τα PDF και να ξαναχτιστούν
    τα FAISS indexes.
    """

    load_retriever.cache_clear()

    build_minilm_vectorstore.cache_clear()
    build_bge_vectorstore.cache_clear()

    load_minilm_embeddings.cache_clear()
    load_bge_embeddings.cache_clear()

    load_chunks.cache_clear()
    load_source_documents.cache_clear()

    print("Το cache των documents και retrievers καθαρίστηκε.")
    
        


