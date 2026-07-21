from functools import lru_cache
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DRIVE_RECURSIVE,
    GOOGLE_CREDENTIALS_FILE,
    GOOGLE_DRIVE_FOLDER_IDS,
    GOOGLE_TOKEN_FILE,
)
from google_drive_loader import load_documents_from_drive_folders


# Ρυθμίσεις embedding models


MINILM_MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

BGE_MODEL_ID = "BAAI/bge-m3"

SEARCH_K = 1
ENSEMBLE_K = 2


# Απλό ensemble retriever

class SimpleEnsembleRetriever:
    """
    Εκτελεί την ίδια ερώτηση σε πολλούς retrievers και ενώνει
    τα αποτελέσματα, αφαιρώντας τα διπλότυπα chunks.

    Δεν κάνει reranking. Η σειρά των αποτελεσμάτων ακολουθεί
    τη σειρά με την οποία επιστρέφονται από τους retrievers.
    """

    def __init__(
        self,
        retrievers: list[Any],
        k: int = 2,
    ):
        self.retrievers = retrievers
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        combined_documents: list[Document] = []
        seen_documents: set[tuple[str, str, int]] = set()

        for retriever in self.retrievers:
            documents = retriever.invoke(query)

            for document in documents:
                key = (
                    document.page_content.strip(),
                    str(document.metadata.get("source", "")),
                    int(document.metadata.get("page", 0)),
                )

                if key in seen_documents:
                    continue

                seen_documents.add(key)
                combined_documents.append(document)

                if len(combined_documents) >= self.k:
                    return combined_documents

        return combined_documents

    def get_relevant_documents(
        self,
        query: str,
    ) -> list[Document]:
        """
        Συμβατότητα με παλαιότερο LangChain code.
        """

        return self.invoke(query)


# Φόρτωση εγγράφων από Google Drive


@lru_cache(maxsize=1)
def load_source_documents() -> tuple[Document, ...]:
    """
    Φορτώνει τα PDF αποκλειστικά από τους δηλωμένους
    Google Drive φακέλους.

    Το αποτέλεσμα γίνεται cache, ώστε να μην ξαναδιαβάζονται
    όλα τα PDF κάθε φορά που δημιουργείται νέος retriever.
    """

    print("\nΦόρτωση εγγράφων από Google Drive...")

    documents = load_documents_from_drive_folders(
        folder_ids=GOOGLE_DRIVE_FOLDER_IDS,
        credentials_file=GOOGLE_CREDENTIALS_FILE,
        token_file=GOOGLE_TOKEN_FILE,
        recursive=DRIVE_RECURSIVE,
    )

    if not documents:
        raise RuntimeError(
            "Δεν βρέθηκε αναγνώσιμο κείμενο στα PDF "
            "των Google Drive φακέλων."
        )

    print(
        f"Φορτώθηκαν {len(documents)} σελίδες/documents."
    )

    # Χρησιμοποιούμε tuple ώστε το cached αποτέλεσμα
    # να μην τροποποιείται κατά λάθος από άλλο σημείο.
    return tuple(documents)



# Chunking


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Document, ...]:
    """
    Διασπά τα έγγραφα σε chunks μία φορά και αποθηκεύει
    το αποτέλεσμα στη μνήμη για επαναχρησιμοποίηση.
    """

    documents = list(load_source_documents())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError(
            "Δεν δημιουργήθηκαν chunks από τα έγγραφα."
        )

    print(f"Δημιουργήθηκαν {len(chunks)} chunks.")

    return tuple(chunks)


# Embedding models

@lru_cache(maxsize=1)
def load_minilm_embeddings() -> HuggingFaceEmbeddings:
    """
    Φορτώνει το multilingual MiniLM μία φορά.

    Το local_files_only=True αποτρέπει νέα σύνδεση στο
    Hugging Face, εφόσον το μοντέλο υπάρχει ήδη στην cache.
    """

    print("Φόρτωση MiniLM embeddings...")

    return HuggingFaceEmbeddings(
        model_name=MINILM_MODEL_ID,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
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
        },
    )


# FAISS vector stores

@lru_cache(maxsize=1)
def build_minilm_vectorstore() -> FAISS:
    """
    Δημιουργεί το FAISS index του MiniLM μόνο την πρώτη φορά.
    """

    print("Δημιουργία MiniLM FAISS index...")

    return FAISS.from_documents(
        documents=list(load_chunks()),
        embedding=load_minilm_embeddings(),
    )


@lru_cache(maxsize=1)
def build_bge_vectorstore() -> FAISS:
    """
    Δημιουργεί το FAISS index του BGE-M3 μόνο την πρώτη φορά.
    """

    print("Δημιουργία BGE-M3 FAISS index...")

    return FAISS.from_documents(
        documents=list(load_chunks()),
        embedding=load_bge_embeddings(),
    )


# Δημιουργία retrievers


@lru_cache(maxsize=3)
def load_retriever(mode: str = "bge"):
    """
    Επιστρέφει τον retriever που ζητήθηκε.

    Διαθέσιμες επιλογές:
    - minilm
    - bge
    - ensemble
    """

    normalized_mode = mode.strip().lower()

    if normalized_mode == "minilm":
        print("Χρήση retriever: MiniLM")

        return build_minilm_vectorstore().as_retriever(
            search_kwargs={"k": SEARCH_K}
        )

    if normalized_mode == "bge":
        print("Χρήση retriever: BGE-M3")

        return build_bge_vectorstore().as_retriever(
            search_kwargs={"k": SEARCH_K}
        )

    if normalized_mode == "ensemble":
        print("Χρήση retriever: Ensemble MiniLM + BGE-M3")

        minilm_retriever = (
            build_minilm_vectorstore().as_retriever(
                search_kwargs={"k": SEARCH_K}
            )
        )

        bge_retriever = (
            build_bge_vectorstore().as_retriever(
                search_kwargs={"k": SEARCH_K}
            )
        )

        return SimpleEnsembleRetriever(
            retrievers=[
                minilm_retriever,
                bge_retriever,
            ],
            k=ENSEMBLE_K,
        )

    raise ValueError(
        f"Άγνωστος retriever mode: {mode}. "
        "Επίλεξε minilm, bge ή ensemble."
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
    
        


