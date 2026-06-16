from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from config import PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MINILM_MODEL, BGE_MODEL, MINILM_WEIGHT, BGE_WEIGHT
from sentence_transformers.util import normalize_embeddings


def load_embeddings():
    """Φορτώνει και τα δύο embedding models"""
    print("Φόρτωση KRIKRI...")
    minilm = HuggingFaceEmbeddings(
        model_name=MINILM_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("Φόρτωση BGE-M3...")
    bge = HuggingFaceEmbeddings(
        model_name=BGE_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return minilm, bge


def load_retriever(mode: str = "ensemble"):
    """
    mode = "minilm"   → μόνο MiniLM
    mode = "bge"      → μόνο BGE-M3
    mode = "ensemble" → και τα δύο
    """
    embeddings_minilm, embeddings_bge = load_embeddings()

    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_docs = text_splitter.split_documents(loader.load())

    retriever_minilm = FAISS.from_documents(
        all_docs, embeddings_minilm
    ).as_retriever(search_kwargs={"k": TOP_K})

    retriever_bge = FAISS.from_documents(
        all_docs, embeddings_bge
    ).as_retriever(search_kwargs={"k": TOP_K})

    if mode == "minilm":
        return retriever_minilm
    elif mode == "bge":
        return retriever_bge
    else:
        return EnsembleRetriever(
            retrievers=[retriever_minilm, retriever_bge],
            weights=[MINILM_WEIGHT, BGE_WEIGHT]
        )


