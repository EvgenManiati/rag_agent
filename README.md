
#RAG AGENT




## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) assistant for organizational knowledge management.

The system retrieves information from organizational documents (PDFs), gives responses based on retrieved context, and answers employee questions in Greek.

The architecture supports multiple local LLMs, multiple retrievers, and gathers information from a local folder. 

## Architecture

User Query
    ↓
[Router]
    ↓
[Retriever]
    ↓
FAISS Vector Store
    ↓
[LLM Generator]
    ↓
Answer

#Supported Models

The system currently supports:
 
-Qwen 2.5 1.5B Instruct
-Llama 3.2 3B (via Ollama)

Future integration:

-Krikri 

##Retrieval Models

The system supports embedding models:

-MiniLM
-BGE-M3

##Sturture

-config.py
- model.py (HuggingFace and OLlama model)
- retriever (PDF loading, Chunking, MiniLM and BGE-M3 embeddings)
- agent.py (routing, retrieval, generation)
- main.py (chat interface)
- evaluation.py (RAGAS Triad evaluation metrics)
- evaluation_deepeval.py (DeepEval metrics)


## Usage 

How to run the agent:

python main.py

Choose a model:
Type 1 to choose Qwen, 2 to choose Krikri and 3 to choose Llama 3.2 (via OLlama)

## Evaluation

Future integration of RAGAS or DeepEval
