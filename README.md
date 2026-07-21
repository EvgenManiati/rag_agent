
#RAG AGENT




## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) assistant for organizational knowledge management.

The system retrieves information from organizational documents (PDFs), gives responses based on retrieved context, and answers employee questions in Greek.

The architecture supports multiple LLMs, multiple retrievers and evaluation frameworks.

## Features

- Agentic RAG architecture using LangGraph
- Multiple LLM providers
  -OpenRouter
  -HuggingFace (local)
-Multiple retrieval strategies
 -MiniLM
 -BGE-M3
 -Ensemble Retriever
-Google Drive document retrieval
-FAISS vector database
-Streamlit graphical interface
-Custom evaluation framework
-DeepEval evaluation


## Architecture

User Question
    ↓
Google Drive Documents
    ↓
Document Loader
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
LangGraph Agent
    ↓
LLM
    ↓
Answer

#Supported Models

The system currently supports:
 
- GPT-OSS 20B
- GPT-OSS 120B
- Gemini Flash
- Gemini Flash Lite
- GPT-4.1 Mini
- Claude Haiku

Future integration:

- Krikri 

##Supported Retrievers

- MiniLM
- BGE-M3
- Ensemble Retriever

##Sturture

- config.py
- model.py (HuggingFace and OLlama model)
- retriever.py (PDF loading, Chunking, MiniLM and BGE-M3 embeddings)
- agent.py (routing, retrieval, generation)
- main.py (chat interface)
- custom_eval.py (custom evaluation metrics)
- evaluation_deepeval.py (DeepEval metrics)
- google_drive_loader.py (Google Drive integration)
- ui.py (Streamlit interface)
- requirements.txt
- README.md

## Installation
Create a virtual environment

΄΄΄bash
python -m venv .venv
΄΄΄

Activate it

΄΄΄bash
.venv\Scripts\activate
΄΄΄

Install the required packages

΄΄΄bash
pip install -r requirements.txt
΄΄΄

## Configuration
Create a ΄.env΄ file containing your API keys.

## Usage 

How to run the agent:

python main.py

The user must select an LLM and a Retriever and starts asking questions.

## Streamlit Interface

Run:

streamlit run ui.py

## Evaluation

### DeepEval
  How to run the evaluation:
  python evaluation_deepeval.py

### Custom Evaluation

##Technologies 

- Python
- LangGraph
- LangChain
- FAISS
- Transformers
- HuggingFace
- OpenRouter
- Streamlit
- Google Drive API
- DeepEval
