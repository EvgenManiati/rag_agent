# RAG Agent

An AI-powered assistant built with **LangGraph** and **Krikri** (a Greek-language Llama-based LLM) that retrieves information from organizational documents (PDFs) and answers employee queries in Greek.

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** agent using an agentic workflow. Instead of relying solely on the LLM's internal knowledge, the agent dynamically retrieves relevant context from a document corpus before generating answers — minimizing hallucinations and grounding responses in organizational policy.

The agent is designed to assist employees with procedural queries (e.g., leaves, travel expenses, HR policies) without prior training, acting as a functional co-pilot.

---

## Architecture

```
User Query
    ↓
[router] → should_retrieve?
    if YES → [retrieve] → FAISS vector search → [generate] → Answer
    if NO  → [generate] → Greeting / Farewell

```

**Key components:**
- `config.py` — paths and model parameters
- `model.py` — Krikri LLM loading with 4-bit quantization
- `retriever.py` — PDF loading, chunking, and FAISS vector store
- `agent.py` — LangGraph nodes, edges, and graph compilation
- `main.py` — chat loop entry point

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | [Krikri 8B Instruct](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct) |
| Agent Framework | LangGraph |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers (multilingual) |
| Quantization | bitsandbytes (4-bit NF4) |

---

## Installation

```bash
git clone https://github.com/EvgenManiati/rag_agent.git
cd rag_agent
pip install -r requirements.txt
```

---

## Configuration

Open `config.py` and set the path to your PDF folder:

```python
PDF_FOLDER = "path/to/your/pdfs"
```

---

## Usage

```bash
python main.py
```

The agent will load the model, index your documents, and start a chat loop:

```
Ερώτηση (ή 'exit' για έξοδο): Τι χρειάζεται για ταξιδιωτικά έξοδα;
Απάντηση: Σύμφωνα με τα έγγραφα, ο υπάλληλος πρέπει να υποβάλει αίτηση...
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (8GB+ VRAM) — CPU mode available but slow
- PDF documents placed in the configured folder

---

## Project Status

This project is part of an ongoing thesis on **Agentic RAG systems for enterprise knowledge management**. Current features are functional but experimental. Planned extensions include:

- [ ] SharePoint integration
- [ ] Multi-document source support (Excel, Word)
- [ ] Evaluation RAG Triad metrics (Faithfulness, Relevance, Context Precision)


