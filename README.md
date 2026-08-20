
#RAG AGENT


## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) assistant for organizational knowledge management. 

The system information from organizational, regulatory and administrative documents, retrieves relevant information from the document corpus, and generates answers based on the retrieved context.

The architecture supports multiple Large Language Models, multiple retrievers and complimentary evaluation frameworks.

## Features

- Agentic RAG architecture using LangGraph
- Multiple LLM providers
  -OpenRouter
  -HuggingFace (local)
-Multiple retrieval strategies
 -MiniLM
 -BGE-M3
 -Ensemble Retriever
-Google Drive Document integration
-FAISS vector database
-Streamlit graphical interface
-Custom Evaluation Framework
-DeepEval evaluation
-Evaluation by thematic category
-Greek language document retrieval
-Diavgeia document corpus
-External PDF document integration
-Retrieval benchmarking
-JSON and CSV evaluation results



## Architecture

User Question
    ↓
LangGraph Agent
    ↓
Retriever
    ↓
FAISS Index
    ↓
Retrieved Context
    ↓
LLM Generator
    ↓
Final Answer


## Supported Models

The system currently supports:

- Llama 2.3 
- GPT-OSS 20B
- GPT-OSS 120B
- Gemini Flash
- Gemini Flash Lite
- GPT-4.1 Mini
- Claude Haiku

Depending on the model, inference can be performed locally or through an external API provider.

The available model configurations are defined in 'model.py'.
Future Integration:

- Krikri 

##Supported Retrievers

- MiniLM
   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

- BGE-M3
   - BAAI/bge-m3

- Ensemble Retrievers
    - It combines the rankings produced by MiniLM and BGE-M3.


## Document Corpus

The system uses a mixed organizational document corpus.


## Diavgeia Documents

Administrative decisions are collected from the Greek Diavgeia platform from 2021 until 2026 and stored in JSON. 
Each document contains metadata such as ADA identifier, subject, issue date, document URL, extracted text.


## External Documents
External documents do not contain Diavgeia ADA and are identified using a unique 'source_id'.
Examples include organizational regulations, funding guides and other administrative information.


## Dataset Pipeline
The document processing pipeline follows the general structure:

Source Documents
	↓
PDF Download / Import
	↓
PDF Processing
	↓
Text Extraction
	↓
Text Cleaning
	↓
Metadata Enrichment
	↓
JSONL Dataset
	↓
  Chunking
	↓
  Embeddings
	↓
FAISS Vector Stores

When the document corpus changes, the FAISS indexes must be rebuilt.



## Retrieval Evaluation

Retriever performance is evaluated independently from answer generation.

The following metrics are used:

- Hit@1
- Hit@3
- Hit@5
- Mean Reciprocal Rank (MRR)

The ground-truth retrieval dataset is divided into:

## Validation Set

Used during retriever configuration and ensemble-weight selection.


## Test set

Used for final evaluation after the retrieval configuration has been fixed.

Run the retrieval benchmark with:

''' bash
python retrieval_bench.py
'''

The evaluation supports both:
- Diavgeia  documents through 'expected_adas'
- External documents through 'expected_source_ids'


## End-to-End RAG Evaluation

The generated answers are evaluated using two complimentary approaches.


### DeepEval

DeepEval is used for semantic and LLM-based evaluation.

The main metrics include:

- Faithfulness
- Answer Relevancy
- Contextual Precision
- Contextual Recall

Unanswerable questions are additionally evaluated for refusal and hallucination behavior.

Run:

'''bash
python evaluation_deepeval.py
'''

### Custom Evaluation

A separate deterministic evaluation framework complements the semantics DeepEval evaluation.

The custom metrics are:

- Answer Exactness
- Number Accuracy
- Source Accuracy
- Source Rank

These metrics do not require an LLM judge and provide deterministic measurements of factual and retrieval correctness.

Run:

'''bash
python custom_eval.py
'''

## Evaluation Dataset

End-to-end evaluation questions are stored in:

'''text
evaluation/rag_eval_dataset.py
'''

A typical test case has the following structure:

'''python

{
  "question": "Natural-language question",
  "expected_answer": "Expected answer",
  "expected_adas": [],
  "expected_source_ids": [],
  "category": "category_name",
  "answerable": True,
}


Evaluation questions are designed to resemble natural user questions rather than artificial keyword queries.

The dataset contains multiple thematic categories, including procurement, contract, travel, scholarship, employment etc.


## Structure
rag_agent/

 -- config.py
 -- model.py (HuggingFace and OLlama model)
 -- retriever.py (PDF loading, Chunking, MiniLM and BGE-M3 embeddings)
 -- agent.py (routing, retrieval, generation)
 -- main.py (chat interface)
 -- custom_eval.py (custom evaluation metrics)
 -- evaluation_deepeval.py (DeepEval metrics)
 -- google_drive_loader.py (Google Drive integration)
 -- ui.py (Streamlit interface)
 -- retrieval_validation.py
 -- retrieval_bench.py

 -- diavgeia/
    -- crawler.py
    -- dataset_builder.py
    -- quality_check.py
    -- config.py

 -- evaluation/
    -- __init__.py
    -- rag_eval_dataset.py
    -- retrieval_eval_ground_truth.py
    -- retrieval_validation_set.py

 -- data/
    -- diavgeia/
    -- external_documents/
    -- vectorstores/
    -- evaluation_results/

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
Create a ΄.env΄ file containing the API keys.

## Usage

How to run the agent:

python main.py

The user must select an LLM and a Retriever before asking questions.

## Streamlit Interface

Run:

streamlit run ui.py


## Building the Vector Indexes

If the dataset changes, delete the existing vector stores and rebuild the indexes.

### MiniLM

'''bash
python -c "from retriever import load_retriever; load_retriever('minilm')"
'''

### BGE-M3

'''bash
python -c "from retriever import load_retriever; load_retriever('bge')"
'''

The Ensemble Retriever uses the existing MiniLM and BGE-M3 indexes.


##Evaluation Workflow

The recommended experimental workflow is:

'''text

Build Corpus
	↓
Build Vector Indexes
	↓
Retrieval Validation
	↓
Fix Retrieval Configuration
	↓
Retrieval Test Benchmark
	↓
Select Retriever
	↓
DeepEval Evaluation
	↓
Custom Deterministic Evaluation
	↓
Compare Generator Models

This seperation makes it possible to evaluate retrieval and generation quality independently.

## Evaluation Results

Evaluation outputs are stored under:

'''text
data/evaluation_results/
'''

Results are exported in JSON and CSV formats.

Detailed outputs preserve individual test-case results for further analysis.

The evaluation methodology separates retrieval effectiveness, semantic answer quality and deterministic factual correctness.

##Technologies 

- Python
- LangGraph
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers
- OpenRouter
- Streamlit
- Google Drive API
- DeepEval


