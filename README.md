
#RAG AGENT


## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) assistant for organizational knowledge management. 

The system information from organizational, regulatory and administrative documents, retrieves relevant information from the document stored in Google Drive, and generates answers based on the retrieved context.

The architecture combines LangGraph-based agent orchestration, semantic and ensemble retrieval over a Google Drive document corpus, multiple Large Language Models (LLMs) and a comprehensive evaluation framework for retrieval and answer quality.

The system was developed and evaluated primarily on Greek-language administrative documents.

## Features

- Agentic RAG architecture using LangGraph
- Google Drive integration through OAuth 2.0
- Recursive document loading from multiple Google Drive folders
- PDF processing and metadata preservation
- Multiple LLM providers
  -OpenRouter
  -HuggingFace (local)
-Multiple retrieval strategies
 -MiniLM
 -BGE-M3
 -Ensemble Retriever
- Weighted Reciproval Rank Fusion for ensemble retrieval 
-FAISS vector indexes
-Streamlit chat interface
-Diavgeia administrative document corpus
-External PDF document integration
-Custom deterministic evaluation
-DeepEval semantic evaluation
-Evaluation by thematic category
-Retrieval benchmarking with Hit@K and MRR 
- File and folder-level retrieval evaluation
-JSON and CSV evaluation results



## Architecture

OFFLINE / INDEX BUILDING

Google Drive
    ↓
Documents
    ↓
Chunking
    ↓
MiniLM Index + BGE-M3 Index
    ↓
Local FAISS Vector Stores


ONLINE / QUERY PIPELINE

User Question
    ↓
LangGraph Agent
    ↓
MiniLM + BGE-M3 Retrieval
    ↓
Weighted RRF
    ↓
Retrieved Context
    ↓
LLM 
    ↓
Final Answer


## Supported Models

The system currently supports multiple local and API-based LLMs, allowing the same RAG pipeline to be tested and evaluated with five different generator models:

- Llama 2.3 - Ollama
- Gemini Flash - OpenRouter
- Gemini Flash Lite - OpenRouter
- GPT-4.1 Mini - OpenRouter
- Claude Haiku - OpenRouter

Depending on the model, inference can be performed locally or through an external API provider.

The available model configurations are defined in `model.py`.

Future Integration:

- Krikri - HuggingFace



##Supported Retrievers

The system supports three retrieval strategies over the Google Drive document corpus.


- MiniLM
   - Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

   - Retriever mode: `drive_minilm`

- BGE-M3
   - Embedding model:  BAAI/bge-m3
   
   - Retriever mode: `drive_bge`

- Ensemble Retrievers
    - Retriever mode: `drive_ensemble`
    - It combines the rankings produced by MiniLM and BGE-M3 using Weighted Reciprocal Rank Fusion (RRF). 

The contribution of each retriever is controlled through configurable weights. 

The Ensemble Retriever achieved the best overall retrieval score thus it was selected for the final Drive-based RAG evaluation.


## Document Corpus

The system uses a Greek mixed organizational document corpus stored in Google Drive.

The corpus contains two main document sources: 

- Diavgeia administrative decisions
- External organizational documents

The documents are processed into chunks and indexed locally in FAISS vector stores while preserving metadata that allows the system to identify and trace the retrieved sources.

## Diavgeia Documents

Administrative decisions are collected from the Greek Diavgeia platform from 2021 until 2026.
The collection and preprocessing pipeline is implemented under `diavgeia/` and includes:

- API-based metadata collection
- PDF downloading, validation and repair 
- text extraction 
- OCR fallback for problematic documents
- text quality checks
- metadata preservation


Each document contains metadata such as ADA identifier, document title, issue date and local filename.

During dataset preparation, the processed Diavgeia records can be exported to JSONL format for validation and quality control. 

For the final Drive-based RAG system, the curated PDF documents are organized in Google Drive and loaded from there for the vector indexing. 

## External Documents

This corpus also includes organizational documents that are not part of the Diavgeia platform.

External documents do not contain Diavgeia ADA identifiers and they are tracked using metadata such as their filename and Google Drive location.

Examples include organizational regulations, funding guides and other administrative information.

External documents are stored alongside the Diavgeia corpus in Google Drive and are processed by the same retrieval pipeline. 


## Dataset Pipeline

The document corpus is created through a multi-stage preprocessing pipeline:

### 1. Document Collection

`crawler.py`

Collects Diavgeia metadata through the public API and downloads the matching PDF documents.

### 2. Dataset Constraction

`dataset_builder.py`


The documents are collected as follows:

- extracting text from PDFs
- validating document quality
- repairing problematic PDF files
- preserving document metadata
- producing structured dataset records

### 3. OCR Repair

`repair_suspicious.py`

When standard text extraction fails, the problematic or scanned PDF documents are curated using OCR. 

### 4. Dataset Merge

`merge_repaired.py`

Merges successfully repaired documents back into the main dataset.

### 5. Quality Validation 

`quality_check.py`

Quality checks are performed on the processed documents.

### 6. Curated Dataset

The preprocessing pipeline produces a well organized document collection that can be exported in JSONL format.

### 7. Google Drive Corpus

The final compilation of PDF documents are organized and stored in Google Drive.

The Drive corpus contains both: 

-Diavgeia administrative decisions
- External organizational decisions

`google_drive_loader.py` recursively loads the documents and preserve their source metadata.

### 8. Vector Indexing

The document loaded from Google Drive are split into chunks and embedded using the supported embedding models.

The embeddings are stored in local FAISS vector indexes:

- MiniLM index
- BGE-M3 index

The Ensemble Retriever combines both the rankings of the two indexes rather than creating a separate vector index.


```text
Diavgeia API
	↓
Metadata + PDF collection
	↓
Text Extraction / PDF Repair
	↓
OCR Fallback
	↓
Quality Validation
	↓
Repaired Document Corpus
	↓
Google Drive
	↓
Document loading
	↓
  Chunking
	↓
MiniLM + BGE-M3 Embeddings
	↓
FAISS Vector Stores
	↓
Weighted RRF and Ensemble Retrieval



## Retrieval Evaluation

Retrievers' performances are evaluated independently from answer generation, allowing retrieval quality to be measured without the results being affected by the behavior of a specific language model.

The evaluation uses manually defined ground-truth queries with expected source documents.

### Retrieval Metrics

- Hit@1 - whether the correct source appears at rank 1
- Hit@3 - whether the correct source appears within the top 3 results
- Hit@5 - whether the correct source appears within the top 5 results
- Mean Reciprocal Rank (MRR) - measures how highly the First correct result is ranked

In addition to document/ source matching, the evaluation also includes:

- File Hit@K / File MRR - evaluates retrieval using the expected filenames
- Folder Hit@K / Folder MRR - evaluates whether a result from the expected Google Drive folder is retrieved

These additional metrics are useful for documents that may be identified through different metadata fields.


### Validation and Test Sets

The validation set was used during retriever configuration and development.


The test set was used for final evaluation after the retrieval configuration has been fixed.

The retrieval ground truth is defined in: 

`evaluation/retrieval_eval_ground_truth.py`

### Retrieval Benchmark

The final benchmark compares three retrieval strategies over the same Google Drive document corpus:

| Retriever     | Hit@1 | Hit@3 | Hit@5 | MRR   |
| Drive MiniLM  | 0.500 | 0.708 | 0.750 | 0.608 |
| Drive Ensemble| 0.875 | 1.000 | 1.000 | 0.924 |
| Drive BGE-M3  | 0.833 | 1.000 | 1.000 | 0.903 |

As the table shows the Ensemble Retriever achieved the best overall retrieval performance with the highest Hit@1 and MRR while maintaining perfect Hit@3 and Hit@5.

Based on the above, `drive_ensemble` was selected as the retriever for the final end-to-end RAG evaluation.

Run the retrieval benchmark with:

```bash
python retrieval_bench.py
```

The benchmark results are stored in:

`data/evaluation/retrieval_benchmark.json`


For the selected Drive Ensemble configuration, folder-level retrieval performed a perfect Folder Hit@1 of 1.000, indicating that the correct Google Drive folder was identified at the first rank for all evaluated test queries.


## End-to-End RAG Evaluation

After selecting the retrieval configuration, the complete RAG pipeline is evaluated across multiple generator models.


The final end-to-end experiment use the `drive_ensemble` retriever for all models, keeping the retrieval component fixed so that differences in the results primarily reflect the behavior of the generator models.

Two complimentary evaluation approaches are used:

1. DeepEval - semantic and LLM-as-a-judge evaluation
2. Custom Evaluation - deterministic evaluation of answer and source correctness

All the generator models referred above were evaluated. 


### DeepEval

DeepEval is used for semantic and LLM-based evaluation.

For answerable questions, the following metrics are used:

- Faithfulness - whether the generated answer is supported by the retrieved context
- Answer Relevancy - whether the generated answer directly addresses the question
- Contextual Precision - whether the retrieved context contains relevant information without excessive irrelevant content 
- Contextual Recall - whether the retrieved context contains the information required to answer the question

Unanswerable questions are additionally evaluated for Refusal Accuracy and Hallucination Rate.
The Refusal Accuracy indicates whether the model correctly refuses to provide an unsupported answer, while the Hallucination Rate shows how often the models provide an answer even if the available context does not support one.

Run the evaluation with:

```bash
python evaluation_deepeval.py
```

## Answerable Results

```markdown

#### Answerable Results — Drive Ensemble

| Model            | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| Llama 3.2        | 0.744        | 0.956 	     | 0.900             | 0.947          |
| Qwen3 14B        | 0.929        | 0.939            | 0.947             | 0.895          |
| GPT-4.1 Mini     | 0.897        | 0.786            | 0.947             | 0.947          |
| Gemini 2.5 Flash | 1.000        | 0.961            | 0.947             | 0.947          |
| Claude Haiku 4.5 | 0.959        | 0.836            | 0.947             | 0.947          |

Gemini 2.5 Flash achieved the strongest overall semantic evaluation results, reaching perfect Faithfulness and the highest Answer Relevancy while maintaining high Contextual Presicion and Contextual Recall.

#### Unanswerable Results — Drive Ensemble

| Model            | Faithfulness | Answer Relevancy | Refusal Accuracy |Hallucination Rate
| Llama 3.2        | 1.000        | 0.667            | 0.125            | 0.875 
| Qwen3 14B        | 1.000        | 0.875            | 1.000            | 0.000 
| GPT-4.1 Mini     | 1.000        | 0.750            | 1.000            | 0.000 
| Gemini 2.5 Flash | 1.000        | **1.000**        | 1.000            | 0.000 
| Claude Haiku 4.5 | 1.000        | 0.708            | 1.000            | 0.000 

Qwen3 14B, GPT-4.1 Mini, Gemini 2.5 Flash and Claude Haiku 4.5 correctly refused all evaluated unanswerable questions without hallucinating unsupported answers. Llama 3.2 showed substantially weaker refusal behavior, with a Refusal Accuracy of 0.125 and a Hallucination Ratw of 0.875.


### Custom Evaluation

A separate deterministic evaluation framework complements the semantics DeepEval evaluation measuring factual properties that can be checked directly without an LLM judge.

The custom metrics are:

- Answer Exactness - strict normalized textual match between the generated and expected answer
- Number Accuracy - whether the expected numerical values are present in the generated answer
- Source Accuracy - whether the expected source appears among the retrieved documents
- Source Rank - the ranking position of the first correct retrieved source

Answer Exactness is intentionally strict. Semanticly equivalent answers with different wording may therefore receive a score of '0'.


Run the Custom evaluation with:

```bash
python custom_eval.py
```


#### Custom Evaluation Results — Drive Ensemble

| Model            | Answer Exactness | Number Accuracy | Source Accuracy | Source Rank |
| Llama 3.2        | 0.000            | 0.077           | 0.895           | 1.235       |
| Qwen3 14B        | 0.211            | 0.846           | 0.895           | 1.235       |
| GPT-4.1 Mini     | 0.000            | 0.846           | 0.895           | 1.235       |
| Gemini 2.5 Flash | 0.158            | 0.769           | 0.895           | 1.235       |
| Claude Haiku 4.5 | 0.053            | 0.846           | 0.895           | 1.235       |


For Source Rank, lower values indicate better retrieval performance, as the correct source appears earlier in the ranked results.

Qwen3 14B achieved the highest Answer Exactness and Number Accuracy. High Number Accuracy also was achieved by GPT-4.1 Mini and Claude Haiku 4.5.

Source Accuracy and Source Rank are identical across all generator models because same fixed `drive_ensemble` retriever was used in every experiment.



## Evaluation Dataset

The end-to-end evaluation uses a manually constructed dataset designed to represent realistic user requests over the document corpus.

The evaluation dataset is defined in:


`evaluation/rag_eval_dataset.py`

A typical test case has the following structure:


```python
{
  "question": "Natural-language question",
  "expected_answer": "Expected answer",
  "expected_adas": ["expected_ada"],
  "expected_source_ids": [],
  "expected_file_names" : [],
  "category": "procurement",
  "answerable": True
}
```

Depending on the document type, the expected source can be identified through:

- ADA - for Diavgeia administrative decisions
- Source ID - for external organizational documents
- Filename - when file-level identification is required

The dataset contains both Answerable and Unanswerable questions. For the first the required information exists in the document corpus, while for the last the answer is not provided by the corpus and the model is expected to refuse rather to answer than generate an unsupported answer.

Evaluation questions are designed to resemble natural user questions rather than artificial keyword queries.

The dataset contains multiple thematic categories including procurement, contract and contract modification, travel, scholarship, employment, Research ethics and leave. 

The evaluation dataset is intentionally kept separate from the retrieval benchmark dataset. Retrieval evaluation measures the retriever independently, whereas the end-to-end dataset
evaluates the complete RAG pipeline. 


## Project Structure

```text
rag_agent/

 -- config.py
 -- model.py
 -- retriever.py 
 -- agent.py 
 -- main.py 
 -- google_drive_loader.py 
 -- ui.py 

 -- add_external_docs.py
 -- prepare_drive_dataset.py
 -- upload_dataset_to_drive.py

 -- retrieval_error_analysis.py
 -- retrieval_bench.py
 -- custom_eval.py 
 -- evaluation_deepeval.py 
 -- extract_unanswerable_results.py

 -- diavgeia/
    -- __init__.py
    -- config.py
    -- crawler.py
    -- dataset_builder.py
    -- quality_check.py
    -- merge_repaired.py
    -- repair_suspicious.py

 -- evaluation/
    -- __init__.py
    -- rag_eval_dataset.py
    -- retrieval_eval_ground_truth.py
    -- retrieval_validation_set.py

 -- data/
    -- evaluation/
       -- retrieval_benchmark.json
    -- evaluation_results/
       -- local_custom_eval_results.csv
       -- local_custom_eval_results.json
       -- local_custom_eval_detailed_results.csv
  
       -- custom_eval_drive_results.csv
       -- custom_eval_drive_results.json
       -- custom_eval_drive_detailed_results.csv

       -- custom_eval_drive_ensemble_results.csv
       -- custom_eval_drive_ensemble_results.json
       -- custom_eval_drive_ensemble_detailed_results.csv
       
       -- deepeval_results_drive.csv
       -- deepeval_results_drive.json
       -- deepeval_detailed_results_drive.csv

       -- deepeval_results_drive_ensemble.csv
       -- deepeval_results_drive_ensemble.json
       -- deepeval_detailed_results_drive_ensemble.csv

- requirements.txt
- README.md
- .gitignore

```

The repository is organized into four main components:
 - Core RAG system - agent orchestration, model loading, retrieval, Google Drive integration and user interface
 - Dataset preparation - Diavgeia collection, PDF processing, repair, OCR and Drive preparation
 - Evaluation framework - retrieval benchmarking, DeepEval evaluation, custom deterministic evaluation and error analysis
 - Evaluation results - preserved experimental outputs for the evaluated system configurations

Generated FAISS vector stores are stored locally under `data/vectorstores/` and are excluded from version control. 


## Installation

### Requirements

- Python 3.11 or newer is recommended
- Git
- Tesseract OCR (required only for OCR-based PDF repair)
- Ollama (required only when using locally hosted Ollama models)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag_agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (Optional)

Tesseract OCR is used by the document repair pipeline when text cannot be extracted normally from problematic or scanned PDF files.

The Python package `pytesseract` does not include the Tesseract executable itself. Tesseract must therefore be installed separately on the operating system if the OCR repair functionality is required.

The main RAG application does not require Tesseract when working with an already prepared document corpus.


## Configuration 

### API Keys

Create a `.env` file in the Project root and add the credentials required by the selected model providers.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

Only the credentials required by the selected models need to be configured.


### Google Drive Authentication

The application uses OAuth 2.0 to access the document corpus stored in Google Drive.

A Google OAuth client credentials file must be placed in the project root as:

`credentials.json`

During the first authentication, the application opens the Google authorization flow. After successful authentication, the generated credentials are stored locally in:

`token.json`

The stored token can then be reused for subsequent sessions.


### Security 

Sensitive authentication files must remain local and must not be committed to the repository. 

The following files should be included in `.gitignore`:

```text
.env
credentials.json
token.json
```


## Usage

Start the Streamlit application from the Project root: 


```bash
streamlit run ui.py
```

The application provides a Chat interface where the user can select the desired LLM and retrieval strategy from the sidebar. 

Available retrieval strategies include:

- MiniLM
- BGE-M3
- Ensemble

The application then processes document-related questions through the LangGraph RAG workflow:

```text
User Question
     ↓
Query Routing
     ↓
Document Retrieval
     ↓
Context Construction
     ↓
LLM Generation
     ↓
Grounded Answer
```

For questions that do not require document retrieval, such as simple greetings, the agent can respond without invoking the retrieval pipeline.

## Building the Vector Indexes

The Google Drive document corpus is converted into local FAISS vector indexes for semantic retrieval. 

Two vector indexes are maintained:

- MiniLM FAISS index
- BGE-M3 FAISS index

When a Drive retriever is loaded, the system checks whether the corresponding FAISS index already exists locally. If it does not exist, the documents are loaded from Google Drive and the index is created.

The generated indexes are stored under:

```text
data/vectorstores/
```

The Ensemble Retriever does not create a separate vector index. Instead, it combines the results returned by the MiniLM and BGE-M3 retrievers using Weighted Reciprocal Rank Fusion.

The indexing architecture is:

```text

Google Drive Corpus
        ↓
Document Loading
        ↓
Chunking 
        ↓
MiniLM  and BGE-M3
        ↓
FAISS (for both)
        ↓
Weighted RRF
        ↓
Ensemble Retrieval
```

If the document corpus changes, the corresponding FAISS indexes should be rebuilt so that the indexed content remains synchronized with Google Drive.

Generated vector indexes are local artifacts and should not be committed to the repository.



##Technologies 

- Python
- LangGraph
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers
- Ollama
- OpenRouter
- Streamlit
- Google Drive API
- DeepEval
- PyMuPDF
- pikepdf
- Tesseract OCR


