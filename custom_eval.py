import csv
import json
import re
import unicodedata

from collections import defaultdict
from pathlib import Path

from model import load_llm
from retriever import load_retriever
from agent import build_agent
from evaluation.rag_eval_dataset import EVAL_DATASET


# CONFIGURATION


MODELS_TO_TEST = [
    #"krikri",
    "llama",
    "qwen",
    "gpt41_mini",
    "gemini_flash",
    "claude_haiku",
]

RETRIEVERS_TO_TEST = [
    "bge",
]

SOURCE_TOP_K = 5


# OUTPUT FILES

RESULTS_DIR = Path("data/evaluation_results")

JSON_RESULTS_FILE = RESULTS_DIR / "custom_eval_results.json"
CSV_RESULTS_FILE = RESULTS_DIR / "custom_eval_results.csv"
DETAILED_CSV_FILE = RESULTS_DIR / "custom_eval_detailed_results.csv"


# TEXT NORMALIZATION

def normalize(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ANSWER EXACTNESS

def calculate_answer_exactness(answer, expected_answer):
    """
    Returns 1.0 when normalized actual and expected answers
    are exactly equal, otherwise 0.0.
    Strictly deterministic.
    """

    actual = normalize(answer)
    expected = normalize(expected_answer)

    return 1.0 if actual == expected else 0.0


# NUMBER ACCURACY

def extract_numbers(text):
    """
    Extract numeric expressions and normalize punctuation.

    Examples:
        17.371,24 -> 1737124
        30/08/2021 -> 30082021
        15-07-2023 -> 15072023
        (15) -> 15
    """

    raw_numbers = re.findall(r"\d+(?:[.,/\-]\d+)*", str(text))
    return [re.sub(r"[^\d]", "", number) for number in raw_numbers]


def calculate_number_accuracy(answer, expected_answer):
    """
    Returns 1.0 when all numbers in the expected answer
    appear in the actual answer.

    Returns None when the expected answer contains no numbers.
    """

    expected_numbers = set(extract_numbers(expected_answer))
    answer_numbers = set(extract_numbers(answer))

    if not expected_numbers:
        return None

    return 1.0 if expected_numbers.issubset(answer_numbers) else 0.0


# SOURCE IDENTIFIER

def get_document_identifier(document):
    ada = str(document.metadata.get("ada") or "").strip()
    source_id = str(document.metadata.get("source_id") or "").strip()

    return ada or source_id


# SOURCE ACCURACY + SOURCE RANK


def evaluate_source(retriever, question, expected_adas, expected_source_ids):
    """
    Returns:
        source_accuracy:
            1.0 if a relevant document is found in top-k,
            otherwise 0.0.

        source_rank:
            1-based rank of the first relevant document.
            None if not found.

        retrieved_ids:
            Unique document identifiers in retrieval order.
    """

    expected_ids = set(expected_adas) | set(expected_source_ids)

    if not expected_ids:
        return None, None, []

    documents = retriever.invoke(question)

    retrieved_ids = []
    seen_ids = set()

    for document in documents:
        document_id = get_document_identifier(document)

        if not document_id or document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        retrieved_ids.append(document_id)

        if len(retrieved_ids) >= SOURCE_TOP_K:
            break

    source_rank = None

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in expected_ids:
            source_rank = rank
            break

    source_accuracy = 1.0 if source_rank is not None else 0.0

    return source_accuracy, source_rank, retrieved_ids


# AVERAGE


def average(values):
    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)


# EVALUATE ONE MODEL + RETRIEVER


def run_model_evaluation(model_key, retriever_mode):

    print("\n" + "=" * 100)
    print(f"CUSTOM EVALUATION | START MODEL: {model_key.upper()} | RETRIEVER: {retriever_mode.upper()}")
    print("=" * 100)

    llm = load_llm(model_key)
    retriever = load_retriever(mode=retriever_mode)
    app = build_agent(llm, retriever)

    overall_scores = {
        "answer_exactness": [],
        "number_accuracy": [],
        "source_accuracy": [],
        "source_rank": [],
    }

    category_scores = defaultdict(
        lambda: {
            "answer_exactness": [],
            "number_accuracy": [],
            "source_accuracy": [],
            "source_rank": [],
        }
    )

    detailed_results = []

    answerable_cases = [item for item in EVAL_DATASET if item["answerable"]]
    total_cases = len(answerable_cases)

    for index, item in enumerate(answerable_cases, start=1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_adas = item.get("expected_adas", [])
        expected_source_ids = item.get("expected_source_ids", [])
        category = item["category"]

        
        print(f"TEST CASE {index}/{total_cases}")
        print(f"Category: {category}")
        print(f"Ερώτηση: {question}")

        result = app.invoke({
            "question": question,
            "context": "",
            "answer": "",
            "iterations": 0,
            "sources": [],
        })

        answer = result.get("answer", "")
        context = result.get("context", "")

        answer_exactness = calculate_answer_exactness(answer, expected_answer)
        number_accuracy = calculate_number_accuracy(answer, expected_answer)

        source_accuracy, source_rank, retrieved_ids = evaluate_source(
            retriever,
            question,
            expected_adas,
            expected_source_ids,
        )

        overall_scores["answer_exactness"].append(answer_exactness)
        overall_scores["source_accuracy"].append(source_accuracy)

        if number_accuracy is not None:
            overall_scores["number_accuracy"].append(number_accuracy)

        if source_rank is not None:
            overall_scores["source_rank"].append(source_rank)

        category_scores[category]["answer_exactness"].append(answer_exactness)
        category_scores[category]["source_accuracy"].append(source_accuracy)

        if number_accuracy is not None:
            category_scores[category]["number_accuracy"].append(number_accuracy)

        if source_rank is not None:
            category_scores[category]["source_rank"].append(source_rank)

        detailed_results.append({
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": answer,
            "retrieved_context": context,
            "category": category,
            "expected_adas": expected_adas,
            "expected_source_ids": expected_source_ids,
            "retrieved_ids": retrieved_ids,
            "answer_exactness": answer_exactness,
            "number_accuracy": number_accuracy,
            "source_accuracy": source_accuracy,
            "source_rank": source_rank,
        })

        print(f"\nΑπάντηση: {answer}")
        print(f"Αναμενόμενη: {expected_answer}")
        print(f"Answer Exactness: {answer_exactness:.0f}")

        if number_accuracy is None:
            print("Number Accuracy: -")
        else:
            print(f"Number Accuracy: {number_accuracy:.0f}")

        print(f"Source Accuracy: {source_accuracy:.0f}")
        print(f"Source Rank: {'-' if source_rank is None else source_rank}")

    
    # OVERALL RESULTS
    

    overall_results = {
        "answer_exactness": average(overall_scores["answer_exactness"]),
        "number_accuracy": average(overall_scores["number_accuracy"]),
        "source_accuracy": average(overall_scores["source_accuracy"]),
        "source_rank": average(overall_scores["source_rank"]),
    }

    
    # CATEGORY RESULTS
    

    category_results = {}

    for category, metrics in category_scores.items():
        category_results[category] = {
            "answer_exactness": average(metrics["answer_exactness"]),
            "number_accuracy": average(metrics["number_accuracy"]),
            "source_accuracy": average(metrics["source_accuracy"]),
            "source_rank": average(metrics["source_rank"]),
        }

    
    # PRINT OVERALL
    

    print(f"RESULTS - {model_key} + {retriever_mode}")

    for metric_name, score in overall_results.items():
        if score is None:
            print(f"{metric_name}: -")
        else:
            print(f"{metric_name}: {score:.3f}")


    # PRINT BY CATEGORY

    print("\nΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑ ΚΑΤΗΓΟΡΙΑ")

    for category, metrics in category_results.items():
        print(f"\nΚατηγορία: {category}")

        for metric_name, score in metrics.items():
            if score is None:
                print(f"  {metric_name}: -")
            else:
                print(f"  {metric_name}: {score:.3f}")

    print("\n" + "=" * 100)
    print(f"END MODEL: {model_key.upper()} | RETRIEVER: {retriever_mode.upper()}")
    print("=" * 100)
    
    return {
        "model": model_key,
        "retriever": retriever_mode,
        "overall": overall_results,
        "by_category": category_results,
        "details": detailed_results,
    }



# FORMAT SCORE

def format_score(score):
    return f"{score:.3f}" if isinstance(score, (int, float)) else "-"


# SAVE RESULTS


def save_results(all_scores):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(all_scores, file, ensure_ascii=False, indent=4)

    # SUMMARY CSV

    summary_rows = []

    for experiment_name, results in all_scores.items():
        model = results.get("model", "")
        retriever = results.get("retriever", "")
        overall = results.get("overall", {})

        summary_rows.append({
            "model": model,
            "retriever": retriever,
            "category": "overall",
            "answer_exactness": overall.get("answer_exactness"),
            "number_accuracy": overall.get("number_accuracy"),
            "source_accuracy": overall.get("source_accuracy"),
            "source_rank": overall.get("source_rank"),
        })

        for category, metrics in results.get("by_category", {}).items():
            summary_rows.append({
                "model": model,
                "retriever": retriever,
                "category": category,
                "answer_exactness": metrics.get("answer_exactness"),
                "number_accuracy": metrics.get("number_accuracy"),
                "source_accuracy": metrics.get("source_accuracy"),
                "source_rank": metrics.get("source_rank"),
            })

    summary_fieldnames = [
        "model",
        "retriever",
        "category",
        "answer_exactness",
        "number_accuracy",
        "source_accuracy",
        "source_rank",
    ]

    with CSV_RESULTS_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # DETAILED CSV

    detailed_rows = []

    for experiment_name, results in all_scores.items():
        model = results.get("model", "")
        retriever = results.get("retriever", "")

        for detail in results.get("details", []):
            detailed_rows.append({
                "model": model,
                "retriever": retriever,
                "category": detail.get("category"),
                "question": detail.get("question"),
                "expected_answer": detail.get("expected_answer"),
                "actual_answer": detail.get("actual_answer"),
                "expected_adas": ", ".join(detail.get("expected_adas", [])),
                "expected_source_ids": ", ".join(detail.get("expected_source_ids", [])),
                "retrieved_ids": ", ".join(detail.get("retrieved_ids", [])),
                "answer_exactness": detail.get("answer_exactness"),
                "number_accuracy": detail.get("number_accuracy"),
                "source_accuracy": detail.get("source_accuracy"),
                "source_rank": detail.get("source_rank"),
                "retrieved_context": detail.get("retrieved_context"),
            })

    detailed_fieldnames = [
        "model",
        "retriever",
        "category",
        "question",
        "expected_answer",
        "actual_answer",
        "expected_adas",
        "expected_source_ids",
        "retrieved_ids",
        "answer_exactness",
        "number_accuracy",
        "source_accuracy",
        "source_rank",
        "retrieved_context",
    ]

    with DETAILED_CSV_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=detailed_fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print("\nΑΠΟΘΗΚΕΥΣΗ CUSTOM EVALUATION")
    print(f"JSON: {JSON_RESULTS_FILE}")
    print(f"Summary CSV: {CSV_RESULTS_FILE}")
    print(f"Detailed CSV: {DETAILED_CSV_FILE}")


# MAIN

if __name__ == "__main__":
    print("CUSTOM RAG EVALUATION")
    print(f"Answerable test cases: {len([item for item in EVAL_DATASET if item['answerable']])}")
    print(f"Models: {len(MODELS_TO_TEST)}")
    print(f"Retrievers: {len(RETRIEVERS_TO_TEST)}")
    print(f"Total experiments: {len(MODELS_TO_TEST) * len(RETRIEVERS_TO_TEST)}")

    all_scores = {}

    for model_key in MODELS_TO_TEST:
        for retriever_mode in RETRIEVERS_TO_TEST:
            experiment_name = f"{model_key}_{retriever_mode}"

            print(f"ΠΕΙΡΑΜΑ: {experiment_name}")

            try:
                all_scores[experiment_name] = run_model_evaluation(model_key, retriever_mode)

            except Exception as error:
                print(f"Απέτυχε το πείραμα {experiment_name}: {error}")

                all_scores[experiment_name] = {
                    "model": model_key,
                    "retriever": retriever_mode,
                    "overall": {},
                    "by_category": {},
                    "details": [],
                }

    # FINAL COMPARISON

    print("\n\nΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ - CUSTOM EVALUATION")

    print(
        f"{'Experiment':<32}"
        f"{'Exactness':<14}"
        f"{'Number Acc.':<14}"
        f"{'Source Acc.':<14}"
        f"{'Source Rank':<14}"
    )

    for experiment_name, results in all_scores.items():
        overall = results.get("overall", {})

        print(
            f"{experiment_name:<32}"
            f"{format_score(overall.get('answer_exactness')):<14}"
            f"{format_score(overall.get('number_accuracy')):<14}"
            f"{format_score(overall.get('source_accuracy')):<14}"
            f"{format_score(overall.get('source_rank')):<14}"
        )

    save_results(all_scores)