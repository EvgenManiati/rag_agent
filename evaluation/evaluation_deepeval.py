import csv
import json
import traceback

from collections import defaultdict
from pathlib import Path

from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from deepeval.models.base_model import (DeepEvalBaseLLM)

from model import load_llm
from retriever import load_retriever
from agent import build_agent

from evaluation.rag_eval_dataset import EVAL_DATASET


import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# CONFIGURATION

MODELS_TO_TEST = [
    "llama",
    "qwen",
    "gpt41_mini",
    "gemini_flash",
    "claude_haiku",
]


RETRIEVERS_TO_TEST =["drive_bge"] 


# Model used only as DeepEval judge.
EVALUATOR_MODEL_KEY = "gpt41_mini"


# OUTPUT FILES


RESULTS_DIR = Path("data/evaluation_results")

JSON_RESULTS_FILE = (RESULTS_DIR/ "deepeval_unanswerable_results_drive_bge_last.json")

CSV_RESULTS_FILE = (RESULTS_DIR/ "deepeval_unanswerable_results_drive_bge_last.csv")

DETAILED_CSV_FILE = (RESULTS_DIR/ "deepeval_unanswerable_detailed_results_drive_bge_last.csv")


# DEEPEVAL MODEL WRAPPER


class LocalLangChainLLM(DeepEvalBaseLLM):
    """
    Adapter that allows an LLM loaded through model.py
    to be used as a DeepEval evaluator model.
    """

    def __init__(self, llm, name="local-llm"):
        self.llm = llm
        self.name = name

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None) -> str:
        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            response = (response.content)
        return str(response)

    async def a_generate(self, prompt: str, schema=None) -> str:

        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.name

EXPECTED_REFUSAL = ("Δεν βρέθηκε σαφής απάντηση στις διαθέσιμες πληροφορίες.")

def is_correct_refusal(answer: str) -> bool:

    """
    Check whether the model correctly refused to answer
    an unanswerable question.
    """

    normalized_answer = (answer.strip().lower())

    normalized_expected = (EXPECTED_REFUSAL.strip().lower())

    return (normalized_expected in normalized_answer)

# EVALUATE ONE MODEL + RETRIEVER COMBINATION

def average(values): 
    return sum(values) / len(values) if values else None

def run_model_evaluation(generator_name: str, retriever_name: str, evaluator_model):
    print(f"EVALUATION | Generator: {generator_name} | Retriever: {retriever_name}")

    print(100 * "-")
    print(f"\nΦόρτωση generator: {generator_name}")
    generator_llm = load_llm(generator_name)

    print(100 * "-")
    print(f"Φόρτωση retriever: {retriever_name}")
    retriever = load_retriever(mode=retriever_name)

    print(100 * "-")
    print("Δημιουργία agent...")
    app = build_agent(generator_llm, retriever)

    answerable_scores = {"faithfulness": [], "answer_relevancy": [], "contextual_precision": [], "contextual_recall": []}
    unanswerable_scores = {"faithfulness": [], "answer_relevancy": [], "refusal_accuracy": [], "hallucination_rate": []}

    category_scores = defaultdict(lambda: {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_precision": [],
        "contextual_recall": [],
        "refusal_accuracy": [],
        "hallucination_rate": [],
    })

    category_counts = defaultdict(int)
    detailed_results = []
    total_cases = len(EVAL_DATASET)

    for index, item in enumerate(EVAL_DATASET, start=1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        category = item["category"]
        answerable = item["answerable"]
        category_counts[category] += 1

        print(f"\nTEST CASE {index}/{total_cases}")
        print(f"Category: {category}")
        print(f"Answerable: {answerable}")
        print(f"Ερώτηση: {question}")

        result = app.invoke({"question": question, "context": "", "answer": "", "iterations": 0, "sources": []})
        answer = result.get("answer", "")
        context = result.get("context", "")

        print(f"\nΑπάντηση: {answer}")

        test_case = LLMTestCase(input=question, actual_output=answer, expected_output=expected_answer, retrieval_context=[context])

        case_scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "contextual_precision": None,
            "contextual_recall": None,
            "refusal_accuracy": None,
            "hallucination_rate": None,
        }

        if answerable:
            metric_specs = [
                ("faithfulness", FaithfulnessMetric(threshold=0.5, model=evaluator_model)),
                ("answer_relevancy", AnswerRelevancyMetric(threshold=0.5, model=evaluator_model)),
                ("contextual_precision", ContextualPrecisionMetric(threshold=0.5, model=evaluator_model)),
                ("contextual_recall", ContextualRecallMetric(threshold=0.5, model=evaluator_model)),
            ]
        else:
            metric_specs = [
                ("faithfulness", FaithfulnessMetric(threshold=0.5, model=evaluator_model)),
                ("answer_relevancy", AnswerRelevancyMetric(threshold=0.5, model=evaluator_model)),
            ]

            correct_refusal = is_correct_refusal(answer)
            refusal_score = 1.0 if correct_refusal else 0.0
            hallucination_score = 0.0 if correct_refusal else 1.0

            case_scores["refusal_accuracy"] = refusal_score
            case_scores["hallucination_rate"] = hallucination_score

            unanswerable_scores["refusal_accuracy"].append(refusal_score)
            unanswerable_scores["hallucination_rate"].append(hallucination_score)
            category_scores[category]["refusal_accuracy"].append(refusal_score)
            category_scores[category]["hallucination_rate"].append(hallucination_score)

        for metric_name, metric in metric_specs:
            try:
                metric.measure(test_case)
                score = metric.score
                print(f"{metric_name}: {score}")

                if score is None:
                    continue

                case_scores[metric_name] = score
                category_scores[category][metric_name].append(score)

                if answerable:
                    answerable_scores[metric_name].append(score)
                else:
                    unanswerable_scores[metric_name].append(score)

            except Exception as error:
                print(f"{metric_name} απέτυχε: {error}")

        detailed_results.append({
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": answer,
            "retrieved_context": context,
            "category": category,
            "answerable": answerable,
            "expected_adas": item.get("expected_adas", []),
            "faithfulness": case_scores["faithfulness"],
            "answer_relevancy": case_scores["answer_relevancy"],
            "contextual_precision": case_scores["contextual_precision"],
            "contextual_recall": case_scores["contextual_recall"],
            "refusal_accuracy": case_scores["refusal_accuracy"],
            "hallucination_rate": case_scores["hallucination_rate"],
        })

    answerable_avg_scores = {"n": sum(1 for item in EVAL_DATASET if item["answerable"])}
    answerable_avg_scores.update({metric_name: average(metric_values) for metric_name, metric_values in answerable_scores.items()})

    unanswerable_avg_scores = {"n": sum(1 for item in EVAL_DATASET if not item["answerable"])}
    unanswerable_avg_scores.update({metric_name: average(metric_values) for metric_name, metric_values in unanswerable_scores.items()})

    category_avg_scores = {}

    for category, metric_dict in category_scores.items():
        category_avg_scores[category] = {"n": category_counts[category]}
        category_avg_scores[category].update({metric_name: average(metric_values) for metric_name, metric_values in metric_dict.items()})

    print(f"\nANSWERABLE OVERALL - {generator_name} + {retriever_name}")
    for metric_name, score in answerable_avg_scores.items():
        print(f"{metric_name}: {score if metric_name == 'n' else format_score(score)}")

    print(f"\nUNANSWERABLE OVERALL - {generator_name} + {retriever_name}")
    for metric_name, score in unanswerable_avg_scores.items():
        print(f"{metric_name}: {score if metric_name == 'n' else format_score(score)}")

    print(f"\nΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑ ΚΑΤΗΓΟΡΙΑ - {generator_name} + {retriever_name}")

    for category, metric_results in category_avg_scores.items():
        print(f"\nΚατηγορία: {category} | n={metric_results['n']}")
        for metric_name, score in metric_results.items():
            if metric_name != "n":
                print(f"  {metric_name}: {format_score(score)}")

    return {
        "answerable_overall": answerable_avg_scores,
        "unanswerable_overall": unanswerable_avg_scores,
        "by_category": category_avg_scores,
        "details": detailed_results,
    }


# SCORE FORMATTER

def format_score(score):

    """
    Format numeric scores with three decimal places.
    """

    if isinstance(score, (int, float)):

        return (f"{score:.3f}")

    return "-"


# SAVE RESULTS

def save_results(all_scores):
    """Save complete JSON, summary CSV and detailed CSV results."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(all_scores, file, ensure_ascii=False, indent=4)

    summary_rows = []

    for experiment_name, experiment_results in all_scores.items():
        model_name = experiment_results.get("model", "")
        retriever_name = experiment_results.get("retriever", "")

        answerable = experiment_results.get("answerable_overall", {})
        summary_rows.append({
            "model": model_name,
            "retriever": retriever_name,
            "scope": "answerable_overall",
            "category": "overall",
            "n": answerable.get("n"),
            "faithfulness": answerable.get("faithfulness"),
            "answer_relevancy": answerable.get("answer_relevancy"),
            "contextual_precision": answerable.get("contextual_precision"),
            "contextual_recall": answerable.get("contextual_recall"),
            "refusal_accuracy": None,
            "hallucination_rate": None,
        })

        unanswerable = experiment_results.get("unanswerable_overall", {})
        summary_rows.append({
            "model": model_name,
            "retriever": retriever_name,
            "scope": "unanswerable_overall",
            "category": "overall",
            "n": unanswerable.get("n"),
            "faithfulness": unanswerable.get("faithfulness"),
            "answer_relevancy": unanswerable.get("answer_relevancy"),
            "contextual_precision": None,
            "contextual_recall": None,
            "refusal_accuracy": unanswerable.get("refusal_accuracy"),
            "hallucination_rate": unanswerable.get("hallucination_rate"),
        })

        for category, metrics in experiment_results.get("by_category", {}).items():
            summary_rows.append({
                "model": model_name,
                "retriever": retriever_name,
                "scope": "category",
                "category": category,
                "n": metrics.get("n"),
                "faithfulness": metrics.get("faithfulness"),
                "answer_relevancy": metrics.get("answer_relevancy"),
                "contextual_precision": metrics.get("contextual_precision"),
                "contextual_recall": metrics.get("contextual_recall"),
                "refusal_accuracy": metrics.get("refusal_accuracy"),
                "hallucination_rate": metrics.get("hallucination_rate"),
            })

    summary_fieldnames = ["model", "retriever", "scope", "category", "n", "faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "refusal_accuracy", "hallucination_rate"]

    with CSV_RESULTS_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    detailed_rows = []

    for experiment_name, experiment_results in all_scores.items():
        model_name = experiment_results.get("model", "")
        retriever_name = experiment_results.get("retriever", "")

        for detail in experiment_results.get("details", []):
            detailed_rows.append({
                "model": model_name,
                "retriever": retriever_name,
                "category": detail.get("category"),
                "answerable": detail.get("answerable"),
                "question": detail.get("question"),
                "expected_answer": detail.get("expected_answer"),
                "actual_answer": detail.get("actual_answer"),
                "retrieved_context": detail.get("retrieved_context"),
                "expected_adas": ", ".join(detail.get("expected_adas", [])),
                "faithfulness": detail.get("faithfulness"),
                "answer_relevancy": detail.get("answer_relevancy"),
                "contextual_precision": detail.get("contextual_precision"),
                "contextual_recall": detail.get("contextual_recall"),
                "refusal_accuracy": detail.get("refusal_accuracy"),
                "hallucination_rate": detail.get("hallucination_rate"),
            })

    detailed_fieldnames = ["model", "retriever", "category", "answerable", "question", "expected_answer", "actual_answer", "retrieved_context", "expected_adas", "faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "refusal_accuracy", "hallucination_rate"]

    with DETAILED_CSV_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=detailed_fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print("\nΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print(f"JSON: {JSON_RESULTS_FILE}")
    print(f"Summary CSV: {CSV_RESULTS_FILE}")
    print(f"Detailed CSV: {DETAILED_CSV_FILE}")

# MAIN

if __name__ == "__main__":
    print("DEEPEVAL RAG EVALUATION")
    print(f"Test cases: {len(EVAL_DATASET)}")
    print(f"Models: {len(MODELS_TO_TEST)}")
    print(f"Retrievers: {len(RETRIEVERS_TO_TEST)}")
    print(f"Total experiments: {len(MODELS_TO_TEST) * len(RETRIEVERS_TO_TEST)}")

    print(f"\nΦόρτωση DeepEval evaluator: {EVALUATOR_MODEL_KEY}")
    evaluator_llm = load_llm(EVALUATOR_MODEL_KEY)
    evaluator_model = LocalLangChainLLM(evaluator_llm, name=f"{EVALUATOR_MODEL_KEY}-evaluator")

    all_scores = {}

    for generator_name in MODELS_TO_TEST:
        for retriever_name in RETRIEVERS_TO_TEST:
            experiment_name = f"{generator_name}_{retriever_name}"
            print(f"\n\nΠΕΙΡΑΜΑ: {experiment_name}")

            try:
                result = run_model_evaluation(generator_name=generator_name, retriever_name=retriever_name, evaluator_model=evaluator_model)
                result["model"] = generator_name
                result["retriever"] = retriever_name
                all_scores[experiment_name] = result

            except Exception as error:
                print(f"\nΑπέτυχε το πείραμα {experiment_name}")
                print(error)

                all_scores[experiment_name] = {
                    "model": generator_name,
                    "retriever": retriever_name,
                    "answerable_overall": {
                        "n": 0,
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "contextual_precision": None,
                        "contextual_recall": None,
                    },
                    "unanswerable_overall": {
                        "n": 0,
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "refusal_accuracy": None,
                        "hallucination_rate": None,
                    },
                    "by_category": {},
                    "details": [],
                }

    print("\n\nΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ - ANSWERABLE OVERALL")
    print(f"{'Experiment':<32}{'N':<6}{'Faithfulness':<16}{'Answer Rel.':<16}{'Context Prec.':<16}{'Context Recall':<16}")

    for experiment_name, experiment_results in all_scores.items():
        overall = experiment_results.get("answerable_overall", {})
        print(f"{experiment_name:<32}{str(overall.get('n', '-')):<6}{format_score(overall.get('faithfulness')):<16}{format_score(overall.get('answer_relevancy')):<16}{format_score(overall.get('contextual_precision')):<16}{format_score(overall.get('contextual_recall')):<16}")

    print("\n\nΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ - UNANSWERABLE OVERALL")
    print(f"{'Experiment':<32}{'N':<6}{'Faithfulness':<16}{'Answer Rel.':<16}{'Refusal Acc.':<16}{'Halluc. Rate':<16}")

    for experiment_name, experiment_results in all_scores.items():
        overall = experiment_results.get("unanswerable_overall", {})
        print(f"{experiment_name:<32}{str(overall.get('n', '-')):<6}{format_score(overall.get('faithfulness')):<16}{format_score(overall.get('answer_relevancy')):<16}{format_score(overall.get('refusal_accuracy')):<16}{format_score(overall.get('hallucination_rate')):<16}")

    print("\n\nΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ - ΑΝΑ ΚΑΤΗΓΟΡΙΑ")
    print(f"{'Experiment':<32}{'Category':<24}{'N':<6}{'Faith.':<12}{'Ans.Rel.':<12}{'Ctx.Prec.':<12}{'Ctx.Recall':<12}{'Refusal':<12}{'Halluc.':<12}")

    for experiment_name, experiment_results in all_scores.items():
        categories = experiment_results.get("by_category", {})

        for category, metric_scores in categories.items():
            print(f"{experiment_name:<32}{category:<24}{str(metric_scores.get('n', '-')):<6}{format_score(metric_scores.get('faithfulness')):<12}{format_score(metric_scores.get('answer_relevancy')):<12}{format_score(metric_scores.get('contextual_precision')):<12}{format_score(metric_scores.get('contextual_recall')):<12}{format_score(metric_scores.get('refusal_accuracy')):<12}{format_score(metric_scores.get('hallucination_rate')):<12}")

    save_results(all_scores)