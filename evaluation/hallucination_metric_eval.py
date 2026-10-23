import csv
import json

from collections import defaultdict
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from model import load_llm


# CONFIGURATION

DEEPEVAL_RESULTS_FILE = Path("data/evaluation_results/deepeval_final_results.json")
CORPUS_FILE = Path("data/diavgeia/final_dataset.jsonl")
RESULTS_DIR = Path("data/evaluation_results")

JSON_OUTPUT_FILE = RESULTS_DIR / "hallucination_metric_final_results.json"
CSV_OUTPUT_FILE = RESULTS_DIR / "hallucination_metric_final_results.csv"
DETAILED_CSV_OUTPUT_FILE = RESULTS_DIR / "hallucination_metric_final_detailed_results.csv"

EVALUATOR_MODEL_KEY = "gpt41_mini"


# DEEPEVAL MODEL WRAPPER

class LocalLangChainLLM(DeepEvalBaseLLM):

    def __init__(self, llm, name="local-llm"):
        self.llm = llm
        self.name = name

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None) -> str:
        response = self.llm.invoke(prompt)
        if hasattr(response, "content"):
            response = response.content
        return str(response)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.name


# LOAD GROUND-TRUTH CORPUS

def load_ground_truth_by_ada():
    documents = {}

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            ada = str(record.get("ada") or "").strip()

            if not ada:
                continue

            text = str(record.get("text") or "").strip()

            if not text:
                pages = record.get("pages") or []
                text = "\n\n".join(str(page) for page in pages).strip()

            if text:
                if ada not in documents:
                    documents[ada] = []
                documents[ada].append(text)

    return {ada: "\n\n".join(texts) for ada, texts in documents.items()}


# LOAD EXISTING DEEPEVAL RESULTS

def load_existing_results():
    with DEEPEVAL_RESULTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


# NORMALIZE EXPECTED ADA FIELD

def normalize_expected_adas(value):
    if not value:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    return []


# EVALUATION

def run_hallucination_evaluation(previous_results, ground_truth_by_ada, evaluator_model):
    final_results = {}

    for experiment_name, experiment_results in previous_results.items():
        model_name = experiment_results.get("model", experiment_name)
        retriever_name = experiment_results.get("retriever", "")
        details = experiment_results.get("details", [])
        answerable_cases = [detail for detail in details if detail.get("answerable") is True]

        if not answerable_cases:
            print(f"\nSKIP {experiment_name}: no completed answerable cases.")
            continue

        print(f"\nHALLUCINATION METRIC | MODEL: {model_name.upper()} | RETRIEVER: {retriever_name.upper()}")

        scores = []
        category_scores = defaultdict(list)
        detailed_results = []
        total = len(answerable_cases)

        for index, detail in enumerate(answerable_cases, start=1):
            question = detail.get("question") or ""
            actual_answer = detail.get("actual_answer") or ""
            category = detail.get("category") or "unknown"
            expected_adas = normalize_expected_adas(detail.get("expected_adas", []))

            print(f"\n[{index}/{total}] {category}")
            print(f"Question: {question}")

            ground_truth_context = []

            for ada in expected_adas:
                document_text = ground_truth_by_ada.get(ada)

                if document_text:
                    ground_truth_context.append(document_text)

            if not ground_truth_context:
                print(f"SKIP: ground-truth context not found for {expected_adas}")
                detailed_results.append({"question": question, "actual_answer": actual_answer, "category": category, "expected_adas": expected_adas, "hallucination_metric": None, "reason": "Ground-truth context not found"})
                continue

            test_case = LLMTestCase(input=question, actual_output=actual_answer, context=ground_truth_context)
            metric = HallucinationMetric(threshold=0.5, model=evaluator_model, include_reason=True, async_mode=False)

            try:
                metric.measure(test_case)

                score = metric.score
                reason = metric.reason

                print(f"HallucinationMetric: {score:.3f}")
                print(f"Reason: {reason}")

                if score is not None and not 0 <= score <= 1:
                    print("WARNING: INVALID HALLUCINATION SCORE")
                    print(f"Score: {score}")
                    print(f"Question: {question}")

                scores.append(score)
                category_scores[category].append(score)

                detailed_results.append({"question": question, "actual_answer": actual_answer, "category": category, "expected_adas": expected_adas, "hallucination_metric": score, "reason": reason})

            except Exception as error:
                print(f"HallucinationMetric failed: {error}")
                detailed_results.append({"question": question, "actual_answer": actual_answer, "category": category, "expected_adas": expected_adas, "hallucination_metric": None, "reason": str(error)})

        print("\n--- SCORE DEBUG ---")
        print("Number of scores:", len(scores))
        print("Scores:", scores)
        print("Sum:", sum(scores))

        if scores:
            print("Min:", min(scores))
            print("Max:", max(scores))
            print("Average:", sum(scores) / len(scores))
            invalid_scores = [score for score in scores if score < 0 or score > 1]
            print("Invalid scores:", invalid_scores)

        overall = sum(scores) / len(scores) if scores else None
        by_category = {}

        for category, values in category_scores.items():
            by_category[category] = {"n": len(values), "hallucination_metric": sum(values) / len(values)}
            
        print("\nOVERALL HALLUCINATION METRIC:")

        if overall is None:
            print("-")
        else:
            print(f"{overall:.3f}")

        print("\nHALLUCINATION METRIC BY CATEGORY:")

        for category, values in category_scores.items():
            category_average = sum(values) / len(values)
            print(f"{category}: n={len(values)} | hallucination_metric={category_average:.3f}")

        final_results[experiment_name] = {"model": model_name, "retriever": retriever_name, "n": len(scores), "hallucination_metric": overall, "by_category": by_category, "details": detailed_results}

    return final_results


# SAVE RESULTS

def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    summary_rows = []

    for experiment_name, result in results.items():
        summary_rows.append({"model": result.get("model"), "retriever": result.get("retriever"), "category": "overall", "n": result.get("n"), "hallucination_metric": result.get("hallucination_metric")})

        for category, category_result in result.get("by_category", {}).items():
            summary_rows.append({"model": result.get("model"), "retriever": result.get("retriever"), "category": category, "n": category_result.get("n"), "hallucination_metric": category_result.get("hallucination_metric")})

    with CSV_OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["model", "retriever", "category", "n", "hallucination_metric"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    detailed_rows = []

    for result in results.values():
        for detail in result.get("details", []):
            detailed_rows.append({"model": result.get("model"), "retriever": result.get("retriever"), "category": detail.get("category"), "question": detail.get("question"), "actual_answer": detail.get("actual_answer"), "expected_adas": ", ".join(detail.get("expected_adas", [])), "hallucination_metric": detail.get("hallucination_metric"), "reason": detail.get("reason")})

    with DETAILED_CSV_OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["model", "retriever", "category", "question", "actual_answer", "expected_adas", "hallucination_metric", "reason"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print("\nRESULTS SAVED")
    print(f"JSON: {JSON_OUTPUT_FILE}")
    print(f"CSV: {CSV_OUTPUT_FILE}")
    print(f"Detailed CSV: {DETAILED_CSV_OUTPUT_FILE}")


# MAIN

if __name__ == "__main__":
    print("DEEPEVAL HALLUCINATION METRIC ONLY")

    print("\nLoading existing DeepEval results...")
    previous_results = load_existing_results()

    print("Loading ground-truth corpus...")
    ground_truth_by_ada = load_ground_truth_by_ada()
    print(f"Ground-truth documents: {len(ground_truth_by_ada)}")

    print(f"\nLoading evaluator: {EVALUATOR_MODEL_KEY}")
    evaluator_llm = load_llm(EVALUATOR_MODEL_KEY)
    evaluator_model = LocalLangChainLLM(evaluator_llm, name=f"{EVALUATOR_MODEL_KEY}-hallucination-evaluator")

    results = run_hallucination_evaluation(previous_results=previous_results, ground_truth_by_ada=ground_truth_by_ada, evaluator_model=evaluator_model)
    save_results(results)