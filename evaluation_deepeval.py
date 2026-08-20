import csv
import json

from collections import defaultdict
from pathlib import Path

from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from deepeval.models.base_model import (
    DeepEvalBaseLLM,
)

from model import load_llm
from retriever import load_retriever
from agent import build_agent

from evaluation.rag_eval_dataset import (
    EVAL_DATASET,
)



# CONFIGURATION

MODELS_TO_TEST = [
    #"krikri",
   "llama",
   "qwen",
   "gpt41_mini",
   "gemini_flash",
   "claude_haiku",
]


RETRIEVERS_TO_TEST =["bge"] 


# Model used only as DeepEval judge.
EVALUATOR_MODEL_KEY = "gpt41_mini"


# OUTPUT FILES


RESULTS_DIR = Path("data/evaluation_results")

JSON_RESULTS_FILE = (RESULTS_DIR/ "deepeval_results.json")

CSV_RESULTS_FILE = (RESULTS_DIR/ "deepeval_results.csv")

DETAILED_CSV_FILE = (RESULTS_DIR/ "deepeval_detailed_results.csv")


# DEEPEVAL MODEL WRAPPER


class LocalLangChainLLM(
    DeepEvalBaseLLM
):
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

EXPECTED_REFUSAL = (
    "Δεν βρέθηκε σαφής απάντηση στις "
    "διαθέσιμες πληροφορίες."
)

def is_correct_refusal(answer: str) -> bool:

    """
    Check whether the model correctly refused to answer
    an unanswerable question.
    """

    normalized_answer = (answer.strip().lower())

    normalized_expected = (EXPECTED_REFUSAL.strip().lower())

    return (normalized_expected in normalized_answer)

# EVALUATE ONE MODEL + RETRIEVER COMBINATION

def run_model_evaluation(
    generator_name: str,
    retriever_name: str,
    evaluator_model,
):
    """
    Evaluate one generator/retriever combination
    against the complete RAG evaluation dataset.

    Returns:
        - overall metric averages
        - averages per category
        - detailed scores for every test case
    """


    print("\n" + "=" * 100)
    print(f"START MODEL: {generator_name.upper()} | RETRIEVER: {retriever_name.upper()}")
    print("=" * 100)

    # LOAD RAG PIPELINE


    print(
        f"\nΦόρτωση generator: "
        f"{generator_name}"
    )

    generator_llm = load_llm(generator_name)

    print(
        f"Φόρτωση retriever: "
        f"{retriever_name}")

    retriever = load_retriever(mode=retriever_name)

    print("Δημιουργία agent..." )

    app = build_agent(generator_llm, retriever)


    # METRICS
    

    metrics = [

        FaithfulnessMetric(threshold=0.5, model=evaluator_model),

        AnswerRelevancyMetric(threshold=0.5, model=evaluator_model),

        ContextualPrecisionMetric(threshold=0.5, model=evaluator_model),

        ContextualRecallMetric(threshold=0.5, model=evaluator_model),
    ] 
    
    # SCORE STORAGE
    

    answerable_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_precision": [],
        "contextual_recall": [],
    }


    unanswerable_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "refusal_accuracy": [],
        "hallucination_rate": [],
    }


    category_scores = defaultdict(
        lambda: {
            "faithfulness": [],
            "answer_relevancy": [],
            "contextual_precision": [],
            "contextual_recall": [],
            "refusal_accuracy": [],
            "hallucination_rate": [],
        }
    )


    detailed_results = []

    # RUN ALL TEST CASES

    total_cases = len(EVAL_DATASET)


    for index, item in enumerate(EVAL_DATASET, start=1):

        question = item["question"]

        expected_answer = item["expected_answer"]

        category = item["category"]

        answerable = item["answerable"]


        print(
        f"\nTEST CASE {index}/{total_cases} "
        f"| MODEL: {generator_name} "
        f"| RETRIEVER: {retriever_name}")

        print(
            f"Category: "
            f"{category}"
        )

        print(
            f"Answerable: "
            f"{answerable}"
        )

        print(
            f"Ερώτηση: "
            f"{question}"
        )


        # RUN RAG


        result = app.invoke({
            "question": question,
            "context": "",
            "answer": "",
            "iterations": 0,
            "sources": [],
        })


        answer = result.get("answer", "")

        context = result.get("context","" )

        print(
            f"\nΑπάντηση: {answer}")


        # CREATE DEEPEVAL TEST CASE

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=[context]
        )


        # SCORES FOR CURRENT QUESTION

        case_scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "contextual_precision": None,
            "contextual_recall": None,
            "refusal_accuracy": None,
            "hallucination_rate": None,
        }

        if answerable:

            metrics_to_run = [FaithfulnessMetric(threshold=0.5, model=evaluator_model),

                AnswerRelevancyMetric(threshold=0.5, model=evaluator_model),

                ContextualPrecisionMetric(threshold=0.5, model=evaluator_model),

                ContextualRecallMetric(threshold=0.5, model=evaluator_model),
            ]

        else:

            metrics_to_run = [
                FaithfulnessMetric(threshold=0.5, model=evaluator_model),

                AnswerRelevancyMetric(threshold=0.5, model=evaluator_model),
            ]

            correct_refusal = (is_correct_refusal(answer))

            refusal_score = (1.0 if correct_refusal else 0.0)

            hallucination_score = (0.0 if correct_refusal else 1.0)

            case_scores["refusal_accuracy"] = refusal_score

            case_scores["hallucination_rate"] = hallucination_score

            unanswerable_scores["refusal_accuracy"].append(refusal_score)

            unanswerable_scores["hallucination_rate"].append(hallucination_score)

            category_scores[category]["refusal_accuracy"].append(refusal_score)

            category_scores[category]["hallucination_rate"].append(hallucination_score)

        # RUN METRICS

        for metric in metrics_to_run:

            try:
                metric.measure(test_case)

                score = metric.score

                print(
                    f"{metric.__class__.__name__}: "
                    f"{score}"
                )

                if score is None:
                    continue

                if isinstance(metric, FaithfulnessMetric):
                    metric_name = ("faithfulness")

                elif isinstance(metric, AnswerRelevancyMetric):
                    metric_name = ("answer_relevancy")

                elif isinstance(metric, ContextualPrecisionMetric):
                    metric_name = ("contextual_precision")

                elif isinstance(metric,ContextualRecallMetric):
                    metric_name = ("contextual_recall")

                else:
                    continue


                case_scores[metric_name] = score


                category_scores[category][metric_name].append(score)


                if answerable:

                    answerable_scores[metric_name].append(score)

                else:

                    if metric_name in ("faithfulness", "answer_relevancy"):

                        unanswerable_scores[metric_name].append(score)

            except Exception as error:

                print(
                    f"{metric.__class__.__name__} "
                    f"απέτυχε: {error}"
                )
                        

        # STORE DETAILED TEST CASE RESULTS

        detailed_results.append({

            "question": question,

            "expected_answer": (expected_answer),

            "actual_answer": (answer), 
            
            "retrieved_context": (context),

            "category": (category),
            
            "answerable": (answerable),
                
            "expected_adas": item.get("expected_adas", []),

            "expected_source_ids": item.get("expected_source_ids", []),
            
            "faithfulness": (case_scores["faithfulness"]),

            "answer_relevancy": ( case_scores["answer_relevancy"]),

            "contextual_precision": (case_scores["contextual_precision"]),

            "contextual_recall": (case_scores["contextual_recall"]),

            "refusal_accuracy": case_scores["refusal_accuracy"],

            "hallucination_rate": case_scores["hallucination_rate"],
        })

    # OVERALL AVERAGES

    answerable_avg_scores = {
    metric_name: (
        sum(metric_values)
        / len(metric_values)
        if metric_values
        else None
    )
    for  (metric_name, metric_values) in answerable_scores.items()
}


    unanswerable_avg_scores = {
        metric_name: (
            sum(metric_values)
            / len(metric_values)
            if metric_values
            else None
        )
        for (
            metric_name,
            metric_values,
        ) in unanswerable_scores.items()
    }

    print(
        f"\nANSWERABLE RESULTS "
        f"- {generator_name} + {retriever_name}" )

    for (
        metric_name,
        score,
    ) in answerable_avg_scores.items():

        if score is None:
            print(
                f"{metric_name}: -"
            )
        else:
            print(
                f"{metric_name}: "
                f"{score:.3f}"
            )


    print(
        f"\nUNANSWERABLE RESULTS "
        f"- {generator_name} + {retriever_name}"
    )

    for (
        metric_name,
        score,
    ) in unanswerable_avg_scores.items():

        if score is None:
            print(
                f"{metric_name}: -"
            )
        else:
            print(
                f"{metric_name}: "
                f"{score:.3f}"
            )
    # CATEGORY AVERAGES


    category_avg_scores = {}


    for (category, metric_dict) in category_scores.items():

        category_avg_scores[category] = {

            metric_name: (
                sum(metric_values)
                / len(metric_values)

                if metric_values

                else None
            )

            for (metric_name, metric_values) in metric_dict.items()
        }


    # PRINT CATEGORY RESULTS

    print("\n")
    print(
        f"ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑ ΚΑΤΗΓΟΡΙΑ "
        f"- {generator_name}"
        f" + {retriever_name}"
    )


    for (category, metric_results) in category_avg_scores.items():

        print(
            f"\nΚατηγορία: "
            f"{category}"
        )


        for (metric_name, score) in metric_results.items():

            if score is None:
                print(f"  {metric_name}: -")

            else:
                print(
                    f"  {metric_name}: "
                    f"{score:.3f}"
                )

    print("\n" + "=" * 100)
    print(f"END MODEL: {generator_name.upper()} | RETRIEVER: {retriever_name.upper()}")
    print("=" * 100)


    # RETURN COMPLETE RESULT

    return {

        "answerable_overall": (answerable_avg_scores),

        "unanswerable_overall": (unanswerable_avg_scores),

        "by_category": (category_avg_scores),

        "details": (detailed_results)
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
    """
    Save summary and detailed evaluation results
    to JSON and CSV files.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    # SAVE COMPLETE JSON


    with JSON_RESULTS_FILE.open("w", encoding="utf-8") as file:

        json.dump(all_scores, file, ensure_ascii=False, indent=4)

    # SUMMARY CSV

    summary_rows = []


    for (experiment_name, experiment_results) in all_scores.items():

        model_name = (experiment_results.get("model", ""))

        retriever_name = (experiment_results.get("retriever", ""))

        # Overall

        answerable_overall = experiment_results.get("answerable_overall", {})
        unanswerable_overall = experiment_results.get("unanswerable_overall", {})

        summary_rows.append({

            "model": model_name,

            "retriever": (retriever_name),

            "category": "answerable_overall",

            "faithfulness": answerable_overall.get("faithfulness"),

            "answer_relevancy": answerable_overall.get("answer_relevancy"),

            "contextual_precision": answerable_overall.get("contextual_precision"),

            "contextual_recall": answerable_overall.get("contextual_recall"),

            "refusal_accuracy": None,

            "hallucination_rate": None,
        })


        # Categories


        category_results = (experiment_results.get("by_category", {}))


        for (category, metrics) in category_results.items():

            summary_rows.append({

                "model": model_name,

                "retriever": (retriever_name),

                "category": "unanswerable_overall",

                "faithfulness": unanswerable_overall.get("faithfulness"),

                "answer_relevancy": unanswerable_overall.get("answer_relevancy"),

                "contextual_precision": None,

                "contextual_recall": None,

                "refusal_accuracy": unanswerable_overall.get("refusal_accuracy"),

                "hallucination_rate": unanswerable_overall.get("hallucination_rate"),
                })


    summary_fieldnames = [
        "model",
        "retriever",
        "category",
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "refusal_accuracy",
        "hallucination_rate",
        ]


    with CSV_RESULTS_FILE.open("w", encoding="utf-8-sig", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=(summary_fieldnames))

        writer.writeheader()

        writer.writerows(summary_rows)

# DETAILED CSV

    detailed_rows = []


    for (experiment_name, experiment_results,) in all_scores.items():

        model_name = (experiment_results.get("model",""))

        retriever_name = ( experiment_results.get("retriever", ""))

        for detail in (experiment_results.get("details", [])):

            detailed_rows.append({

                "model": model_name,

                "retriever": (retriever_name),

                "category": detail.get("category"),

                "answerable": detail.get("answerable"),

                "question": detail.get("question"),

                "expected_answer": detail.get("expected_answer"),

                "actual_answer": detail.get("actual_answer"),

                "retrieved_context": detail.get("retrieved_context"),

                "expected_adas": (", ".join(detail.get("expected_adas", []))),

                "expected_source_ids": "," .join(detail.get("expected_source_ids", [])), 

                "faithfulness": detail.get("faithfulness"),

                "answer_relevancy": detail.get("answer_relevancy"),

                "contextual_precision": detail.get("contextual_precision"),

                "contextual_recall": detail.get("contextual_recall"),

                "refusal_accuracy": detail.get("refusal_accuracy"),

                "hallucination_rate": detail.get("hallucination_rate")
            })


    detailed_fieldnames = [
        "model",
        "retriever",
        "category",
        "answerable",
        "question",
        "expected_answer",
        "actual_answer",
        "retrieved_context",
        "expected_adas",
        "expected_source_ids",
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "refusal_accuracy",
        "hallucination_rate"
    ]


    with DETAILED_CSV_FILE.open("w", encoding="utf-8-sig", newline="",) as file:

        writer = csv.DictWriter(file, fieldnames=(detailed_fieldnames))

        writer.writeheader()

        writer.writerows(
            detailed_rows
        )


    print("\n")
    print("ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")

    print(
        f"JSON: "
        f"{JSON_RESULTS_FILE}")

    print(
        f"Summary CSV: "
        f"{CSV_RESULTS_FILE}")

    print(
        f"Detailed CSV: "
        f"{DETAILED_CSV_FILE}")


# MAIN


if __name__ == "__main__":

    print("DEEPEVAL RAG EVALUATION")

    print(
        f"Test cases: "
        f"{len(EVAL_DATASET)}"
    )

    print(
        f"Models: "
        f"{len(MODELS_TO_TEST)}"
    )

    print(
        f"Retrievers: "
        f"{len(RETRIEVERS_TO_TEST)}"
    )

    print(
        f"Total experiments: "
        f"{len(MODELS_TO_TEST) * len(RETRIEVERS_TO_TEST)}"
    )


    # LOAD EVALUATOR ONCE

    print(
        f"\nΦόρτωση DeepEval evaluator: "
        f"{EVALUATOR_MODEL_KEY}"
    )


    evaluator_llm = load_llm(
        EVALUATOR_MODEL_KEY
    )


    evaluator_model = (
        LocalLangChainLLM(
            evaluator_llm,
            name=(
                f"{EVALUATOR_MODEL_KEY}"
                f"-evaluator"
            ),
        )
    )


    # RUN ALL EXPERIMENTS

    all_scores = {}


    for generator_name in (MODELS_TO_TEST):

        for retriever_name in (RETRIEVERS_TO_TEST):

            experiment_name = (
                f"{generator_name}_"
                f"{retriever_name}")


            print("\n\n")

            print(
                f"ΠΕΙΡΑΜΑ: "
                f"{experiment_name}")

            try:

                result = (
                    run_model_evaluation(generator_name=(generator_name),

                        retriever_name=(retriever_name),

                        evaluator_model=(evaluator_model))
                )


                # Add model/retriever metadata.
                result["model"] = generator_name

                result["retriever"] = retriever_name

                all_scores[experiment_name] = result


            except Exception as error:

                print(
                    f"\nΑπέτυχε το πείραμα "
                    f"{experiment_name}"
                )

                print(error)


                all_scores[experiment_name] = {
                    "model": (generator_name),

                    "retriever": (retriever_name),

                    "answerable_overall": {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "contextual_precision": None,
                    "contextual_recall": None,
                    },

                   "unanswerable_overall": {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "refusal_accuracy": None,
                    "hallucination_rate": None,
                     },

                    "by_category": {},

                    "details": [],
                }


    # OVERALL COMPARISON TABLE

    print("\n\n")
    print("ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ -ANSWERABLE OVERALL")


    print(
        f"{'Experiment':<32}"
        f"{'Faithfulness':<16}"
        f"{'Answer Rel.':<16}"
        f"{'Context Prec.':<16}"
        f"{'Context Recall':<16}"
    )


    for (experiment_name, experiment_results) in all_scores.items():

        overall = experiment_results.get("answerable_overall",{})


        print(
            f"{experiment_name:<32}"
            f"{format_score(overall.get('faithfulness')):<16}"
            f"{format_score(overall.get('answer_relevancy')):<16}"
            f"{format_score(overall.get('contextual_precision')):<16}"
            f"{format_score(overall.get('contextual_recall')):<16}"
        )


    # CATEGORY COMPARISON TABLE
    print("\n\n")
    print(
        "ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ "
        "- ΑΝΑ ΚΑΤΗΓΟΡΙΑ"
    )
    print(
        f"{'Experiment':<32}"
        f"{'Category':<24}"
        f"{'Faith.':<14}"
        f"{'Ans.Rel.':<14}"
        f"{'Ctx.Prec.':<14}"
        f"{'Ctx.Recall':<14}"
    )

    for (experiment_name, experiment_results) in all_scores.items():

        categories = (experiment_results.get("by_category", {})
        )


        for (category, metric_scores) in categories.items():

            print(
                f"{experiment_name:<32}"
                f"{category:<24}"
                f"{format_score(metric_scores.get('faithfulness')):<14}"
                f"{format_score(metric_scores.get('answer_relevancy')):<14}"
                f"{format_score(metric_scores.get('contextual_precision')):<14}"
                f"{format_score(metric_scores.get('contextual_recall')):<14}"
            )

    # SAVE EVERYTHING

    save_results(
        all_scores
    )