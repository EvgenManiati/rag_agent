import csv
from pathlib import Path
from collections import defaultdict

DETAILED_FILE = Path("data/evaluation_results/deepeval_detailed_results_drive_ensemble.csv")

results = defaultdict(lambda: {
    "faithfulness": [],
    "answer_relevancy": [],
    "refusal_accuracy": [],
    "hallucination_rate": [],
})


def add_metric(container, metric, value):
    if value is None:
        return

    value = str(value).strip()

    if value == "":
        return

    try:
        container[metric].append(float(value))
    except ValueError:
        pass


with DETAILED_FILE.open("r", encoding="utf-8-sig", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        # Κρατάμε μόνο unanswerable test cases.
        answerable = str(row.get("answerable", "")).strip().lower()

        if answerable not in {"false", "0", "no"}:
            continue

        model = str(row.get("model", "")).strip()

        if not model:
            continue

        add_metric(results[model], "faithfulness", row.get("faithfulness"))

        add_metric(results[model], "answer_relevancy", row.get("answer_relevancy"))

        add_metric(results[model], "refusal_accuracy", row.get("refusal_accuracy"))

        add_metric(results[model], "hallucination_rate", row.get("hallucination_rate"))


print("\nUNANSWERABLE RESULTS - DRIVE ENSEMBLE\n")

print(
    f"{'Model':<20}"
    f"{'Faithfulness':<16}"
    f"{'Answer Rel.':<16}"
    f"{'Refusal Acc.':<16}"
    f"{'Halluc. Rate':<16}"
)


for model, metrics in results.items():

    averages = {}

    for metric, values in metrics.items():

        if values:
            averages[metric] = (sum(values) / len(values))
        else:
            averages[metric] = None

    def display(value):
        if value is None:
            return "-"

        return f"{value:.3f}"

    print(
        f"{model:<20}"
        f"{display(averages['faithfulness']):<16}"
        f"{display(averages['answer_relevancy']):<16}"
        f"{display(averages['refusal_accuracy']):<16}"
        f"{display(averages['hallucination_rate']):<16}"
    )