import json
from pathlib import Path

RESULTS_FILE = Path("data/evaluation/retrieval_benchmark.json")

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

queries = results["drive_bge"]["queries"]

print("\nDOCUMENT / FILE HIT@1 FAILURES\n")

for query in queries:
    metrics = query["metrics"]

    if metrics["hit@1"] == 0:
        print("=" * 80)
        print("Query:", query["query"])
        print("Expected ADA:", query.get("expected_adas", []))
        print("Expected source:", query.get("expected_source_ids", []))
        print("Expected file:", query.get("expected_file_names", []))
        print("Correct rank:", metrics["rank"])
        print("\nRanking:")

        for rank, result in enumerate(query["ranking"], start=1):
            print(f"{rank}. {result.get('document_id')} | file={result.get('file_name')} | folder={result.get('folder_name')}")

print("\n\nFOLDER HIT@1 FAILURES\n")

for query in queries:
    folder_metrics = query.get("folder_metrics", {})

    if folder_metrics.get("folder_hit@1") == 0:
        print("=" * 80)
        print("Query:", query["query"])
        print("Expected folder:", query.get("expected_folder_names", []))
        print("Correct folder rank:", folder_metrics.get("folder_rank"))
        print("\nRanking:")

        for rank, result in enumerate(query["ranking"], start=1):
            print(f"{rank}. {result.get('document_id')} | file={result.get('file_name')} | folder={result.get('folder_name')}")