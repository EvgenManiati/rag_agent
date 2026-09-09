from retriever import load_retriever
from evaluation.retrieval_eval_ground_truth import (
    VALIDATION_SET, TEST_SET,
)

import json
from pathlib import Path

from google_drive_loader import authenticate_google_drive, collect_pdf_files
from config import GOOGLE_DRIVE_ROOT_FOLDER_ID

RESULTS_DIRECTORY = Path(
    "data/evaluation"
)

RESULTS_FILE = (
    RESULTS_DIRECTORY
    / "retrieval_benchmark.json"
)


RETRIEVERS = ["drive_minilm", "drive_ensemble", "drive_bge"]

BENCHMARK_SET = TEST_SET  # Use TEST_SET for evaluation

# Ζητάμε περισσότερα chunks από όσα τελικά αξιολογούμε,
# επειδή μπορεί πολλά chunks να ανήκουν στον ίδιο ΑΔΑ.
#RAW_RETRIEVAL_K = 15

# Αξιολογούμε μέχρι τους πρώτους 5 μοναδικούς ΑΔΑ.
EVALUATION_K = 5


# Helpers

def get_unique_document_ranking(documents, max_results=5):
    """Convert retrieved chunks into a ranking of unique source documents."""

    unique_results = []
    seen_documents = set()

    for document in documents:
        ada = str(document.metadata.get("ada") or "").strip()
        source_id = str(document.metadata.get("source_id") or "").strip()
        file_name = str(document.metadata.get("file_name") or "").strip()
        folder_name = str(document.metadata.get("folder_name") or "").strip()
        drive_path = str(document.metadata.get("drive_path") or "").strip()

        document_id = ada or source_id or file_name

        if not document_id or document_id in seen_documents:
            continue

        seen_documents.add(document_id)

        unique_results.append({
            "document_id": document_id,
            "ada": ada or None,
            "source_id": source_id or None,
            "file_name": file_name or None,
            "folder_name": folder_name or None,
            "drive_path": drive_path or None,
            "source": document.metadata.get("source"),
            "chunk_id": document.metadata.get("chunk_id"),
        })

        if len(unique_results) >= max_results:
            break

    return unique_results

def find_first_relevant_rank(ranking, expected_adas=None, expected_source_ids=None, expected_file_names=None):
    expected_adas = expected_adas or []
    expected_source_ids = expected_source_ids or []
    expected_file_names = expected_file_names or []

    expected_documents = set(expected_adas) | set(expected_source_ids) | set(expected_file_names)

    for rank, result in enumerate(ranking, start=1):
        if result["document_id"] in expected_documents:
            return rank

    return None


def calculate_query_metrics(ranking, expected_adas=None, expected_source_ids=None, expected_file_names=None):
    rank = find_first_relevant_rank(ranking, expected_adas, expected_source_ids, expected_file_names)

    if rank is None:
        return {"rank": None, 
                "hit@1": 0, 
                "hit@3": 0, 
                "hit@5": 0, 
                "rr": 0.0}

    return {"rank": rank, 
            "hit@1": int(rank <= 1), 
            "hit@3": int(rank <= 3), 
            "hit@5": int(rank <= 5), 
            "rr": 1.0 / rank}

def find_first_file_rank(ranking, expected_adas=None, expected_file_names=None):
    expected_adas = expected_adas or []
    expected_file_names = expected_file_names or []

    expected_files = set(expected_file_names)

    for ada in expected_adas:
        expected_files.add(f"{ada}.pdf")

    for rank, result in enumerate(ranking, start=1):
        if result.get("file_name") in expected_files:
            return rank

    return None

def calculate_file_metrics(ranking, expected_adas=None, expected_file_names=None):
    rank = find_first_file_rank(ranking, expected_adas, expected_file_names)

    if rank is None:
        return {"file_rank": None, "file_hit@1": 0, "file_hit@3": 0, "file_hit@5": 0, "file_rr": 0.0}

    return {"file_rank": rank, "file_hit@1": int(rank <= 1), "file_hit@3": int(rank <= 3), "file_hit@5": int(rank <= 5), "file_rr": 1.0 / rank}


def find_first_folder_rank(ranking, expected_folder_names=None):
    expected_folder_names = expected_folder_names or []
    expected_folders = set(expected_folder_names)

    for rank, result in enumerate(ranking, start=1):
        if result.get("folder_name") in expected_folders:
            return rank

    return None

def calculate_folder_metrics(ranking, expected_folder_names=None):
    rank = find_first_folder_rank(ranking, expected_folder_names)

    if rank is None:
        return {"folder_rank": None,
                "folder_hit@1": 0, 
                "folder_hit@3": 0, 
                "folder_hit@5": 0, 
                "folder_rr": 0.0}

    return {"folder_rank": rank, 
            "folder_hit@1": int(rank <= 1), 
            "folder_hit@3": int(rank <= 3), 
            "folder_hit@5": int(rank <= 5), 
            "folder_rr": 1.0 / rank}


def build_drive_folder_map():
    """Δημιουργεί mapping ADA/source filename -> άμεσο parent folder από τη δομή του Drive."""

    service = authenticate_google_drive()

    files = collect_pdf_files(service=service, folder_id=GOOGLE_DRIVE_ROOT_FOLDER_ID, recursive=True)

    folder_map = {}

    for file_info in files:
        file_name = file_info.get("name", "")
        drive_path = file_info.get("virtual_path", file_name)
        path_parts = drive_path.split("/")

        folder_name = path_parts[-2] if len(path_parts) >= 2 else ""
        file_stem = Path(file_name).stem

        folder_map[file_stem] = folder_name
        folder_map[file_name] = folder_name

    return folder_map

# Retriever evaluation

def evaluate_retriever(
    retriever_name,
):
    """
    Evaluate one retriever against the ground-truth set.
    """

    print(
        f"EVALUATING RETRIEVER: "
        f"{retriever_name.upper()}"
    )

    retriever = load_retriever(
        retriever_name
    )

    drive_folder_map = build_drive_folder_map() if retriever_name == "drive_bge" else {}

    total_hit_1 = 0
    total_hit_3 = 0
    total_hit_5 = 0
    total_rr = 0.0

    total_file_hit_1 = 0
    total_file_hit_3 = 0
    total_file_hit_5 = 0
    total_file_rr = 0.0

    total_folder_hit_1 = 0
    total_folder_hit_3 = 0
    total_folder_hit_5 = 0
    total_folder_rr = 0.0
    folder_query_count = 0

    query_results = []

    for test_number, test_case in enumerate(
        BENCHMARK_SET,
        start=1,
    ):
        query = test_case["query"]
        expected_adas = test_case.get("expected_adas", [])
        expected_source_ids = test_case.get("expected_source_ids", [])
        expected_file_names = test_case.get("expected_file_names", [])
        expected_folder_names = test_case.get("expected_folder_names", [])

        if retriever_name == "drive_bge" and not expected_folder_names:
            expected_folder_names = [drive_folder_map[ada] for ada in expected_adas if ada in drive_folder_map]

        # Retrieve chunks.
        documents = retriever.invoke(query)

        # Convert chunk results into unique ADA ranking.
        ranking = get_unique_document_ranking(
            documents,
            max_results=EVALUATION_K,
        )

        metrics = calculate_query_metrics(
            ranking,
            expected_adas= expected_adas,
            expected_source_ids= expected_source_ids,
            expected_file_names= expected_file_names
        )

        file_metrics = calculate_file_metrics(
            ranking,
            expected_adas= expected_adas,
            expected_file_names= expected_file_names)

        folder_metrics = calculate_folder_metrics(
            ranking,
            expected_folder_names= expected_folder_names)
        
        total_hit_1 += metrics["hit@1"]
        total_hit_3 += metrics["hit@3"]
        total_hit_5 += metrics["hit@5"]
        total_rr += metrics["rr"]


        total_file_hit_1 += file_metrics["file_hit@1"]
        total_file_hit_3 += file_metrics["file_hit@3"]
        total_file_hit_5 += file_metrics["file_hit@5"]
        total_file_rr += file_metrics["file_rr"]

        if expected_folder_names:
            total_folder_hit_1 += folder_metrics["folder_hit@1"]
            total_folder_hit_3 += folder_metrics["folder_hit@3"]
            total_folder_hit_5 += folder_metrics["folder_hit@5"]
            total_folder_rr += folder_metrics["folder_rr"]
            folder_query_count += 1

        query_results.append(
            {
                "query": query,
                "expected_adas": expected_adas,
                "expected_source_ids" : expected_source_ids,
                "expected_file_names": expected_file_names,
                "expected_folder_names": expected_folder_names,
                "file_metrics": file_metrics,
                "folder_metrics": folder_metrics,
                "ranking": ranking,
                "metrics": metrics,
            }
        )

        print("Expected folders:", expected_folder_names)
        # Print query result
    

        print(
            f"\n[{test_number}/"
            f"{len(BENCHMARK_SET)}]"
        )

        print(f"Query: {query}")

        if expected_adas:
            print("Expected ADA:", "," .join(expected_adas))

        if expected_source_ids:
            print("Expected source_id:", "," .join(expected_source_ids))

        if expected_folder_names:
            print("Expected folder names:", "," .join(expected_folder_names))

        print("\nRetrieved ranking:")

        expected_documents = set(expected_adas) | set(expected_source_ids)  | set(expected_file_names)

        for rank, result in enumerate(ranking, start=1):
            marker = "  <-- CORRECT" if result["document_id"] in expected_documents else ""

            print(
                f"  {rank}. "
                f"{result['document_id']} | "
                f"{result['source']} | "
                f"{marker}"
            )

        if metrics["rank"] is None:
            print(
                "\nResult: NOT FOUND "
                "in top 5"
            )
        else:
            print(
                f"\nResult: found at "
                f"rank {metrics['rank']}"
            )

        print(
            f"RR: {metrics['rr']:.3f}"
        )

    # Aggregate metrics

    number_of_queries = len(
        BENCHMARK_SET
    )

    results = {
        "retriever": retriever_name,
        "hit@1": (total_hit_1/ number_of_queries),
        "hit@3": (total_hit_3/ number_of_queries),
        "hit@5": (total_hit_5/ number_of_queries),
        "mrr": (total_rr/ number_of_queries),
        "queries": query_results,
        "file_hit@1": total_file_hit_1 / number_of_queries,
        "file_hit@3": total_file_hit_3 / number_of_queries,
        "file_hit@5": total_file_hit_5 / number_of_queries,
        "file_mrr": total_file_rr / number_of_queries,
        "folder_hit@1": total_folder_hit_1 / folder_query_count if folder_query_count else None,
        "folder_hit@3": total_folder_hit_3 / folder_query_count if folder_query_count else None,
        "folder_hit@5": total_folder_hit_5 / folder_query_count if folder_query_count else None,
        "folder_mrr": total_folder_rr / folder_query_count if folder_query_count else None,
            }

    print(
        f"RESULTS FOR "
        f"{retriever_name.upper()}"
    )

    print(f"Hit@1: {results['hit@1']:.3f}")

    print(f"Hit@3: {results['hit@3']:.3f}")

    print(
        f"Hit@5: {results['hit@5']:.3f}")

    print(f"MRR: {results['mrr']:.3f}")

    print(f"File Hit@1: {results['file_hit@1']:.3f}")

    print(f"File Hit@3: {results['file_hit@3']:.3f}")

    print(f"File Hit@5: {results['file_hit@5']:.3f}")

    print(f"File MRR:   {results['file_mrr']:.3f}")

    if results["folder_hit@1"] is not None:
        print(f"Folder Hit@1: {results['folder_hit@1']:.3f}")

        print(f"Folder Hit@3: {results['folder_hit@3']:.3f}")

        print(f"Folder Hit@5: {results['folder_hit@5']:.3f}")

        print(f"Folder MRR:   {results['folder_mrr']:.3f}")

    return results



# Full benchmark


def run_benchmark():
    """
    Evaluate MiniLM, BGE-M3 and Ensemble and print
    a comparative retrieval table.
    """

    all_results = {}

    for retriever_name in RETRIEVERS:
        all_results[
            retriever_name
        ] = evaluate_retriever(
            retriever_name
        )

    print("\n\n")
    print("FINAL RETRIEVAL BENCHMARK")

    print(
        f"{'Retriever':<15}"
        f"{'Hit@1':<12}"
        f"{'Hit@3':<12}"
        f"{'Hit@5':<12}"
        f"{'MRR':<12}"
    )


    for retriever_name in RETRIEVERS:
        result = all_results[
            retriever_name
        ]

        print(
            f"{retriever_name:<15}"
            f"{result['hit@1']:<12.3f}"
            f"{result['hit@3']:<12.3f}"
            f"{result['hit@5']:<12.3f}"
            f"{result['mrr']:<12.3f}"
        )

        RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"\nResults saved to: "
        f"{RESULTS_FILE}"
    )

if __name__ == "__main__":
    run_benchmark()