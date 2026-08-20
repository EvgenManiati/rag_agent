from retriever import load_retriever
from evaluation.retrieval_eval_ground_truth import (
    VALIDATION_SET, TEST_SET,
)

import json
from pathlib import Path

RESULTS_DIRECTORY = Path(
    "data/evaluation"
)

RESULTS_FILE = (
    RESULTS_DIRECTORY
    / "retrieval_benchmark.json"
)


RETRIEVERS = [
    "minilm",
    "bge",
    "ensemble",
]

BENCHMARK_SET = TEST_SET  # Use TEST_SET for evaluation

# Ζητάμε περισσότερα chunks από όσα τελικά αξιολογούμε,
# επειδή μπορεί πολλά chunks να ανήκουν στον ίδιο ΑΔΑ.
#RAW_RETRIEVAL_K = 15

# Αξιολογούμε μέχρι τους πρώτους 5 μοναδικούς ΑΔΑ.
EVALUATION_K = 5


# Helpers


def get_unique_document_ranking(documents, max_results=5):
    """
    Convert retrieved chunks into a ranking of unique source documents.

    Diavgeia documents are identified by ADA.
    External PDFs are identified by source_id.
    """

    unique_results = []
    seen_documents = set()

    for document in documents:
        ada = str(document.metadata.get("ada") or "").strip()
        source_id = str(document.metadata.get("source_id") or "").strip()

        document_id = ada or source_id

        if not document_id:
            continue

        if document_id in seen_documents:
            continue

        seen_documents.add(document_id)

        unique_results.append({
            "document_id": document_id,
            "ada": ada or None,
            "source_id": source_id or None,
            "source": document.metadata.get("source"),
            "subject": document.metadata.get("subject"),
            "issue_date": document.metadata.get("issue_date"),
            "chunk_id": document.metadata.get("chunk_id"),
        })

        if len(unique_results) >= max_results:
            break

    return unique_results

def find_first_relevant_rank(ranking, expected_adas=None, expected_source_ids=None):
    """Return the rank of the first relevant source document."""

    expected_adas = expected_adas or []
    expected_source_ids = expected_source_ids or []

    expected_documents = set(expected_adas) | set(expected_source_ids)

    for rank, result in enumerate(ranking, start=1):
        if result["document_id"] in expected_documents:
            return rank

    return None


def calculate_query_metrics(ranking, expected_adas=None, expected_source_ids=None):
    """Calculate Hit@1, Hit@3, Hit@5 and Reciprocal Rank for one query."""

    rank = find_first_relevant_rank(
        ranking,
        expected_adas=expected_adas,
        expected_source_ids=expected_source_ids,
    )

    if rank is None:
        return {
            "rank": None,
            "hit@1": 0,
            "hit@3": 0,
            "hit@5": 0,
            "rr": 0.0,
        }

    return {
        "rank": rank,
        "hit@1": int(rank <= 1),
        "hit@3": int(rank <= 3),
        "hit@5": int(rank <= 5),
        "rr": 1.0 / rank,
    }


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

    total_hit_1 = 0
    total_hit_3 = 0
    total_hit_5 = 0
    total_rr = 0.0

    query_results = []

    for test_number, test_case in enumerate(
        BENCHMARK_SET,
        start=1,
    ):
        query = test_case["query"]
        expected_adas = test_case.get("expected_adas", [])
        expected_source_ids = test_case.get("expected_source_ids", [])


        # Retrieve chunks.
        documents = retriever.invoke(
            query
        )

        # Convert chunk results into unique ADA ranking.
        ranking = get_unique_document_ranking(
            documents,
            max_results=EVALUATION_K,
        )

        metrics = calculate_query_metrics(
            ranking,
            expected_adas= expected_adas,
            expected_source_ids= expected_source_ids
        )

        total_hit_1 += metrics["hit@1"]
        total_hit_3 += metrics["hit@3"]
        total_hit_5 += metrics["hit@5"]
        total_rr += metrics["rr"]

        query_results.append(
            {
                "query": query,
                "expected_adas": expected_adas,
                "expected_source_ids" : expected_source_ids,
                "ranking": ranking,
                "metrics": metrics,
            }
        )

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

        print("\nRetrieved ranking:")

        expected_documents = set(expected_adas) | set(expected_source_ids)

        for rank, result in enumerate(ranking, start=1):
            marker = "  <-- CORRECT" if result["document_id"] in expected_documents else ""

            print(
                f"  {rank}. "
                f"{result['document_id']} | "
                f"{result['source']} | "
                f"{result['subject']}"
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
        "hit@1": (
            total_hit_1
            / number_of_queries
        ),
        "hit@3": (
            total_hit_3
            / number_of_queries
        ),
        "hit@5": (
            total_hit_5
            / number_of_queries
        ),
        "mrr": (
            total_rr
            / number_of_queries
        ),
        "queries": query_results,
    }

    print(
        f"RESULTS FOR "
        f"{retriever_name.upper()}"
    )

    print(
        f"Hit@1: "
        f"{results['hit@1']:.3f}"
    )

    print(
        f"Hit@3: "
        f"{results['hit@3']:.3f}"
    )

    print(
        f"Hit@5: "
        f"{results['hit@5']:.3f}"
    )

    print(
        f"MRR:   "
        f"{results['mrr']:.3f}"
    )

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