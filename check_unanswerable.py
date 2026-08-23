from retriever import load_retriever
from evaluation.rag_eval_dataset import EVAL_DATASET


retriever = load_retriever("bge")

unanswerable_cases = [
    item
    for item in EVAL_DATASET
    if item["answerable"] is False
]

print(f"\nUnanswerable test cases: {len(unanswerable_cases)}")


for index, item in enumerate(unanswerable_cases, start=1):

    question = item["question"]

    print("\n" + "=" * 100)
    print(f"UNANSWERABLE TEST {index}/{len(unanswerable_cases)}")
    print(f"QUESTION: {question}")
    print("=" * 100)

    documents = retriever.invoke(question)

    for rank, document in enumerate(documents[:5], start=1):

        metadata = document.metadata

        print(f"\n--- RESULT {rank} ---")
        print("source:", metadata.get("source"))
        print("source_id:", metadata.get("source_id"))
        print("ada:", metadata.get("ada"))
        print("subject:", metadata.get("subject"))

        print("\nTEXT:")
        print(document.page_content[:1200])

    print()