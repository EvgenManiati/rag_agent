from retriever import load_retriever


TEST_QUERIES = [
    "σύμβαση μίσθωσης έργου",
    "πρόσκληση εκδήλωσης ενδιαφέροντος",
    "αποτελέσματα αξιολόγησης υποψηφιοτήτων",
    "προμήθεια εξοπλισμού",
    "τροποποίηση σύμβασης",
    "μετακίνηση προσωπικού",
]

RETRIEVERS = [
    "minilm",
    "bge",
    "ensemble",
]

TOP_K = 5


def print_result(
    retriever_name,
    query,
    documents,
):
    print(
        f"Retriever: {retriever_name} | "
        f"Query: {query}"
    )
    

    for index, document in enumerate(
        documents[:TOP_K],
        start=1,
    ):
        metadata = document.metadata

        print(f"\nRESULT {index}")
        

        print("ADA:", metadata.get("ada"))

        print("Source:", metadata.get("source"))


        print("Source ID:", metadata.get("source_id"))

        print("Subject:", metadata.get("subject"))

        print("Date:", metadata.get("issue_date"))

        print("Chunk ID:", metadata.get("chunk_id"))

        print("URL:", metadata.get("document_url"))

        print("\nText snippet:")

        print(document.page_content[:700])


def run_validation():
    for retriever_name in RETRIEVERS:


        print(
            f"LOADING RETRIEVER: "
            f"{retriever_name}"
        )

        retriever = load_retriever(
            retriever_name
        )

        for query in TEST_QUERIES:

            documents = retriever.invoke(
                query
            )

            print_result(
                retriever_name,
                query,
                documents,
            )


if __name__ == "__main__":
    run_validation()