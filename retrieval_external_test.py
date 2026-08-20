from retriever import load_retriever


QUESTIONS = [
    "Από πόσα τακτικά μέλη αποτελείται η ΕΗΔΕ του Ερευνητικού Κέντρου Αθηνά;"
    #"Πόσες ημερολογιακές ημέρες είναι η ελάχιστη προθεσμία υποβολής αιτήσεων σε δημόσια πρόσκληση του ΕΛΚΕ;",
]


for retriever_name in ["minilm", "bge", "ensemble"]:

    print(f"RETRIEVER: {retriever_name}")


    retriever = load_retriever(retriever_name)

    for question in QUESTIONS:

        print(f"\nΕρώτηση: {question}")

        docs = retriever.invoke(question)

        for index, doc in enumerate(docs[:5], start=1):

            print(f"\n--- RESULT {index} ---")
            print("source:", doc.metadata.get("source"))
            print("source_id:", doc.metadata.get("source_id"))
            print("ada:", doc.metadata.get("ada"))
            print("subject:", doc.metadata.get("subject"))
            print(doc.page_content[:500])