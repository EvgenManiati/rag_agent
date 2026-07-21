from model import load_llm, list_available_models
from retriever import load_retriever
from agent import build_agent


TEST_DATA = [
    {
        "question": "Πόσες μέρες άδεια μητρότητας δικαιούμαι;",
        "expected_answer": "17 εβδομάδες",
        "expected_terms": ["17", "εβδομάδες"],
        "expected_source": "mitrotita.pdf"
    },
    {
        "question": "Πόσες μέρες άδεια πατρότητας δικαιούμαι;",
        "expected_answer": "14 εργάσιμες ημέρες",
        "expected_terms": ["14", "εργάσιμες"],
        "expected_source": "patrotita.pdf"
    },
    {
        "question": "Πόσες μέρες κανονική άδεια δικαιούμαι;",
        "expected_answer": "25 εργάσιμες ημέρες σε πενθήμερο ή 30 σε εξαήμερο",
        "expected_terms": ["25", "30"],
        "expected_source": "kanoniki.pdf"
    },
]

AVALAILABLE_RETRIEVERS = {
    "bge": "BGE-M3",
    "minilm": "MiniLM",
    "ensemble": "Ensemble MiniLM and BGE-M3"
}


def choose_model():
    models = list_available_models()

    print("Διάλεξε μοντέλο:")

    for i, model_key in enumerate(models.keys(), start=1):
        print(f"{i}. {models[model_key].name}")

    while True:
        choice = input("Επίλεξε μοντέλο: ")
        if (choice.isdigit() and 1 <= int(choice) <= len(models)):
            return list(models.keys())[int(choice) - 1]

        print("Μη έγκυρη επιλογή. Προσπάθησε ξανά.")


def normalize(text):
    return str(text).lower().strip()


def answer_ok(answer, expected_terms):
    answer = normalize(answer)
    return all(normalize(term) in answer for term in expected_terms)


def source_ok(context, expected_source):
    return normalize(expected_source) in normalize(context)


def run_model_evaluation(model_key, retriever_mode):

    print(f"MODEL: {model_key}")
    print(f"RETRIEVER: {retriever_mode}")

    llm = load_llm(model_key)
    retriever = load_retriever(mode=retriever_mode)
    app = build_agent(llm, retriever)

    total = len(TEST_DATA)
    answer_hits = 0
    source_hits = 0

    for i, item in enumerate(TEST_DATA, start=1):


        print("Ερώτηση:")
        print(item["question"])

        result = app.invoke({
            "question": item["question"],
            "context": "",
            "answer": "",
            "iterations": 0
        })

        answer = result.get("answer", "")
        context = result.get("context", "")

        ans_ok = answer_ok(answer, item["expected_terms"])
        src_ok = source_ok(context, item["expected_source"])

        answer_hits += int(ans_ok)
        source_hits += int(src_ok)

        print("\nΑπάντηση μοντέλου:")
        print(answer)

        print("\nΑναμενόμενη απάντηση:")
        print(item["expected_answer"])

        print("\nΑναμενόμενη πηγή:")
        print(item["expected_source"])

        print(f"\nSource OK: {src_ok}")
        print(f"Answer OK: {ans_ok}")

    source_accuracy = source_hits / total
    answer_accuracy = answer_hits / total


    print("ΑΠΟΤΕΛΕΣΜΑΤΑ")

    print(f"Model: {model_key}")
    print(f"Retriever: {retriever_mode}")

    print(f"\nSource Accuracy: {source_hits}/{total} ({source_accuracy:.2f})")
    print(f"Answer Accuracy: {answer_hits}/{total} ({answer_accuracy:.2f})")

    return {
        "model": model_key,
        "retriever": retriever_mode,
        "source_accuracy": source_accuracy,
        "answer_accuracy": answer_accuracy
    }


if __name__ == "__main__":

   model = choose_model()
   retrievers = ["bge", "minilm", "ensemble"]

   all_scores = {}

   for retriever in retrievers:
       experiment = f"{model}_{retriever}"
       
       print(f"Πείραμα: {experiment}")

       try:
           scores = run_model_evaluation(model, retriever)
           all_scores[experiment] = scores
       except Exception as e:
           print(f"Σφάλμα κατά την εκτέλεση του πειράματος {experiment}: {e}")

           print(e)

print("\nΣΥΝΟΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
print(
    f"{'Experiment':<30} ",
    f"{'Source Accuracy':<20}",
    f"{'Answer Accuracy':<20}"
)

for experiment, scores in all_scores.items():
    source_acc = scores.get("source_accuracy", 0)
    answer_acc = scores.get("answer_accuracy", 0)

    print(
        f"{experiment:<30} "
        f"{source_acc:<20.2f} "
        f"{answer_acc:<20.2f}"
    )