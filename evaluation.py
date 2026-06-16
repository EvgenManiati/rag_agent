"""
evaluation.py
Συγκριτική αξιολόγηση RAG agent με το RAG Triad (RAGAS):
- Faithfulness
- Answer Relevancy
- Context Precision
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from model import load_model
from retriever import load_retriever
from agent import build_agent
from ragas.llms import LangchainLLMWrapper

#Test questions 
# Αντικατέστησε με ερωτήσεις από τα δικά σου ΦΕΚ
TEST_DATA = [
    {
        "question": "Τι χρειάζεται για ταξιδιωτικά έξοδα;",
        "ground_truth": "Αίτηση 5 εργάσιμες μέρες πριν"
    },
    {
        "question": "Πόσες μέρες άδεια χρειάζονται ιατρική γνωμάτευση;",
        "ground_truth": "Άνω των 3 ημερών"
    },
    {
        "question": "Τι απαιτείται για προμήθεια άνω των 1000 ευρώ;",
        "ground_truth": "Έγκριση από τον προϊστάμενο"
    },
]

#Evaluation function
def run_evaluation(app, test_data: list, mode_name: str, llm) -> dict:
    print(f"Αξιολόγηση: {mode_name}")


    results = []
    for item in test_data:
        print(f"  Ερώτηση: {item['question']}")

        result = app.invoke({
            "question":   item["question"],
            "context":    "",
            "answer":     "",
            "iterations": 0
        })

        results.append({
            "question":     item["question"],
            "answer":       result["answer"],
            "contexts":     [result["context"]],
            "ground_truth": item["ground_truth"]
        })

    dataset = Dataset.from_list(results)

    ragas_llm = LangchainLLMWrapper(llm)

    scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm
    )

    print(f"\nΑποτελέσματα {mode_name}:")
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")

    return scores


#Main 
if __name__ == "__main__":

    print("Φόρτωση LLM...")
    llm = load_model()
    print("Φόρτωση μοντέλου ολοκληρώθηκε!\n")
    #αντί για γραμμή 79 βάζουμε: 
    '''
    print("Διαθέσιμα LLM models:")
    print("1. krikri")
    print("2. mistral")

    choice = input("Διάλεξε μοντέλο [Enter = krikri]: ").strip()

    if choice == "2":
        selected_model = "mistral"
    else:
        selected_model = "krikri"

    llm = load_model(selected_model)
    '''

    all_scores = {}

    for mode in ["minilm", "bge", "ensemble"]:
        print(f"\nΦόρτωση retriever ({mode})...")
        retriever = load_retriever(mode=mode)
        app = build_agent(llm, retriever)
        all_scores[mode] = run_evaluation(app, TEST_DATA, mode, llm)

    #Συγκριτικός πίνακας
    print("ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ")
    print(f"{'Μετρική':<25} {'MiniLM':>10} {'BGE-M3':>10} {'Ensemble':>10}")

    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        row = f"{metric:<25}"
        for mode in ["minilm", "bge", "ensemble"]:
            row += f" {all_scores[mode][metric]:>10.3f}"
        print(row)

    print(f"Evaluation ολοκληρώθηκε!")

