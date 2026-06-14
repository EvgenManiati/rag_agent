"""
Συγκριτική αξιολόγηση RAG agent με το RAG Triad (RAGAS):
- Faithfulness
- Answer Relevancy
- Context Precision
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from model import load_llm
from retriever import load_retriever
from agent import build_agent

# Test data for evaluation
TEST_DATA = [
    {
        "question": "Πόσες μέρες γονικής άδειας δικαιούμαι ανά έτος;",
        "ground_truth": "Κάθε γονέας ή πρόσωπο που ασκεί τη γονική μέριμνα δικαιούται 4 μήνες συνεχόμενης ή τμηματικής άδειας μέχρι τη συμπλήρωση των 8 ετών του τέκνου."
    },
    {
        "question": "Πόσες μέρες άδεια πατρότητας δικαιούμαι;",
        "ground_truth": "Κάθε εργαζόμενος πατέρας δικαιούται άδεια πατρότητας δεκατεσσάρων (14) εργάσιμων ημερών."
    },
    {
        "question": "Τι απαιτείται για προμήθεια άνω των 1000 ευρώ;",
        "ground_truth": "Έγκριση από τον προϊστάμενο"
    },
]

#Evaluation function
def run_evaluation(app, test_data: list, mode_name: str) -> dict:
    print(f"Αξιολόγηση: {mode_name}")
   

    results = []
    for item in test_data:
        print(f"Ερώτηση: {item['question']}")

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

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    print(f"\n Αποτελέσματα {mode_name}:")
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")

    return scores


# Main
if __name__ == "__main__":

    print("Φόρτωση LLM...")
    llm = load_llm()

    all_scores = {}

    for mode in ["minilm", "bge", "ensemble"]:
        print(f"\n Φόρτωση retriever ({mode})...")
        retriever = load_retriever(mode=mode)
        app = build_agent(llm, retriever)
        all_scores[mode] = run_evaluation(app, TEST_DATA, mode)

    # ── Συγκριτικός πίνακας 
    print("ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ")
    print(f"{'Μετρική':<25} {'MiniLM':>10} {'BGE-M3':>10} {'Ensemble':>10}")
    

    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        row = f"{metric:<25}"
        for mode in ["minilm", "bge", "ensemble"]:
            row += f" {all_scores[mode][metric]:>10.3f}"
        print(row)

    print(f"Evaluation ολοκληρώθηκε!")
