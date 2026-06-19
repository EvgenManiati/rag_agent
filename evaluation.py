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
from langchain_huggingface import HuggingFaceEmbeddings
from config import MINILM_MODEL

#Test questions 

TEST_DATA = [
    {
    
        "question": "Πόσες μέρες αναρρωτικής άδειας δικαιούμαι;",
        "ground_truth": "Mέχρι τρεις (3) ημέρες, οι εργαζόμενοι δικαιούνται να λάβουν το 1/2 του ημερησίου μισθού τους για κάθε μέρα ασθενείας."
    }
    ]
  
#Evaluation function
def run_evaluation(app, test_data: list, mode_name: str, evaluator_llm) -> dict:
    print(f"Αξιολόγηση: {mode_name}")


    results = []
    test_items = list(test_data)[:1]  # Limit to the first item for testing

    for item in test_items:
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
    
    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    ragas_embeddings = HuggingFaceEmbeddings(model_name=MINILM_MODEL)

    scores = evaluate(
    dataset,
    metrics=[faithfulness], #, answer_relevancy, context_precision],
    llm= ragas_llm,
    embeddings=ragas_embeddings
    )

    print(f"\nΑποτελέσματα {mode_name}:")
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}")
    #print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    #print(f"  Context Precision: {scores['context_precision']:.3f}")

    return scores


#Main evaluation 

if __name__ == "__main__":

    # a dict to store the scores for each model
    all_scores = {}

    # main retriever mode
    retriever_mode = "ensemble"

  
    print("Φόρτωση evaluator LLM: Qwen")
    evaluator_llm = load_model("qwen") #loading the evaluator model (Krikri) once to avoid double loading


    print(f"Φόρτωση retriever: {retriever_mode}")
    retriever = load_retriever(mode=retriever_mode)

    # compare the two models: Krikri and TinyLlama

    for model_name in ["krikri", "qwen"]:

        print(f"Evaluation Generator: {model_name}")
        
        
        if model_name == "krikri":
            generator_llm = evaluator_llm #loading the same model for evaluation and generation to avoid double loading
        else:
            print(f"Φόρτωση generator: {model_name}")
            generator_llm = load_model(model_name)

       
        app = build_agent(
            generator_llm,
            retriever
        )

        all_scores[model_name] = run_evaluation(
            app,
            TEST_DATA,
            model_name,
            evaluator_llm
        )

    #comparative table of scores

    print("ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ ΜΟΝΤΕΛΩΝ")
    print(
        f"{'Μετρική':<25}"
        f"{'Krikri':>12}"
        f"{'Qwen':>12}"
    )

    for metric in [
        "faithfulness",
        #"answer_relevancy",
        #"context_precision"
    ]:

        row = f"{metric:<25}"
        #row += f"{all_scores['krikri'].get(metric, float('nan')):>12.3f}"
        #row += f"{all_scores['qwen'].get(metric, float('nan')):>12.3f}"

        row += f"{all_scores['krikri'][metric]:>12.3f}"
        row += f"{all_scores['qwen'][metric]:>12.3f}"

        print(row)

    print("\nEvaluation ολοκληρώθηκε!")

'''δεν τρέχει με το κρικρι γιατί είναι πολύ μεγάλο και η μνήμη δεν επαρκεί. 
επίσης δεν τρέχει ούτε το tinyllama γιατί δεν μπορεί να κάνει σωστό ragas 
και βγάζει failed to parse output. returning none'''



  '''
    {
        "question": "Πόσες μέρες κανονική άδεια δικαιούμαι;",
        "ground_truth": "Η αναλογία της χορηγούμενης άδειας υπολογίζεται βάσει ετήσιας αδείας 20 εργασίμων ημερών επί πενθημέρου εβδομαδιαίας εργασίας."
     },
    {
        "question": "Πώς αποζημιώνομαι για την εκτός έδρας εργασία;",
        "ground_truth": ". Η αποζημίωση για την εκτός έδρας μετακίνηση ισούται με ένα (1) ημερομίσθιο ή με 1/25 του νόμιμου μισθού για όσους αμείβονται με μισθό."
    },'''
