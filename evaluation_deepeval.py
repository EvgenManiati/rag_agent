from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import OllamaModel
from torchgen import gen
from model import load_ollama_model, load_llm
from retriever import load_retriever
from agent import build_agent


TEST_DATA = [
    {
        "question": "Πόσες μέρες άδεια μητρότητας δικαιούμαι;",
        "ground_truth": "Η άδεια μητρότητας είναι 17 εβδομάδες."
    },

    {
        "question": "Πόσες μέρες κανονική άδεια δικαιούμαι;",
        "ground_truth": "20 ημέρες σε πενθήμερο ή 24 σε εξαήμερο για πλήρη ετήσια άδεια, με αυξήσεις ανάλογα με την προϋπηρεσία."

    },

    {
        "question": "Πόσες μέρες άδεια πατρότητας δικαιούμαι;",
        "ground_truth": "Η άδεια πατρότητας είναι 14 εργάσιμες ημέρες."
    },

    {"question": "Πώς αποζημιώνομαι για την εκτός έδρας εργασία;",
     "ground_truth": "Η αποζημίωση για την εκτός έδρας μετακίνηση ισούται με ένα (1) ημερομίσθιο ή με 1/25 του νόμιμου μισθού για όσους αμείβονται με μισθό."
    }
]

class LocalLangChainLLM(DeepEvalBaseLLM):
    def __init__(self, llm, name="local-llm"):
        self.llm = llm
        self.name = name

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None) -> str:
        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            response = response.content

        return str(response)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.name


def run_model_evaluation(generator_name: str, retriever_name: str, evaluator_model):
    print(f"Evaluation Generator: {generator_name}, Retriever: {retriever_name}")

    generator_llm = load_llm(generator_name)

    retriever = load_retriever(mode=retriever_name)
    app = build_agent(generator_llm, retriever)

    metrics = [
        FaithfulnessMetric(threshold=0.5, model=evaluator_model),
        AnswerRelevancyMetric(threshold=0.5, model=evaluator_model),
        ContextualRelevancyMetric(threshold=0.5, model=evaluator_model),
        ContextualPrecisionMetric(threshold=0.5, model=evaluator_model),
        ContextualRecallMetric(threshold=0.5, model=evaluator_model),
    ]

    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_relevancy": [],
        "contextual_precision": [],
        "contextual_recall": [],
    }

    for item in TEST_DATA:
        print(f"\nΕρώτηση: {item['question']}")

        result = app.invoke({
            "question": item["question"],
            "context": "",
            "answer": "",
            "iterations": 0
        })

        answer = result.get("answer", "")
        context = result.get("context", "")

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=answer,
            expected_output=item["ground_truth"],
            retrieval_context=[context]
        )

        #print("Απάντηση:", answer)

        for metric in metrics:
            try:
                metric.measure(test_case)
                score = metric.score

                print(f"{metric.__class__.__name__}: {score}")

                if score is None:
                    continue

                if isinstance(metric, FaithfulnessMetric):
                    scores["faithfulness"].append(score)
                elif isinstance(metric, AnswerRelevancyMetric):
                    scores["answer_relevancy"].append(score)
                elif isinstance(metric, ContextualRelevancyMetric):
                    scores["contextual_relevancy"].append(score)
                elif isinstance(metric, ContextualPrecisionMetric):
                    scores["contextual_precision"].append(score)
                elif isinstance(metric, ContextualRecallMetric):
                    scores["contextual_recall"].append(score)
            except Exception as e:
                print(f"{metric.__class__.__name__} απέτυχε: {e}")

        avg_scores = {
                metric_name: (
                    sum(metric_scores) / len(metric_scores)
                    if metric_scores
                    else None
                )
                for metric_name, metric_scores in scores.items()
        }

    print(f"Μέσοι όροι: {generator_name} + {retriever_name}")


    for metric_name, score in avg_scores.items():
            if score is None:
                print(f"{metric_name}: -")
            else:
                print(f"{metric_name}: {score:.3f}")

    return avg_scores



if __name__ == "__main__":

    # Το μοντέλο που χρησιμοποιεί το DeepEval για να βαθμολογεί
    # τις απαντήσεις και τα retrieved contexts.

        print("Επιλέξτε μοντέλο για αξιολόγηση:")
        print("1. Krikri")
        print("2. Llama 3.2")
        print("3. GPT-OSS 20B")
        print("4. GPT-OSS 120B")
        print("5. Gemini Flash Lite")
        print("6. GPT-4.1 Mini")
        print("7. Gemini Flash")
        print("8. Claude Haiku")

        model_choice = input(
            "\nΕπιλογή μοντέλου [Enter = Llama]: "
        ).strip()

        model_map = {
            "1": "krikri",
            "2": "llama",
            "3": "gptoss20b",
            "4": "gptoss120b",
            "5": "gemini_flash_lite",
            "6": "gpt41_mini",
            "7": "gemini_flash",
            "8": "claude_haiku",
        }

        # Αν ο χρήστης πατήσει μόνο Enter, επιλέγεται το Llama.
        selected_generator = model_map.get(
            model_choice,
            "llama",
        )

        # Το επιλεγμένο generator θα αξιολογηθεί με όλους
        # τους διαθέσιμους retrievers.
        retrievers_to_test = [
            "minilm",
            "bge",
            "ensemble",
        ]

        all_scores = {}

        for retriever_name in retrievers_to_test:

            experiment_name = (
                f"{selected_generator}_{retriever_name}"
            )

            print(f"ΠΕΙΡΑΜΑ: {experiment_name}")

            try:
                evaluator_model = LocalLangChainLLM(
                    load_llm("gpt41_mini"),
                    name="gpt41_mini-evaluator",
                )

                all_scores[experiment_name] = (
                    run_model_evaluation(
                        generator_name=selected_generator,
                        retriever_name=retriever_name,
                        evaluator_model=evaluator_model,
                    )
                )

            except Exception as error:
                print(
                    f"\nΑπέτυχε το πείραμα "
                    f"{experiment_name}:"
                )
                print(error)

                all_scores[experiment_name] = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "contextual_relevancy": None,
                    "contextual_precision": None,
                    "contextual_recall": None,
                }

        def format_score(score):
            """
            Μετατρέπει αριθμητικό score σε κείμενο με 3 δεκαδικά.
            Αν το score λείπει, εμφανίζει παύλα αντί να κρασάρει.
            """
            if isinstance(score, (int, float)):
                return f"{score:.3f}"

            return "-"

        print("ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ")
        print(
            f"{'Experiment':<32}"
            f"{'Faithfulness':<16}"
            f"{'Answer Rel.':<16}"
            f"{'Context Rel.':<16}"
            f"{'Context Prec.':<16}"
            f"{'Context Recall':<16}"
        )

        for experiment_name, experiment_scores in all_scores.items():

            faithfulness = experiment_scores.get(
                "faithfulness"
            )

            answer_relevancy = experiment_scores.get(
                "answer_relevancy"
            )

            contextual_relevancy = experiment_scores.get(
                "contextual_relevancy"
            )

            contextual_precision = experiment_scores.get(
                "contextual_precision"
            )

            contextual_recall = experiment_scores.get(
                "contextual_recall"
            )

            print(
                f"{experiment_name:<32}"
                f"{format_score(faithfulness):<16}"
                f"{format_score(answer_relevancy):<16}"
                f"{format_score(contextual_relevancy):<16}"
                f"{format_score(contextual_precision):<16}"
                f"{format_score(contextual_recall):<16}"
            )