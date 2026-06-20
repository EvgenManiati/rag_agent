from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import OllamaModel
from model import load_ollama_model
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


def run_model_evaluation(generator_name: str, evaluator_model):
    print(f"\n==============================")
    print(f"Evaluation Generator: {generator_name}")
    print(f"==============================")

    generator_llm = load_ollama_model(generator_name, max_new_tokens=180)

    retriever = load_retriever(mode="bge")
    app = build_agent(generator_llm, retriever)

    metrics = [
        FaithfulnessMetric(
            threshold=0.5,
            model=evaluator_model
        ),
        AnswerRelevancyMetric(
            threshold=0.5,
            model=evaluator_model
        ),
        ContextualRelevancyMetric(
            threshold=0.5,
            model=evaluator_model
        )
    ]

    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_relevancy": []
    }

    for item in TEST_DATA:
        print(f"\nΕρώτηση: {item['question']}")

        result = app.invoke({
            "question": item["question"],
            "context": "",
            "answer": "",
            "iterations": 0
        })

        answer = result["answer"]
        context = result["context"]

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=answer,
            expected_output=item["ground_truth"],
            retrieval_context=[context]
        )

        print("Απάντηση:", answer)

        for metric in metrics:
            metric.measure(test_case)

            print(f"{metric.__class__.__name__}: {metric.score}")

            if isinstance(metric, FaithfulnessMetric):
                scores["faithfulness"].append(metric.score)
            elif isinstance(metric, AnswerRelevancyMetric):
                scores["answer_relevancy"].append(metric.score)
            elif isinstance(metric, ContextualRelevancyMetric):
                scores["contextual_relevancy"].append(metric.score)

    avg_scores = {
        key: sum(values) / len(values) if values else 0
        for key, values in scores.items()
    }

    print(f"\nΜέσος όρος για {generator_name}:")
    for key, value in avg_scores.items():
        print(f"{key}: {value:.3f}")

    return avg_scores


if __name__ == "__main__":
    print("Φόρτωση evaluator LLM: Qwen")
    evaluator_llm = load_ollama_model("llama3.2:3b", max_new_tokens=80)
    evaluator_model = LocalLangChainLLM(evaluator_llm, name="llama.3.2:3b-evaluator")

    all_scores = {}

    for generator_name in ["llama3.2:3b"]:
        all_scores[generator_name] = run_model_evaluation(
            generator_name,
            evaluator_model
        )

    print("\nΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ")
    print(f"{'Metric':<25} {'Qwen':>10} {'Llama':>10}")

    for metric in ["faithfulness", "answer_relevancy", "contextual_relevancy"]:
        print(f"{metric:<25} {all_scores['qwen'][metric]:>10.3f}")