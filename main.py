from model import load_model
from retriever import load_retriever
from agent import build_agent

if __name__ == "__main__":
    print("Φόρτωση μοντέλου...")
    llm = load_model()

    print("Φόρτωση εγγράφων...")
    retriever = load_retriever()

    app = build_agent(llm, retriever)
    print("Agent έτοιμος!\n")

    while True:
        user_input = input("Ερώτηση (ή 'exit' για έξοδο): ")
        if user_input.lower() == "exit":
            print("Αντίο!")
            break
        result = app.invoke({
            "question":   user_input,
            "context":    "",
            "answer":     "",
            "iterations": 0
        })
        print(f"Απάντηση: {result['answer']}\n")