
from model import load_model
from retriever import load_retriever
from agent import build_retriever, AgentState


if __name__ == "__main__":

    print("Μοντέλο έτοιμο!")
    llm = load_model()

    print(f"Vector store έτοιμο! Φόρτωση αρχείων...")
    retriever = load_retriever()

    app = build_retriever(llm, retriever)
    print("Agent έτοιμος!")

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