from model import load_model
from retriever import load_retriever
from agent import build_agent

if __name__ == "__main__":
    print("Διάλεξε μοντέλο:")
    print("1. Krikri")
    print("2. Qwen")

    choice = input("Επιλογή [Enter = Qwen]: ").strip() #Krikri has been set as the default model 
    if choice == "1":
        selected_model = "krikri"
    elif choice == "2":
        selected_model = "qwen"
    else:
        selected_model = "qwen"

    llm = load_model(selected_model)

    retriever = load_retriever(mode="ensemble")  # ή minilm/bge
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