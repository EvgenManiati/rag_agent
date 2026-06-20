from model import load_ollama_model #load_model
from retriever import load_retriever
from agent import build_agent

if __name__ == "__main__":
    print("Διάλεξε retriever:")
    print("1. MiniLM")
    print("2. BGE-M3")
    

    choice = input("Επιλογή [Enter = BGE]: ").strip() #Krikri has been set as the default model 
    if choice == "1":
        retriever_mode = "minilm"
    else:
        retriever_mode = "bge"
        
   

    llm = load_ollama_model("llama3.2:3b", max_new_tokens=120)

    retriever = load_retriever(mode="bge")  # ή minilm/bge
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

        
       