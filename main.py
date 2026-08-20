from model import load_llm, list_available_models
from retriever import load_retriever
from agent import build_agent

if __name__ == "__main__":
    models = list_available_models()

    print("Διάλεξε μοντέλο:")
    print("1. Krikri")
    print("2. Llama 3.2 3B")
    print("3. Qwen3 14B")
    print("4. GPT-4.1 Mini")
    print("5. Gemini 2.5 Flash")
    print("6. Claude Haiku 4.5")

    choice = input("Επιλογή [Enter = Llama]: ").strip()

    model_map = {
        "1": "krikri",
        "2": "llama",
        "3": "qwen",
        "4": "gpt41_mini",
        "5": "gemini_flash",
        "6": "claude_haiku",
}

    model_key = model_map.get(choice, "llama")

    llm = load_llm(model_key)

   #για επιλογή και φόρτωση retriever

    print("Διάλεξε retriever:")
    print("1. MiniLM")
    print("2. BGE-M3")
    print("3. Ensemble MiniLM and BGE-M3")
    

    choice = input("Επιλογή [Enter = BGE]: ").strip() 
    retriever_map = {
        "1": "minilm",
        "2": "bge",
        "3": "ensemble",
    }

    retriever_mode = retriever_map.get(
        choice,
        "bge",
    )
        
   

    #llm = load_ollama_model("llama3.2:3b", max_new_tokens=120)

    retriever = load_retriever(retriever_mode)  # ή minilm/bge/ensemble
    app = build_agent(llm, retriever)
    print("Agent έτοιμος!\n")

    while True:
        user_input = input("Ερώτηση (ή 'exit' για έξοδο): ")
        if user_input.lower() == "exit":
            print("Αντίο!")
            break

        result = app.invoke({
            "question": user_input,
            "context": "",
            "answer": "",
            "iterations": 0,
            "sources": [],
        })
        print(f"Απάντηση: {result['answer']}\n")

        
       