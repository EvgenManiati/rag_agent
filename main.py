#from model import load_ollama_model, load_model, load_openrouter_model
#from retriever import load_retriever
#from agent import build_agent

from model import load_llm, list_available_models
from retriever import load_retriever
from agent import build_agent

if __name__ == "__main__":
    models = list_available_models()

    print("Διάλεξε μοντέλο:")
    print("1. Krikri")
    print("2. Llama 3.2 3B")
    print("3. GPT-OSS 20B")
    print("4. GPT-OSS 120B")
    print("5. Gemini 2.5 Flash Lite")
    print("6. GPT-4.1 Mini")
    print("7. Gemini 2.5 Flash")
    print("8. Claude Haiku 4.5")

    choice = input("Επιλογή [Enter = Llama]: ").strip()

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

    model_key = model_map.get(choice, "llama")

    llm = load_llm(model_key)

   #για επιλογή και φόρτωση retriever

    print("Διάλεξε retriever:")
    print("1. MiniLM")
    print("2. BGE-M3")
    print("3. Ensemble MiniLM and BGE-M3")
    

    choice = input("Επιλογή [Enter = BGE]: ").strip() #Krikri has been set as the default model 
    if choice == "1":
        retriever_mode = "minilm"
    elif choice == "2":
        retriever_mode = "bge"
    elif choice == "3":
        retriever_mode = "ensemble"
    else:
        retriever_mode = "bge"
        
   

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
            "question":   user_input,
            "context":    "",
            "answer":     "",
            "iterations": 0
        })
        print(f"Απάντηση: {result['answer']}\n")

        
       