from model import load_llm
from retriever import load_retriever
from agent import build_agent

if __name__ == "__main__":

    print("Διάλεξε μοντέλο:")
    print("1. Llama 3.2 3B")
    print("2. Qwen3 14B")
    print("3. GPT-4.1 Mini")
    print("4. Gemini 2.5 Flash")
    print("5. Claude Haiku 4.5")

    choice = input("Επιλογή [Enter = Llama]: ").strip()

    model_map = {
        "": "llama",
        "2": "qwen",
        "3": "gpt41_mini",
        "4": "gemini_flash",
        "5": "claude_haiku",
}

    model_key = model_map.get(choice, "llama")

    llm = load_llm(model_key)

   #για επιλογή και φόρτωση retriever

    print("Διάλεξε retriever:")
    print("1. MiniLM")
    print("2. BGE-M3 (προτεινόμενο)")
    print("3. Ensemble MiniLM and BGE-M3")
    

    choice = input("Επιλογή [Enter = BGE]: ").strip() 
    retriever_map = {
        "1": "drive_minilm",
        "2": "drive_bge",
        "3": "drive_ensemble",
    }

    retriever_mode = retriever_map.get(choice, "drive_bge")
        
    retriever = load_retriever(retriever_mode)  
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

        
       