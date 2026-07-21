import streamlit as st

from agent import build_agent
from model import load_llm
from retriever import load_retriever

# Βασικές ρυθμίσεις σελίδας


st.set_page_config(
    page_title="ΙΕΛ RAG Assistant",
    page_icon="🤖",
    layout="wide",
)
st.markdown(
    """
    <style>
    /* 
       Κύρια χρώματα, βασισμένα στην εταιρική ταυτότητα ΙΕΛ
    */

    :root {
        --iel-primary: #10069F;
        --iel-primary-dark: #0B0476;
        --iel-primary-light: #EAE9F8;

        --iel-text: #16394A;
        --iel-text-muted: #62727B;

        --iel-accent-red: #ED1C2E;
        --iel-logo-blue: #005477;

        --iel-background: #FFFFFF;
        --iel-surface: #F5F5FA;
        --iel-border: #D9D9E8;
    }

    /* 
       Βασική εφαρμογή
    */

    html,
    body,
    [class*="css"],
    .stApp {
        font-family: Arial, Helvetica, sans-serif;
        color: var(--iel-text);
    }

    .stApp {
        background-color: var(--iel-background);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* 
       Τίτλοι
     */

    h1 {
        color: var(--iel-primary);
        font-size: 2.45rem;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }

    h2 {
        color: var(--iel-primary);
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    h3 {
        color: var(--iel-logo-blue);
        font-weight: 700;
    }

    p,
    label,
    .stMarkdown {
        color: var(--iel-text);
    }

    /*
       Sidebar
    */

    [data-testid="stSidebar"] {
        background-color: var(--iel-primary);
        border-right: none;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.25);
    }

    /* Selectboxes στο sidebar */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.45);
        border-radius: 4px;
        color: var(--iel-text);
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: var(--iel-text);
    }

    /*
       Κουμπιά
    */

    .stButton > button {
        width: 100%;
        min-height: 2.8rem;

        background-color: var(--iel-primary);
        color: #FFFFFF;

        border: 2px solid var(--iel-primary);
        border-radius: 3px;

        font-weight: 700;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        background-color: var(--iel-primary-dark);
        border-color: var(--iel-primary-dark);
        color: #FFFFFF;
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(16, 6, 159, 0.18);
    }

    /* Κόκκινα κουμπιά ρυθμίσεων στο sidebar */
[data-testid="stSidebar"] .stButton > button {
    background-color: var(--iel-accent-red);
    color: #FFFFFF;
    border: 2px solid var(--iel-accent-red);
    border-radius: 3px;
    font-weight: 700;
    min-height: 2.8rem;
    transition: all 0.18s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #C91525;
    color: #FFFFFF;
    border-color: #C91525;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stButton > button:focus {
    color: #FFFFFF;
    border-color: var(--iel-accent-red);
    box-shadow: 0 0 0 3px rgba(237, 28, 46, 0.22);
}

[data-testid="stSidebar"] .stButton > button:active {
    background-color: #A9101D;
    color: #FFFFFF;
    border-color: #A9101D;
    transform: translateY(0);
}
       Chat input
    */

    [data-testid="stChatInput"] {
        background-color: #FFFFFF;
        border: 2px solid var(--iel-primary);
        border-radius: 4px;
        box-shadow: none;
    }

    [data-testid="stChatInput"]:focus-within {
        box-shadow: 0 0 0 3px rgba(16, 6, 159, 0.13);
    }

    /* 
       Chat messages
    */

    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid var(--iel-border);
        border-left: 5px solid var(--iel-primary);
        border-radius: 2px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
        box-shadow: none;
    }

    /* Πρώτο chat message style μπορεί να είναι user/assistant,
       ανάλογα με την έκδοση του Streamlit. */
    [data-testid="stChatMessage"] p {
        color: var(--iel-text);
        line-height: 1.65;
    }

    /* 
       Alerts
    */

    [data-testid="stAlert"] {
        border-radius: 3px;
        border-left-width: 5px;
        box-shadow: none;
    }

    /* 
       Expanders / retrieved context
    */

    [data-testid="stExpander"] {
        background-color: var(--iel-surface);
        border: 1px solid var(--iel-border);
        border-radius: 3px;
    }

    [data-testid="stExpander"] summary {
        color: var(--iel-primary);
        font-weight: 700;
    }

    /* 
       Horizontal rule
    */

    hr {
        border: none;
        border-top: 2px solid var(--iel-primary);
        opacity: 0.16;
        margin: 1.5rem 0;
    }

    /*
       Μικρό κόκκινο accent
        */

    .iel-red-line {
        width: 72px;
        height: 5px;
        background-color: var(--iel-accent-red);
        margin: 0.6rem 0 1.4rem 0;
    }

    /* 
    /*
   Hero section
*/

.iel-hero {
    background: #10069F;
    padding: 45px 55px;
    margin-bottom: 35px;
    position: relative;
    overflow: hidden;
}

.iel-hero::after{
    content:"";
    position:absolute;
    left:-5%;
    bottom:-30px;
    width:110%;
    height:70px;
    background:white;
    transform:rotate(-4deg);
}

.iel-small-title{
    color:#D6D6E8;
    font-size:17px;
    font-weight:600;
    margin-bottom:15px;
}

.iel-big-title{
    color:white;
    font-size:48px;
    font-weight:800;
    line-height:1.15;
    max-width:850px;
    position:relative;
    z-index:2;
}

.iel-description{
    color:white;
    opacity:0.9;
    margin-top:20px;
    max-width:750px;
    font-size:18px;
    line-height:1.6;
    position:relative;
    z-index:2;
}

    /*
       Κατάσταση ενεργού agent
     */

    .iel-status-card {
        background-color: var(--iel-primary-light);
        border-left: 5px solid var(--iel-primary);
        padding: 0.85rem 1rem;
        margin-bottom: 1.2rem;
        color: var(--iel-text);
        font-weight: 600;
    }

    /* 
       Απόκρυψη Streamlit branding
       Προαιρετικό
    */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    
    </style>


    """,
    unsafe_allow_html=True,
)

# Τίτλος εφαρμογής


st.markdown(
    """
<div class="iel-hero">


<div class="iel-big-title">
ΙΕΛ's Intelligent Assistant
</div>

<div class="iel-description">
Agentic Retrieval-Augmented Generation (RAG) σύστημα
για αναζήτηση, ανάκτηση και παραγωγή απαντήσεων από
εταιρικά έγγραφα με χρήση LangGraph και Μεγάλων
Γλωσσικών Μοντέλων (LLMs).
</div>

</div>
""",
unsafe_allow_html=True
)

st.caption(
    "Agentic RAG assistant με Google Drive, LangGraph, "
    "πολλαπλά LLMs και πολλαπλούς retrievers."
)

# Βοηθητικές συναρτήσεις


def extract_answer(raw_answer) -> str:
    """
    Μετατρέπει την απάντηση οποιουδήποτε model provider
    σε απλό string.

    Τα OpenRouter wrappers επιστρέφουν συνήθως string,
    ενώ το ChatOllama μπορεί να επιστρέψει AIMessage.
    """

    if hasattr(raw_answer, "content"):
        return str(raw_answer.content)

    return str(raw_answer)


@st.cache_resource(show_spinner=False)
def create_agent(model_key: str, retriever_mode: str):
    """
    Φορτώνει το επιλεγμένο LLM και τον retriever
    και δημιουργεί τον LangGraph agent.

    Το cache_resource αποτρέπει την επαναφόρτωση
    των μοντέλων και των vector stores σε κάθε UI rerun.
    """

    llm = load_llm(model_key)
    retriever = load_retriever(mode=retriever_mode)

    return build_agent(llm, retriever)


# Αρχικοποίηση session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "active_configuration" not in st.session_state:
    st.session_state.active_configuration = None


# Sidebar: ρυθμίσεις του agent

with st.sidebar:
    st.header("Ρυθμίσεις")

    model_options = {
        "Krikri — Hugging Face": "krikri",
        "Llama 3.2 - Ollama": "llama",
        "GPT-OSS 20B — OpenRouter" : "gptoss20b",
        "GPT-OSS 120B — OpenRouter": "gptoss120b",
        "Gemini 2.5 Flash Lite — OpenRouter": "gemini_flash_lite",
        "GPT-4.1 Mini — OpenRouter": "gpt41_mini",
        "Gemini 2.5 Flash — OpenRouter": "gemini_flash",
        "Claude Haiku — OpenRouter": "claude_haiku",
    }

    selected_model_label = st.selectbox(
        "Μοντέλο",
        options=list(model_options.keys()),
        index=0,
    )

    retriever_options = {
        "MiniLM": "minilm",
        "BGE-M3": "bge",
        "Ensemble": "ensemble",
    }

    selected_retriever_label = st.selectbox(
        "Retriever",
        options=list(retriever_options.keys()),
        index=1,
    )

    show_context = st.checkbox(
        "Εμφάνιση retrieved context",
        value=False,
    )

    initialize_clicked = st.button(
        "Φόρτωση agent",
        type="primary",
        use_container_width=True,
    )

    clear_clicked = st.button(
        "Καθαρισμός συνομιλίας",
        use_container_width=True,
    )


# Καθαρισμός ιστορικού


if clear_clicked:
    st.session_state.messages = []
    st.rerun()

# Δημιουργία / αλλαγή agent


selected_model_key = model_options[selected_model_label]
selected_retriever_mode = retriever_options[selected_retriever_label]

selected_configuration = (
    selected_model_key,
    selected_retriever_mode,
)

if initialize_clicked:
    try:
        with st.spinner(
            "Φόρτωση μοντέλου, εγγράφων και retriever..."
        ):
            st.session_state.agent = create_agent(
                model_key=selected_model_key,
                retriever_mode=selected_retriever_mode,
            )

            st.session_state.active_configuration = (
                selected_configuration
            )

        st.success(
            f"Ο agent φορτώθηκε με "
            f"{selected_model_label} και "
            f"{selected_retriever_label}."
        )

    except Exception as error:
        st.session_state.agent = None
        st.exception(error)


# Εμφάνιση ενεργής ρύθμισης


if st.session_state.active_configuration:
    active_model, active_retriever = (
        st.session_state.active_configuration
    )

    st.info(
        f"Ενεργό μοντέλο: `{active_model}`  |  "
        f"Retriever: `{active_retriever}`"
    )
else:
    st.warning(
        "Επίλεξε μοντέλο και retriever και πάτησε "
        "«Φόρτωση agent»."
    )


# Εμφάνιση ιστορικού συνομιλίας


for message in st.session_state.messages:

    if message["role"] == "user":
        avatar = "assets/user_avatar.png"
    else:
        avatar = "assets/assistant_avatar.png"

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_context
            and message.get("context")
        ):
            with st.expander("Retrieved context"):
                st.text(message["context"])

        if (
            message["role"] == "assistant"
            and show_context
            and message.get("context")
        ):
            with st.expander("Retrieved context"):
                st.text(message["context"])


# Πεδίο ερώτησης


user_question = st.chat_input(
    "Γράψε την ερώτησή σου..."
)


# Εκτέλεση του RAG agent

if user_question:
    if st.session_state.agent is None:
        st.error(
            "Πρέπει πρώτα να φορτώσεις τον agent από το αριστερό μενού."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message(
    "user",
    avatar="👤"):
        st.markdown(user_question)

    with st.chat_message(
    "assistant",
    avatar="💬"):
        try:
            with st.spinner("Αναζήτηση στα έγγραφα..."):
                result = st.session_state.agent.invoke(
                    {
                        "question": user_question,
                        "context": "",
                        "answer": "",
                        "iterations": 0,
                    }
                )

            answer = extract_answer(
                result.get("answer", "")
            )

            context = str(
                result.get("context", "")
            )

            st.markdown(answer)

            if show_context and context:
                with st.expander("Retrieved context"):
                    st.text(context)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "context": context,
                }
            )

        except Exception as error:
            st.error(
                "Παρουσιάστηκε σφάλμα κατά την παραγωγή της απάντησης."
            )
            st.exception(error)
