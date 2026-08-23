"""
RAG evaluation dataset.

Το αρχείο περιέχει φυσικές ερωτήσεις χρηστών για την
αξιολόγηση του end-to-end RAG pipeline.

Κάθε test case περιλαμβάνει:

- question:
    Η ερώτηση που θα σταλεί στον agent.

- expected_answer:
    Η αναμενόμενη απάντηση που χρησιμοποιεί το DeepEval
    ως reference / ground truth.

- expected_adas:
    Οι ΑΔΑ των αποφάσεων που περιέχουν την πληροφορία.
    Χρησιμοποιούνται για ανάλυση και debugging.

- category:
    Η θεματική κατηγορία της ερώτησης.

- answerable:
    True όταν η απάντηση υπάρχει στο dataset.
    False όταν θέλουμε να ελέγξουμε αν το RAG αποφεύγει
    hallucination όταν η πληροφορία δεν υπάρχει.
"""


# ANSWERABLE TEST CASES

NATURAL_TEST_CASES = [

    # SMS-CBA

    {
        "question": (
            "Πόσο κόστισε ο ηλεκτρονικός εξοπλισμός "
            "και το λογισμικό για το SMS-CBA;"
        ),
        "expected_answer": (
            "17.371,24 Ευρώ, πλέον ΦΠΑ."
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3",
        ],
        "category": "procurement",
        "answerable": True,
    },

    {
        "question": (
            "Ποια εταιρεία ανέλαβε την προμήθεια "
            "για το SMS-CBA;"
        ),
        "expected_answer": (
            "COSMOS BUSINESS SYSTEMS AEBE."
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3",
        ],
        "category": "procurement",
        "answerable": True,
    },

    {
        "question": (
            "Τι αγοράστηκε για το έργο SMS-CBA;"
        ),
        "expected_answer": (
            "Ηλεκτρονικός εξοπλισμός "
            "και ειδικό λογισμικό."
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3",
        ],
        "category": "procurement",
        "answerable": True,
    },

    {
        "question": (
            "Μέχρι πότε διαρκούσε η σύμβαση "
            "προμήθειας για το SMS-CBA;"
        ),
        "expected_answer": (
            "Μέχρι 30/08/2021."
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3",
        ],
        "category": "procurement",
        "answerable": True,
    },

    {
        "question": (
            "Πότε ξεκινούσε η σύμβαση προμήθειας "
            "για το SMS-CBA;"
        ),
        "expected_answer": (
            "Στις 15/07/2021."
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3",
        ],
        "category": "procurement",
        "answerable": True,
    },


    # BEHAVE

    {
        "question": (
            "Πόσες μέρες θα διαρκούσε η μετακίνηση "
            "του συνεργάτη για το BEHAVE;"
        ),
        "expected_answer": (
            "36 ημέρες."
        ),
        "expected_adas": [
            "Ψ640469ΗΞΩ-30Ο",
        ],
        "category": "travel",
        "answerable": True,
    },

    {
        "question": (
            "Πού θα ταξίδευε ο συνεργάτης "
            "για το έργο BEHAVE;"
        ),
        "expected_answer": (
            "Από την Αθήνα στο Λος Άντζελες."
        ),
        "expected_adas": [
            "Ψ640469ΗΞΩ-30Ο",
        ],
        "category": "travel",
        "answerable": True,
    },

    {
        "question": (
            "Από ποια πόλη θα ξεκινούσε "
            "η μετακίνηση για το BEHAVE;"
        ),
        "expected_answer": (
            "Από την Αθήνα."
        ),
        "expected_adas": [
            "Ψ640469ΗΞΩ-30Ο",
        ],
        "category": "travel",
        "answerable": True,
    },


    # TRUSTEE

    {
        "question": (
            "Τι άλλαξε στη σύμβαση "
            "του έργου TRUSTEE;"
        ),
        "expected_answer": (
            "Τροποποιήθηκε το οικονομικό "
            "αντικείμενο της σύμβασης."
        ),
        "expected_adas": [
            "67ΖΙ469ΗΞΩ-1ΧΧ",
        ],
        "category": "contract_modification",
        "answerable": True,
    },


    # AutoFAIR

    {
        "question": (
            "Με τι αντικείμενο σχετιζόταν "
            "η υποτροφία AutoFAIR;"
        ),
        "expected_answer": (
            "Με έρευνα στον χώρο της δικαιοσύνης "
            "και της επεξηγησιμότητας αλγορίθμων "
            "μηχανικής μάθησης."
        ),
        "expected_adas": [
            "ΡΖΑΟ469ΗΞΩ-ΒΤΑ",
        ],
        "category": "scholarship",
        "answerable": True,
    },


    # ΟΡΙΖΟΝΤΙΟ ΙΠΣΥ

    {
        "question": (
            "Πόσο ήταν το συνολικό κόστος "
            "της συνεργασίας της Αικατερίνης "
            "στο Οριζόντιο ΙΠΣΥ;"
        ),
        "expected_answer": (
            "7.350,00 Ευρώ."
        ),
        "expected_adas": [
            "6Θ5Β469ΗΞΩ-ΣΧΛ",
        ],
        "category": "contract",
        "answerable": True,
    },

    {
        "question": (
            "Πόσο ήταν το συνολικό κόστος "
            "της συνεργασίας της Αντωνίας "
            "στο Οριζόντιο ΙΠΣΥ;"
        ),
        "expected_answer": (
            "9.990,00 Ευρώ."
        ),
        "expected_adas": [
            "6Θ5Β469ΗΞΩ-ΣΧΛ",
        ],
        "category": "contract",
        "answerable": True,
    },


    # ARIA

    {
        "question": (
            "Πόσα μόρια μπορεί να δώσει "
            "η συνέντευξη στην πρόσκληση ARIA;"
        ),
        "expected_answer": (
            "Από 0 έως 10 μόρια."
        ),
        "expected_adas": [
            "9Ζ87469ΗΞΩ-ΕΩΟ",
        ],
        "category": "recruitment",
        "answerable": True,
    },

    {
        "question": (
            "Ποια είναι η μέγιστη συνολική "
            "βαθμολογία στην αξιολόγηση ARIA;"
        ),
        "expected_answer": (
            "100 μόρια."
        ),
        "expected_adas": [
            "9Ζ87469ΗΞΩ-ΕΩΟ",
        ],
        "category": "recruitment",
        "answerable": True,
    },

# RESEARCH ETHICS - ΕΗΔΕ ΕΚ ΑΘΗΝΑ

    {
        "question": "Από πόσα τακτικά μέλη αποτελείται η ΕΗΔΕ του Ερευνητικού Κέντρου Αθηνά;",
        "expected_answer": "Η ΕΗΔΕ του ΕΚ Αθηνά αποτελείται από πέντε (5) τακτικά μέλη και τους αναπληρωτές τους.",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "answerable": True,
    },

    {
        "question": "Πόσα από τα μέλη της ΕΗΔΕ πρέπει να είναι εκτός του ΕΚ Αθηνά;",
        "expected_answer": "Τουλάχιστον δύο (2) από τα μέλη της ΕΗΔΕ πρέπει να είναι πρόσωπα εκτός του ΕΚ Αθηνά.",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "answerable": True,
    },

    {
        "question": "Κάθε πότε συνεδριάζει κανονικά η ΕΗΔΕ;",
        "expected_answer": "Η ΕΗΔΕ συνεδριάζει τακτικά μία (1) φορά τον μήνα.",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "answerable": True,
    },

    {
        "question": "Πόσα μέλη πρέπει να είναι παρόντα για να υπάρχει απαρτία στην ΕΗΔΕ;",
        "expected_answer": "Για να υπάρχει απαρτία πρέπει να είναι παρόντα τουλάχιστον τρία (3) μέλη, συμπεριλαμβανομένου του Προέδρου ή του Αντιπροέδρου και ενός (1) μέλους που δεν ανήκει στο ΕΚ Αθηνά.",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "answerable": True,
    },

    {
        "question": "Σε πόσες μέρες πρέπει να αποφασίσει η ΕΗΔΕ για μια αίτηση;",
        "expected_answer": "Η ΕΗΔΕ πρέπει να αποφασίσει μέσα σε χρονικό διάστημα που δεν μπορεί να υπερβαίνει τις δεκαπέντε (15) ημέρες από την υποβολή της αίτησης και τη συγκέντρωση όλων των απαραίτητων συνοδευτικών εγγράφων.",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "answerable": True,
    },
]


# UNANSWERABLE TEST CASES
#
# Αυτά ελέγχουν αν το RAG αποφεύγει να επινοήσει απάντηση
# όταν η ζητούμενη πληροφορία δεν υπάρχει στο corpus.
#
# Η expected_answer είναι ίδια με το fallback που έχουμε
# ορίσει στο prompt του agent.

UNANSWERABLE_TEST_CASES = [

    {
        "question": (
            "Πόσες μέρες άδεια μητρότητας δικαιούμαι;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "leave",
        "answerable": False,
    },

    {
        "question": (
            "Πόσες μέρες άδεια πατρότητας δικαιούμαι;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "leave",
        "answerable": False,
    },

    {
        "question": (
            "Πόσες μέρες κανονική άδεια δικαιούμαι;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "leave",
        "answerable": False,
    },

    {
        "question": (
            "Πόσες μέρες άδεια ασθενείας "
            "επί πληρωμή δικαιούμαι;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "leave",
        "answerable": False,
    },

    {
        "question": (
            "Πόσες μέρες άδεια γάμου δικαιούμαι;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "leave",
        "answerable": False,
    },

    {
        "question": (
            "Πόσο διάλειμμα δικαιούμαι "
            "κατά τη διάρκεια της εργασίας μου;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "employment",
        "answerable": False,
    },

{
    "question": "Τι επίδομα τηλεργασίας δικαιούται ένας εργαζόμενος;",
    "expected_answer": ("Δεν υπάρχει σχετική "
                "πληροφορία στα διαθέσιμα έγγραφα."
                ),
    "expected_adas": [],
    "expected_source_ids": [],
    "category": "employment",
    "answerable": False,
},

    {
        "question": (
            "Τι προσαύξηση παίρνω αν δουλέψω Κυριακή;"
        ),
        "expected_answer": (
            "Δεν βρέθηκε σαφής απάντηση στις "
            "διαθέσιμες πληροφορίες."
        ),
        "expected_adas": [],
        "category": "employment",
        "answerable": False,
    },
]


# COMPLETE DATASET

EVAL_DATASET = (
    NATURAL_TEST_CASES
    + UNANSWERABLE_TEST_CASES
)


# OPTIONAL HELPERS

ANSWERABLE_DATASET = [
    case
    for case in EVAL_DATASET
    if case["answerable"]
]


UNANSWERABLE_DATASET = [
    case
    for case in EVAL_DATASET
    if not case["answerable"]
]


# SIMPLE VALIDATION

def validate_eval_dataset(dataset):
    """
    Perform basic validation of the RAG evaluation dataset.

    Raises ValueError if a test case does not contain
    the expected fields.
    """

    required_fields = {
        "question",
        "expected_answer",
        "expected_adas",
        "category",
        "answerable",
    }

    for index, case in enumerate(dataset, start=1):
        missing_fields = required_fields - case.keys()

        if missing_fields:
            raise ValueError(
                f"Test case {index}: missing fields {missing_fields}"
            )

        if not isinstance(case["expected_adas"], list):
            raise ValueError(
                f"Test case {index}: expected_adas must be a list."
            )

        if "expected_source_ids" in case and not isinstance(case["expected_source_ids"], list):
            raise ValueError(
                f"Test case {index}: expected_source_ids must be a list."
            )
    for index, case in enumerate(
        EVAL_DATASET,
        start=1,
    ):
        missing_fields = (
            required_fields
            - set(case.keys())
        )

        if missing_fields:
            raise ValueError(
                f"Test case {index} is missing fields: "
                f"{missing_fields}"
            )

        if not case["question"].strip():
            raise ValueError(
                f"Test case {index} has an empty question."
            )

        if not case[
            "expected_answer"
        ].strip():
            raise ValueError(
                f"Test case {index} has an empty "
                f"expected answer."
            )

        if not isinstance(
            case["expected_adas"],
            list,
        ):
            raise ValueError(
                f"Test case {index}: expected_adas "
                f"must be a list."
            )


# Validate automatically when imported.
validate_eval_dataset(EVAL_DATASET)