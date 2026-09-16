"""
Evaluation dataset για την αξιολόγηση του RAG agent.

Δομή:
- 20 answerable ερωτήσεις
- 8 unanswerable ερωτήσεις

Σύνολο: 28 evaluation cases
"""


UNANSWERABLE_RESPONSE = ("Δεν βρέθηκε σαφής απάντηση στις διαθέσιμες πληροφορίες.")


EVAL_DATASET = [

    # ANSWERABLE CASES


        {
        "question": "Πόσο κόστισε η προμήθεια 4 δίσκων για το έργο AdVENt;",
        "expected_answer": "967,60 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["69ΑΟ469ΗΞΩ-4ΔΥ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 2
    {
        "question": "Ποια εταιρεία ανέλαβε την προμήθεια ηλεκτρονικών ειδών για το έργο ΔΙΟΙΚΗΣΗ το 2021;",
        "expected_answer": "Creative Minds M. ΕΠΕ.",
        "expected_adas": ["Ψ57Χ469ΗΞΩ-Ν74"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 3
    {
        "question": "Ποιο ποσό εγκρίθηκε για την προμήθεια ηλεκτρονικού εξοπλισμού στο έργο MORE το 2021;",
        "expected_answer": "5.592,93 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["ΩΞ2Ε469ΗΞΩ-ΜΤ5"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 4
    {
        "question": "Τι προμηθεύτηκε το έργο Ανάπτυξη και Λειτουργία με δαπάνη 74,02 ευρώ;",
        "expected_answer": "Φαρμακευτικό υλικό.",
        "expected_adas": ["ΨΕΜΓ469ΗΞΩ-ΝΓΒ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 5
    {
        "question": "Από ποια εταιρεία έγινε η προμήθεια γραμματοσήμων για το έργο Ανάπτυξη και Λειτουργία το 2021;",
        "expected_answer": "ΕΛΛΗΝΙΚΑ ΤΑΧΥΔΡΟΜΕΙΑ Α.Ε.",
        "expected_adas": ["ΨΩΘ3469ΗΞΩ-9ΙΡ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 6
    {
        "question": "Ποιο ποσό εγκρίθηκε στο έργο Visual Facts για τη δημοσίευση επιστημονικού άρθρου;",
        "expected_answer": "2.178,00 ευρώ.",
        "expected_adas": ["ΨΒΙΓ469ΗΞΩ-Ζ3Ζ"],
        "expected_source_ids": [],
        "category": "publication",
        "answerable": True,
    },

    # 7
    {
        "question": "Για ποιο σκοπό εγκρίθηκε δαπάνη 620,00 ευρώ στο έργο NEANIAS το 2022;",
        "expected_answer": "Για την εκτύπωση προωθητικού υλικού του έργου.",
        "expected_adas": ["93ΑΗ469ΗΞΩ-ΦΝΖ"],
        "expected_source_ids": [],
        "category": "promotion",
        "answerable": True,
    },

    # 8
    {
        "question": "Ποιο ποσό εγκρίθηκε για τη φιλοξενία στο πλαίσιο της εναρκτήριας συνάντησης του έργου STELAR;",
        "expected_answer": "1.008,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["9Β8Φ469ΗΞΩ-02Σ"],
        "expected_source_ids": [],
        "category": "event",
        "answerable": True,
    },

    # 9
    {
        "question": "Ποιο ποσό εγκρίθηκε στο έργο ΑΡΧΙΜΗΔΗΣ για γραφική ύλη και είδη γραφείου;",
        "expected_answer": "1.160,50 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["Ω5ΑΡ469ΗΞΩ-4ΜΓ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 10
    {
        "question": "Τι αγοράστηκε για το έργο ERA4TB με εγκεκριμένη δαπάνη 90,00 ευρώ;",
        "expected_answer": "Ένας σκληρός δίσκος.",
        "expected_adas": ["6Ρ4Γ469ΗΞΩ-ΑΕΗ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 11
    {
        "question": "Σε ποιο συνέδριο αφορούσε η εγγραφή μέλους ΔΕΠ στο πλαίσιο του έργου LAZARUS το 2023;",
        "expected_answer": "Στο συνέδριο IEEE DAPPS 2023.",
        "expected_adas": ["6400469ΗΞΩ-9Δ0"],
        "expected_source_ids": [],
        "category": "conference",
        "answerable": True,
    },

    # 12
    {
        "question": "Ποιο ποσό εγκρίθηκε για την προμήθεια προωθητικού υλικού στο έργο EASIER;",
        "expected_answer": "50,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["980Μ469ΗΞΩ-1ΑΔ"],
        "expected_source_ids": [],
        "category": "promotion",
        "answerable": True,
    },

    # 13
    {
        "question": "Τι είδους εξοπλισμός αγοράστηκε για το έργο SciLake το 2023;",
        "expected_answer": "Μνήμη τυχαίας προσπέλασης (RAM).",
        "expected_adas": ["9Ν91469ΗΞΩ-ΠΚΨ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 14
    {
        "question": "Ποιο ποσό εγκρίθηκε για αναλώσιμα είδη Η/Υ στο έργο EDITH το 2024;",
        "expected_answer": "818,40 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["9ΒΟΩ469ΗΞΩ-ΑΤ5"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 15
    {
        "question": "Σε ποια διοργάνωση αφορούσε η εγγραφή συνεργάτη του ΙΠΣΥ στο έργο GRAPES το 2024;",
        "expected_answer": "Στο Summer School HYPATIA 2024.",
        "expected_adas": ["96Β6469ΗΞΩ-97Λ"],
        "expected_source_ids": [],
        "category": "training",
        "answerable": True,
    },

    # 16
    {
        "question": "Τι κάλυπτε η δαπάνη των 2.000,00 ευρώ στο έργο HDMS2024;",
        "expected_answer": "Την προμήθεια προωθητικού υλικού και την ενοικίαση εξοπλισμού για τη διοργάνωση του HDMS 2024.",
        "expected_adas": ["ΨΧΔ7469ΗΞΩ-122"],
        "expected_source_ids": [],
        "category": "event",
        "answerable": True,
    },

    # 17
    {
        "question": "Ποιο εργαστηριακό αναλώσιμο εγκρίθηκε για προμήθεια στο έργο ALGEBRA;",
        "expected_answer": "Κιτ για το NIR.",
        "expected_adas": ["9ΠΚΦ469ΗΞΩ-ΙΣ2"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 18
    {
        "question": "Τι αφορούσε η προμήθεια ύψους 1.016,80 ευρώ στο έργο EBRAINS 2.0 το 2025;",
        "expected_answer": "Ηλεκτρονικό εξοπλισμό και άδειες χρήσης λογισμικού.",
        "expected_adas": ["9ΒΑΨ469ΗΞΩ-ΘΜ2"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 19
    {
        "question": "Ποιος ήταν ο προμηθευτής αναλώσιμων ειδών Η/Υ για το έργο EU BabyRobot+;",
        "expected_answer": "ΠΛΑΙΣΙΟ COMPUTERS AEBE.",
        "expected_adas": ["Ρ9Α3469ΗΞΩ-1ΥΕ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 20
    {
        "question": "Ποιο ποσό εγκρίθηκε για τη δημιουργία ιστοσελίδας του έργου ENABLE 6G το 2025;",
        "expected_answer": "6.200,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["Ψ1ΕΚ469ΗΞΩ-ΤΗΝ"],
        "expected_source_ids": [],
        "category": "web_services",
        "answerable": True,
    },
    
    # UNANSWERABLE CASES
    

    # 21. Άδεια μητρότητας
    {
        "question": (
            "Πόσες μέρες άδεια μητρότητας δικαιούμαι;"
        ),
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 22. Άδεια πατρότητας
    {
        "question": (
            "Πόσες μέρες άδεια πατρότητας δικαιούμαι;"
        ),
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 23. Κανονική άδεια
    {
        "question": (
            "Πόσες μέρες κανονική άδεια δικαιούμαι;"
        ),
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 24. Άδεια ασθενείας
    {
        "question": (
            "Πόσες μέρες άδεια ασθενείας επί πληρωμή δικαιούμαι;"
        ),
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 25. Άδεια γάμου
    {
        "question": (
            "Πόσες μέρες άδεια γάμου δικαιούμαι;"
        ),
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 26. Άδεια πένθους
    {"question": "Πόσες μέρες άδεια πένθους δικαιούμαι;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [], 
     "expected_source_ids": [], 
     "category": "unanswerable", 
     "answerable": False
     },

    # 27. Άδεια αιμοδοσίας
    {"question": "Πόσες μέρες άδεια αιμοδοσίας δικαιούμαι;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [],
    "expected_source_ids": [], 
    "category": "unanswerable", 
    "answerable": False
    },

    # 28. Κυριακή
    {
        "question": (
            "Τι προσαύξηση παίρνω αν δουλέψω Κυριακή;"
        ),
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },
]

# Βοηθητικά subsets


ANSWERABLE_CASES = [
    case
    for case in EVAL_DATASET
    if case["answerable"]
]


UNANSWERABLE_CASES = [
    case
    for case in EVAL_DATASET
    if not case["answerable"]
]


# Validation


def validate_dataset():
    """
    Βασικός έλεγχος της δομής του evaluation dataset.
    """

    assert len(EVAL_DATASET) == 28, (f"Αναμένονταν 28 cases, βρέθηκαν {len(EVAL_DATASET)}.")

    assert len(ANSWERABLE_CASES) == 20, (f"Αναμένονταν 20 answerable cases, βρέθηκαν {len(ANSWERABLE_CASES)}.")

    assert len(UNANSWERABLE_CASES) == 8, (f"Αναμένονταν 8 unanswerable cases,βρέθηκαν {len(UNANSWERABLE_CASES)}.")

    required_fields = {
        "question",
        "expected_answer",
        "expected_adas",
        "expected_source_ids",
        "category",
        "answerable",
    }

    for index, case in enumerate(EVAL_DATASET, start=1):
        missing_fields = (required_fields - set(case.keys()))

        assert not missing_fields, (
            f"Case {index}: λείπουν fields {sorted(missing_fields)}")

        assert case["question"].strip(), (f"Case {index}: κενή ερώτηση.")

        assert case["expected_answer"].strip(), (f"Case {index}: κενή expected_answer.")

        if case["answerable"]:
            assert case["expected_adas"], (f"Case {index}: answerable case χωρίς expected ADA.")

        else:
            assert not case["expected_adas"], (f"Case {index}: unanswerable case με expected ADA.")

    print("RAG evaluation dataset is valid.")
    print(f"Total cases: {len(EVAL_DATASET)}")
    print(f"Answerable: {len(ANSWERABLE_CASES)}")
    print(f"Unanswerable: {len(UNANSWERABLE_CASES)}")


if __name__ == "__main__":
    validate_dataset()