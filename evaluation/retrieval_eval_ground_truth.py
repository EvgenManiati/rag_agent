"""
Retrieval ground-truth dataset.

VALIDATION_SET:
Used for retriever configuration and ensemble-weight tuning.

TEST_SET:
Used only after retriever parameters have been frozen.

The Diavgeia cases are based on documents contained in the
new curated corpus.

Queries avoid using specific natural persons as retrieval keys.
"""


# VALIDATION SET

VALIDATION_SET = [

    # DIAVGEIA

    # 1. GeCoInt
    {
        "query": (
            "Ποια απόφαση αφορά ανατροπή ποσού 5.952 ευρώ "
            "για δαπάνες προσωπικού του έργου GeCoInt;"
        ),
        "expected_adas": ["6Ξ82469ΗΞΩ-ΘΛΜ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "budget",
        "difficulty": "natural",
    },

    # 2. LOCARD
    {
        "query": (
            "Ποια απόφαση αφορά σύναψη ιδιωτικού συμφωνητικού "
            "στο έργο LOCARD για εργασίες ανάπτυξης λογισμικού, "
            "ασφάλειας, ιδιωτικότητας και blockchains;"
        ),
        "expected_adas": ["ΕΒΟ9469ΗΞΩ-ΙΤΕ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "contract",
        "difficulty": "semantic",
    },

    # 3. 2023 payment
    {
        "query": (
            "Ποια απόφαση αφορά οριστικοποίηση πληρωμής ποσού "
            "1.475,39 ευρώ για έξοδα μετακίνησης στη Σουηδία;"
        ),

        "expected_adas": ["6ΠΜΛ469ΗΞΩ-ΚΕΦ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "payment",
        "difficulty": "natural",
    },

    # 4. PREFERRED
    {
        "query": (
            "Ποια απόφαση αφορά σύναψη σύμβασης στο έργο PREFERRED "
             "για ανάπτυξη τεχνολογιών πρόβλεψης και πρόληψης πυρκαγιών;"
        ),

        "expected_adas": ["ΨΑΧ1469ΗΞΩ-ΓΧΣ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "contract",
        "difficulty": "semantic",
    },

    # 5. Kahoot / ΕπιSTEAMουσική
    {
        "query": (
            "Ποια απόφαση αφορά πληρωμή 133,92 ευρώ "
            "για ανανέωση συνδρομής στο Kahoot στο έργο "
            "ΕπιSTEAMουσική;"
        ),
        "expected_adas": ["9Ε53469ΗΞΩ-ΤΛΠ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "payment",
        "difficulty": "natural",
    },

    # 6. Remote work
    {
        "query": (
        "Ποια απόφαση αφορά χορήγηση άδειας τηλεργασίας "
        "για 44 ημέρες κατά το Α΄ τρίμηνο του 2025;"
        ),

        "expected_adas": ["68ΓΑ469ΗΞΩ-8Ν9"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "employment",
        "difficulty": "semantic",
    },

    # 7. Electricity bill payment
    {
        "query": (
            "Ποια απόφαση αφορά οριστικοποίηση πληρωμής ποσού "
            "262 ευρώ για λογαριασμό ηλεκτρικής ενέργειας;"
        ),

        "expected_adas": ["ΨΑΣΜ469ΗΞΩ-ΗΩΛ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "payment",
        "difficulty": "natural",
    },
    # 8. contracts
    {
    "query": (
        "Ποια απόφαση αφορά τη σύναψη συμφωνητικών χορήγησης "
        "υποτροφίας στο έργο IntelComp το 2022;"
      ),
    "expected_adas": ["9ΡΔ0469ΗΞΩ-ΒΗΛ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "payment",
    "difficulty": "natural",
    },

    # 9. Archimedes
    {
    "query": (
        "Ποια απόφαση εγκρίνει δαπάνη 1.160,50 ευρώ "
        "για γραφική ύλη και είδη γραφείου στο έργο ΑΡΧΙΜΗΔΗΣ;"
    ),
    "expected_adas": ["Ω5ΑΡ469ΗΞΩ-4ΜΓ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "payment",
    "difficulty": "natural",
},

    # 10. SI_CLUSTER-2
    {
        "query":(
        "Ποια απόφαση αφορά εντολή πληρωμής ποσού 2.199,76 ευρώ "
        "στο έργο SI_CLUSTER-2 για έξοδα σχεδιασμού και ανάπτυξης του site;"
        ),
        "expected_adas": ["6ΝΩΓ469ΗΞΩ-ΜΙΛ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "payment",
        "difficulty": "natural",
    },

    # EXTERNAL


    # 11. Research ethics
    {
        "query": (
            "Από πόσα τακτικά μέλη αποτελείται η ΕΗΔΕ "
            "του Ερευνητικού Κέντρου Αθηνά;"
        ),
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "expected_file_names": ["kanonismos_ehde_athina.pdf"],
        "category": "research_ethics",
        "difficulty": "natural",
    },

    # 12. Research ethics
    {
        "query": (
            "Σε πόσες ημέρες πρέπει να αποφασίσει "
            "η ΕΗΔΕ για μια αίτηση;"
        ),
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "expected_file_names": ["kanonismos_ehde_athina.pdf"],
        "category": "research_ethics",
        "difficulty": "semantic",
    },
]



# TEST SET

TEST_SET = [

    # DIAVGEIA

    # 1. ARCHIMEDES equipment
    {
        "query": (
            "Ποια διακήρυξη αφορά προμήθεια ηλεκτρονικού "
            "εξοπλισμού για τις ανάγκες της Μονάδας ΑΡΧΙΜΗΔΗΣ;"
        ),
        "expected_adas": ["65Κ0469ΗΞΩ-0ΜΤ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "procurement",
        "difficulty": "natural",
    },

    # 2. 2025 utility obligation
    {
        "query": (
            "Ποια απόφαση του Ιουνίου 2025 αφορά ανάληψη "
            "υποχρέωσης για λογαριασμό παροχής με ποσό "
            "366 ευρώ;"
        ),
        "expected_adas": ["ΨΡΔ1469ΗΞΩ-1ΗΜ"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "budget",
        "difficulty": "semantic",
    },

    # 3. 2022 assignment
    {
        "query": (
            "Ποια απόφαση του Αυγούστου 2022 αφορά "
            "ανάθεση έργου ή υπηρεσίας στο Ερευνητικό "
            "Κέντρο Αθηνά;"
        ),
        "expected_adas": ["ΨΙΜΩ469ΗΞΩ-ΩΣ6"],
        "expected_source_ids": [],
        "expected_file_names": [],
        "category": "procurement",
        "difficulty": "semantic",
    },

    # 4. MAST 
{
    "query": (
        "Ποια απόφαση εγκρίνει δαπάνη 400 ευρώ στο έργο MAST "
        "για προωθητικό υλικό στο πλαίσιο συνάντησης Matchmaking event;"
    ),
    "expected_adas": ["9ΚΓ0469ΗΞΩ-ΠΩΝ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "expense",
    "difficulty": "natural",
},

# 5. Slack for XMANAI 
{
    "query": (
        "Ποια απόφαση εγκρίνει δαπάνη 407,57 ευρώ στο έργο XMANAI "
        "για ετήσια άδεια χρήσης του λογισμικού Slack;"
    ),
    "expected_adas": ["ΩΤΗΕ469ΗΞΩ-ΙΙΦ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "expense",
    "difficulty": "natural",
},

# 6. Archimedes catering
{
    "query": (
        "Ποια απόφαση εγκρίνει δαπάνη 210 ευρώ στο έργο ΑΡΧΙΜΗΔΗΣ "
        "για catering στο Open Day ARCHIMEDES;"
    ),
    "expected_adas": ["6Ε9Ζ469ΗΞΩ-4ΚΓ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "expense",
    "difficulty": "natural",
},

# 7. Transition to 8 
{
    "query": (
        "Ποια απόφαση αφορά δέσμευση ποσού 34.242,37 ευρώ "
        "για το έργο Transition to 8 για το οικονομικό έτος 2023;"
    ),
    "expected_adas": ["6ΞΖΕ469ΗΞΩ-ΘΝΣ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "commitment",
    "difficulty": "natural",
},

# 8. SciLake evaluation results
{
    "query": (
        "Ποια απόφαση αφορά τη δημοσίευση αποτελεσμάτων αξιολόγησης "
        "υποψηφίων για χορήγηση υποτροφίας στο έργο SciLake;"
    ),
    "expected_adas": ["9Κ0Ε469ΗΞΩ-ΤΡ0"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "evaluation_results",
    "difficulty": "natural",
},

# 9. GeCoInt - semantic formulation
{
    "query": (
        "Ποια διακήρυξη αφορά την προμήθεια ρομποτικών συστημάτων "
        "και περιλαμβάνει υποβρύχιο ρομποτικό βραχίονα "
        "εκτιμώμενης αξίας 20.354,84 ευρώ χωρίς ΦΠΑ;"
    ),
    "expected_adas": ["98ΘΦ469ΗΞΩ-ΧΚ3"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "procurement",
    "difficulty": "natural",
},
# 10. DT4GS payment
{
    "query": (
        "Ποια απόφαση αφορά εντολή πληρωμής ποσού 399,92 ευρώ "
        "στο έργο DT4GS - Open collaboration and open Digital Twin "
        "infrastructure for Green Smart Shipping;"
    ),
    "expected_adas": ["62ΚΞ469ΗΞΩ-ΝΥΣ"],
    "expected_source_ids": [],
    "expected_file_names": [],
    "category": "payment",
    "difficulty": "natural",
},

    # EXTERNAL


    # 11. EKPA funding guide
    {
        "query": (
            "Πόσες ημέρες έχει κάποιος για να υποβάλει "
            "ένσταση στα αποτελέσματα δημόσιας πρόσκλησης "
            "του ΕΛΚΕ;"
        ),
        "expected_adas": [],
        "expected_source_ids": ["ekpa_funding_guide_2024"],
        "expected_file_names": ["odigos_xrimatodotisis_ekpa_2024.pdf"],
        "category": "recruitment",
        "difficulty": "natural",
    },

    # 12. EKPA funding guide
    {
        "query": (
            "Πόσες ώρες θεωρούνται ένα έτος "
            "πλήρους απασχόλησης;"
        ),
        "expected_adas": [],
        "expected_source_ids": ["ekpa_funding_guide_2024"],
        "expected_file_names": ["odigos_xrimatodotisis_ekpa_2024.pdf"],
        "category": "employment",
        "difficulty": "natural",
    },
]


ALL_RETRIEVAL_CASES = VALIDATION_SET + TEST_SET