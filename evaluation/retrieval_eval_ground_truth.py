
# RETRIEVAL GROUND-TRUTH DATASET
#
# VALIDATION_SET:
# Used for retriever configuration and ensemble-weight tuning.
#
# TEST_SET:
# Used only after all retriever parameters have been frozen.
#
# Important:
# Do not tune the ensemble weights using TEST_SET.


# VALIDATION SET
# 15 queries

VALIDATION_SET = [

    # 1. CONTRACT - ΟΡΙΖΟΝΤΙΟ ΙΠΣΥ
    # ADA: 6Θ5Β469ΗΞΩ-ΣΧΛ

    {
        "query": (
            "Ποια απόφαση αφορά τη σύναψη συμβάσεων "
            "μίσθωσης έργου στο Οριζόντιο ΙΠΣΥ;"
        ),
        "expected_adas": [
            "6Θ5Β469ΗΞΩ-ΣΧΛ"
        ],
        "category": "contract",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου τις συμβάσεις συνεργατών που "
            "συνάφθηκαν για το έργο Οριζόντιο ΙΠΣΥ."
        ),
        "expected_adas": [
            "6Θ5Β469ΗΞΩ-ΣΧΛ"
        ],
        "category": "contract",
        "difficulty": "natural",
    },


    # 2. INVITATION - ARIA
    # ADA: 9Ζ87469ΗΞΩ-ΕΩΟ


    {
        "query": (
            "Ποια απόφαση αφορά την πρόσκληση "
            "ILSP.293.ARIA-0421;"
        ),
        "expected_adas": [
            "9Ζ87469ΗΞΩ-ΕΩΟ"
        ],
        "category": "invitation",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου την πρόσκληση εκδήλωσης "
            "ενδιαφέροντος που σχετίζεται με το ARIA."
        ),
        "expected_adas": [
            "9Ζ87469ΗΞΩ-ΕΩΟ"
        ],
        "category": "invitation",
        "difficulty": "natural",
    },

    # 3. RESULTS - AutoFAIR
    # ADA: ΡΖΑΟ469ΗΞΩ-ΒΤΑ

    {
        "query": (
            "Ποια απόφαση περιέχει τα αποτελέσματα "
            "της πρόσκλησης AutoFAIR_upotr_022023;"
        ),
        "expected_adas": [
            "ΡΖΑΟ469ΗΞΩ-ΒΤΑ"
        ],
        "category": "results",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου τα αποτελέσματα αξιολόγησης "
            "υποψηφίων για την υποτροφία AutoFAIR."
        ),
        "expected_adas": [
            "ΡΖΑΟ469ΗΞΩ-ΒΤΑ"
        ],
        "category": "results",
        "difficulty": "natural",
    },


    # 4. PROCUREMENT - SMS-CBA
    # ADA: ΩΤΑΜ469ΗΞΩ-ΤΛ3

    {
        "query": (
            "Ποια απόφαση αφορά την προμήθεια "
            "ηλεκτρονικού εξοπλισμού για το SMS-CBA;"
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3"
        ],
        "category": "procurement",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου την αγορά εξοπλισμού και ειδικού "
            "λογισμικού που έγινε για το έργο SMS-CBA."
        ),
        "expected_adas": [
            "ΩΤΑΜ469ΗΞΩ-ΤΛ3"
        ],
        "category": "procurement",
        "difficulty": "natural",
    },


    # 5. CONTRACT MODIFICATION - TRUSTEE
    # ADA: 67ΖΙ469ΗΞΩ-1ΧΧ

    {
        "query": (
            "Ποια απόφαση τροποποίησε σύμβαση "
            "μίσθωσης έργου στο TRUSTEE;"
        ),
        "expected_adas": [
            "67ΖΙ469ΗΞΩ-1ΧΧ"
        ],
        "category": "contract_modification",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου την αλλαγή που έγινε σε σύμβαση "
            "συνεργάτη του έργου TRUSTEE."
        ),
        "expected_adas": [
            "67ΖΙ469ΗΞΩ-1ΧΧ"
        ],
        "category": "contract_modification",
        "difficulty": "natural",
    },


    # 6. TRAVEL - BEHAVE
    # ADA: Ψ640469ΗΞΩ-30Ο

    {
        "query": (
            "Ποια απόφαση εγκρίνει μετακίνηση "
            "επιστημονικού συνεργάτη για το BEHAVE;"
        ),
        "expected_adas": [
            "Ψ640469ΗΞΩ-30Ο"
        ],
        "category": "travel",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου την απόφαση για ταξίδι "
            "συνεργάτη του έργου BEHAVE."
        ),
        "expected_adas": [
            "Ψ640469ΗΞΩ-30Ο"
        ],
        "category": "travel",
        "difficulty": "natural",
    },


    # 7. INVITATION - ΑΡΧΙΜΗΔΗΣ
    # ADA: 6ΠΙΟ469ΗΞΩ-1ΦΛ

    {
        "query": (
            "Ποια απόφαση αφορά δημοσίευση πρόσκλησης "
            "εκδήλωσης ενδιαφέροντος για το έργο ΑΡΧΙΜΗΔΗΣ;"
        ),
        "expected_adas": [
            "6ΠΙΟ469ΗΞΩ-1ΦΛ"
        ],
        "category": "invitation",
        "difficulty": "semantic",
    },


    # 8. INVITATION - ENVISION II
    # ADA: 6ΙΧΒ469ΗΞΩ-ΕΤΦ

    {
        "query": (
            "Βρες μου την πρόσκληση ENVISION II "
            "με κωδικό 001.2023."
        ),
        "expected_adas": [
            "6ΙΧΒ469ΗΞΩ-ΕΤΦ"
        ],
        "category": "invitation",
        "difficulty": "semantic",
    },

    # 9. CONTRACT MODIFICATION - ΑΠΤΟΣ
    # ADA: 6ΛΔΦ469ΗΞΩ-Λ9Δ

    {
        "query": (
            "Ποια απόφαση αφορά τροποποίηση συμβάσεων "
            "μίσθωσης έργου στο έργο ΑΠΤΟΣ;"
        ),
        "expected_adas": [
            "6ΛΔΦ469ΗΞΩ-Λ9Δ"
        ],
        "category": "contract_modification",
        "difficulty": "semantic",
    },
]


# FINAL TEST SET
#
# 15 queries
# Different relevant documents from VALIDATION_SET.
#
# DO NOT use this set for ensemble-weight tuning.

TEST_SET = [

    # 1. INVITATION - ΟΡΙΖΟΝΤΙΟ ΙΠΣΥ
    # ADA: ΨΨ7Α469ΗΞΩ-Ι9Ο

    {
        "query": (
            "Ποια απόφαση δημοσίευσε πρόσκληση "
            "για συνεργάτη στο Οριζόντιο ΙΠΣΥ;"
        ),
        "expected_adas": [
            "ΨΨ7Α469ΗΞΩ-Ι9Ο"
        ],
        "category": "invitation",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου την πρόσκληση για νέο συνεργάτη "
            "στο Οριζόντιο έργο ΙΠΣΥ."
        ),
        "expected_adas": [
            "ΨΨ7Α469ΗΞΩ-Ι9Ο"
        ],
        "category": "invitation",
        "difficulty": "natural",
    },


    # 2. RESULTS - ΠΕΡΙΠΛΟΥΣ
    # ADA: ΡΕΒΜ469ΗΞΩ-7Φ1

    {
        "query": (
            "Ποια απόφαση αφορά τα αποτελέσματα "
            "της πρόσκλησης του έργου ΠΕΡΙΠΛΟΥΣ;"
        ),
        "expected_adas": [
            "ΡΕΒΜ469ΗΞΩ-7Φ1"
        ],
        "category": "results",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου τα αποτελέσματα της πρόσκλησης "
            "ILSP.356.PER-07.ΧΑΝ.0423."
        ),
        "expected_adas": [
            "ΡΕΒΜ469ΗΞΩ-7Φ1"
        ],
        "category": "results",
        "difficulty": "natural",
    },


    # 3. INVITATION - CEI BOOST
    # ADA: 6Β3Λ469ΗΞΩ-Ξ1Ε

    {
        "query": (
            "Ποια απόφαση αφορά την πρόσκληση "
            "CEI BOOST 002.2023;"
        ),
        "expected_adas": [
            "6Β3Λ469ΗΞΩ-Ξ1Ε"
        ],
        "category": "invitation",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου τη δημοσιευμένη πρόσκληση "
            "για το CEI BOOST."
        ),
        "expected_adas": [
            "6Β3Λ469ΗΞΩ-Ξ1Ε"
        ],
        "category": "invitation",
        "difficulty": "natural",
    },


    # 4. SCHOLARSHIP AGREEMENT - AutoFAIR
    # ADA: 6ΒΥ8469ΗΞΩ-Κ0Τ

    {
        "query": (
            "Ποια απόφαση αφορά σύναψη συμφωνητικού "
            "χορήγησης υποτροφίας στο AutoFAIR;"
        ),
        "expected_adas": [
            "6ΒΥ8469ΗΞΩ-Κ0Τ"
        ],
        "category": "scholarship",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου τη σύμβαση υποτροφίας που "
            "υπογράφηκε στο πλαίσιο του AutoFAIR."
        ),
        "expected_adas": [
            "6ΒΥ8469ΗΞΩ-Κ0Τ"
        ],
        "category": "scholarship",
        "difficulty": "natural",
    },


    # 5. RESULTS - AutoFAIR 122022
    # ADA: 9ΖΝ0469ΗΞΩ-Ο7Μ

    {
        "query": (
            "Ποια απόφαση περιέχει τα αποτελέσματα "
            "της πρόσκλησης AutoFAIR_122022;"
        ),
        "expected_adas": [
            "9ΖΝ0469ΗΞΩ-Ο7Μ"
        ],
        "category": "results",
        "difficulty": "semantic",
    },

    {
        "query": (
            "Βρες μου ποια απόφαση δημοσίευσε "
            "τα αποτελέσματα του AutoFAIR τον Δεκέμβριο."
        ),
        "expected_adas": [
            "9ΖΝ0469ΗΞΩ-Ο7Μ"
        ],
        "category": "results",
        "difficulty": "natural",
    },


    # 6. EMPLOYMENT - Βοΐσκα / ΑΠΤΟΣ / ΑΠΟΗΧΟΙ
    # ADA: ΨΕΤ0469ΗΞΩ-ΤΒΑ

    {
        "query": (
            "Ποια απόφαση αφορά σύμβαση εργασίας "
            "ορισμένου χρόνου για τα έργα "
            "Βοΐσκα, ΑΠΤΟΣ και ΑΠΟΗΧΟΙ;"
        ),
        "expected_adas": [
            "ΨΕΤ0469ΗΞΩ-ΤΒΑ"
        ],
        "category": "employment",
        "difficulty": "semantic",
    },


    # 7. PROCUREMENT EXPENSE - EOSCFuture
    # ADA: ΡΒ14469ΗΞΩ-9ΩΣ


    {
        "query": (
            "Βρες μου την απόφαση έγκρισης δαπάνης "
            "για ηλεκτρονικό εξοπλισμό στο EOSCFuture."
        ),
        "expected_adas": [
            "ΡΒ14469ΗΞΩ-9ΩΣ"
        ],
        "category": "procurement",
        "difficulty": "natural",
    },


    # 8. PAYMENT - INTRACOM 1 - ART
    # ADA: 6ΜΓ8469ΗΞΩ-6Τ6


    {
        "query": (
            "Ποια απόφαση αφορά εντολή πληρωμής "
            "για εξοπλισμό και αναλώσιμα στο "
            "INTRACOM 1 - ART;"
        ),
        "expected_adas": [
            "6ΜΓ8469ΗΞΩ-6Τ6"
        ],
        "category": "payment",
        "difficulty": "semantic",
    },


    # 9. TRAVEL - OpenGPT-X/ARC
    # ADA: 6ΛΟΘ469ΗΞΩ-ΓΧ3

    {
        "query": (
            "Βρες μου την έγκριση μετακίνησης "
            "συνεργάτη στο πλαίσιο του OpenGPT-X/ARC."
        ),
        "expected_adas": [
            "6ΛΟΘ469ΗΞΩ-ΓΧ3"
        ],
        "category": "travel",
        "difficulty": "natural",
    },


    # 10. TRAVEL - EASIER
    # ADA: 69Ζ2469ΗΞΩ-ΜΗΖ

    {
        "query": (
            "Ποια απόφαση αφορά μετακίνηση "
            "δύο συνεργατών στο έργο EASIER;"
        ),
        "expected_adas": [
            "69Ζ2469ΗΞΩ-ΜΗΖ"
        ],
        "category": "travel",
        "difficulty": "semantic",
    },

    # 11. RESEARCH ETHICS - ΕΗΔΕ ΕΚ ΑΘΗΝΑ
    # source_id: athena_ehde_regulation

    {
        "query": "Από πόσα τακτικά μέλη αποτελείται η ΕΗΔΕ του Ερευνητικού Κέντρου Αθηνά;",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "difficulty": "natural",
    },

    {
        "query": "Πόσα από τα μέλη της ΕΗΔΕ πρέπει να είναι εκτός του ΕΚ Αθηνά;",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "difficulty": "natural",
    },

    {
        "query": "Κάθε πότε συνεδριάζει κανονικά η ΕΗΔΕ;",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "difficulty": "natural",
    },

    {
        "query": "Πόσα μέλη πρέπει να είναι παρόντα για να υπάρχει απαρτία στην ΕΗΔΕ;",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "difficulty": "natural",
    },

    {
        "query": "Σε πόσες μέρες πρέπει να αποφασίσει η ΕΗΔΕ για μια αίτηση;",
        "expected_adas": [],
        "expected_source_ids": ["athena_ehde_regulation"],
        "category": "research_ethics",
        "difficulty": "natural",
    },

    # 12. ΕΛΚΕ ΕΚΠΑ - RECRUITMENT / EMPLOYMENT
    # source_id: ekpa_funding_guide_2024

    {
        "query": "Πόσες μέρες έχω για να κάνω αίτηση σε δημόσια πρόσκληση του ΕΛΚΕ;",
        "expected_adas": [],
        "expected_source_ids": ["ekpa_funding_guide_2024"],
        "category": "recruitment",
        "difficulty": "natural",
    },

    {
        "query": "Πόσες μέρες έχω για να κάνω ένσταση στα αποτελέσματα μιας πρόσκλησης του ΕΛΚΕ;",
        "expected_adas": [],
        "expected_source_ids": ["ekpa_funding_guide_2024"],
        "category": "recruitment",
        "difficulty": "natural",
    },

    {
        "query": "Πόσες ώρες θεωρούνται ένα έτος πλήρους απασχόλησης;",
        "expected_adas": [],
        "expected_source_ids": ["ekpa_funding_guide_2024"],
        "category": "employment",
        "difficulty": "natural",
    },

    {
        "query": "Μέχρι πόσες ώρες υπερωρία μπορώ να κάνω τον χρόνο σε έργα που χρηματοδοτούνται από ιδιωτικούς, διεθνείς ή ίδιους πόρους;",
        "expected_adas": [],
        "expected_source_ids": ["ekpa_funding_guide_2024"],
        "category": "employment",
        "difficulty": "natural",
    },

]