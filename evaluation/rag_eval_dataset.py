"""
Evaluation dataset για την αξιολόγηση του RAG agent αποκλειστικά πάνω σε έγγραφα της Διαύγειας.

Δομή:
- 39 answerable ερωτήσεις
- 11 unanswerable ερωτήσεις

Σύνολο: 50 evaluation cases
"""


UNANSWERABLE_RESPONSE = ("Δεν βρέθηκε σαφής απάντηση στις διαθέσιμες πληροφορίες.")


EVAL_DATASET = [

    # ANSWERABLE CASES

    # 1. AdVENt - κόστος προμήθειας δίσκων

        {
        "question": "Πόσο κόστισε η προμήθεια 4 δίσκων για το έργο AdVENt;",
        "expected_answer": "967,60 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["69ΑΟ469ΗΞΩ-4ΔΥ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 2. ΔΙΟΙΚΗΣΗ - προμηθευτής ηλεκτρονικών ειδών
    {
        "question": "Ποια εταιρεία ανέλαβε την προμήθεια ηλεκτρονικών ειδών για το έργο ΔΙΟΙΚΗΣΗ το 2021;",
        "expected_answer": "Creative Minds M. ΕΠΕ.",
        "expected_adas": ["Ψ57Χ469ΗΞΩ-Ν74"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },
     # 3. MORE - κόστος ηλεκτρονικού εξοπλισμού
    {
        "question": "Ποιο ποσό εγκρίθηκε για την προμήθεια ηλεκτρονικού εξοπλισμού στο έργο MORE το 2021;",
        "expected_answer": "5.592,93 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["ΩΞ2Ε469ΗΞΩ-ΜΤ5"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

   # 4. Ανάπτυξη και Λειτουργία - προμήθεια φαρμακευτικού υλικού
    {
        "question": "Τι προμηθεύτηκε το έργο Ανάπτυξη και Λειτουργία με δαπάνη 74,02 ευρώ;",
        "expected_answer": "Φαρμακευτικό υλικό.",
        "expected_adas": ["ΨΕΜΓ469ΗΞΩ-ΝΓΒ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

   # 5. Ανάπτυξη και Λειτουργία - προμηθευτής γραμματοσήμων
    {
        "question": "Από ποια εταιρεία έγινε η προμήθεια γραμματοσήμων για το έργο Ανάπτυξη και Λειτουργία το 2021;",
        "expected_answer": "ΕΛΛΗΝΙΚΑ ΤΑΧΥΔΡΟΜΕΙΑ Α.Ε.",
        "expected_adas": ["ΨΩΘ3469ΗΞΩ-9ΙΡ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 6. Visual Facts - κόστος δημοσίευσης επιστημονικού άρθρου
    {
        "question": "Ποιο ποσό εγκρίθηκε στο έργο Visual Facts για τη δημοσίευση επιστημονικού άρθρου;",
        "expected_answer": "2.178,00 ευρώ.",
        "expected_adas": ["ΨΒΙΓ469ΗΞΩ-Ζ3Ζ"],
        "expected_source_ids": [],
        "category": "publication",
        "answerable": True,
    },

    # 7. NEANIAS - εκτύπωση προωθητικού υλικού
    {
        "question": "Για ποιο σκοπό εγκρίθηκε δαπάνη 620,00 ευρώ στο έργο NEANIAS το 2022;",
        "expected_answer": "Για την εκτύπωση προωθητικού υλικού του έργου.",
        "expected_adas": ["93ΑΗ469ΗΞΩ-ΦΝΖ"],
        "expected_source_ids": [],
        "category": "promotion",
        "answerable": True,
    },

    # 8. STELAR - κόστος φιλοξενίας εναρκτήριας συνάντησης
    {
        "question": "Ποιο ποσό εγκρίθηκε για τη φιλοξενία στο πλαίσιο της εναρκτήριας συνάντησης του έργου STELAR;",
        "expected_answer": "1.008,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["9Β8Φ469ΗΞΩ-02Σ"],
        "expected_source_ids": [],
        "category": "event",
        "answerable": True,
    },

    # 9. ΑΡΧΙΜΗΔΗΣ - γραφική ύλη και είδη γραφείου
    {
        "question": "Ποιο ποσό εγκρίθηκε στο έργο ΑΡΧΙΜΗΔΗΣ για γραφική ύλη και είδη γραφείου;",
        "expected_answer": "1.160,50 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["Ω5ΑΡ469ΗΞΩ-4ΜΓ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    
# 10. ERA4TB - προμήθεια σκληρού δίσκου
    {
        "question": "Τι αγοράστηκε για το έργο ERA4TB με εγκεκριμένη δαπάνη 90,00 ευρώ;",
        "expected_answer": "Ένας σκληρός δίσκος.",
        "expected_adas": ["6Ρ4Γ469ΗΞΩ-ΑΕΗ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },
# 11. LAZARUS - εγγραφή σε συνέδριο
    {
        "question": "Σε ποιο συνέδριο αφορούσε η εγγραφή μέλους ΔΕΠ στο πλαίσιο του έργου LAZARUS το 2023;",
        "expected_answer": "Στο συνέδριο IEEE DAPPS 2023.",
        "expected_adas": ["6400469ΗΞΩ-9Δ0"],
        "expected_source_ids": [],
        "category": "conference",
        "answerable": True,
    },
# 12. EASIER - κόστος προωθητικού υλικού
    {
        "question": "Ποιο ποσό εγκρίθηκε για την προμήθεια προωθητικού υλικού στο έργο EASIER;",
        "expected_answer": "50,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["980Μ469ΗΞΩ-1ΑΔ"],
        "expected_source_ids": [],
        "category": "promotion",
        "answerable": True,
    },


# 13. SciLake - προμήθεια μνήμης RAM
    {
        "question": "Τι είδους εξοπλισμός αγοράστηκε για το έργο SciLake το 2023;",
        "expected_answer": "Μνήμη τυχαίας προσπέλασης (RAM).",
        "expected_adas": ["9Ν91469ΗΞΩ-ΠΚΨ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

# 14. EDITH - κόστος αναλώσιμων Η/Υ
    {
        "question": "Ποιο ποσό εγκρίθηκε για αναλώσιμα είδη Η/Υ στο έργο EDITH το 2024;",
        "expected_answer": "818,40 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["9ΒΟΩ469ΗΞΩ-ΑΤ5"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

# 15. GRAPES - συμμετοχή στο Summer School HYPATIA 2024
    {
        "question": "Σε ποια διοργάνωση αφορούσε η εγγραφή συνεργάτη του ΙΠΣΥ στο έργο GRAPES το 2024;",
        "expected_answer": "Στο Summer School HYPATIA 2024.",
        "expected_adas": ["96Β6469ΗΞΩ-97Λ"],
        "expected_source_ids": [],
        "category": "training",
        "answerable": True,
    },

  # 16. HDMS2024 - προωθητικό υλικό και ενοικίαση εξοπλισμού
    {
        "question": "Τι κάλυπτε η δαπάνη των 2.000,00 ευρώ στο έργο HDMS2024;",
        "expected_answer": "Την προμήθεια προωθητικού υλικού και την ενοικίαση εξοπλισμού για τη διοργάνωση του HDMS 2024.",
        "expected_adas": ["ΨΧΔ7469ΗΞΩ-122"],
        "expected_source_ids": [],
        "category": "event",
        "answerable": True,
    },

# 17. ALGEBRA - προμήθεια εργαστηριακού αναλώσιμου
    {
        "question": "Ποιο εργαστηριακό αναλώσιμο εγκρίθηκε για προμήθεια στο έργο ALGEBRA;",
        "expected_answer": "Κιτ για το NIR.",
        "expected_adas": ["9ΠΚΦ469ΗΞΩ-ΙΣ2"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },


# 18. EBRAINS 2.0 - ηλεκτρονικός εξοπλισμός και άδειες λογισμικού
    {
        "question": "Τι αφορούσε η προμήθεια ύψους 1.016,80 ευρώ στο έργο EBRAINS 2.0 το 2025;",
        "expected_answer": "Ηλεκτρονικό εξοπλισμό και άδειες χρήσης λογισμικού.",
        "expected_adas": ["9ΒΑΨ469ΗΞΩ-ΘΜ2"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },


# 19. EU BabyRobot+ - προμηθευτής αναλώσιμων Η/Υ

    {
        "question": "Ποιος ήταν ο προμηθευτής αναλώσιμων ειδών Η/Υ για το έργο EU BabyRobot+;",
        "expected_answer": "ΠΛΑΙΣΙΟ COMPUTERS AEBE.",
        "expected_adas": ["Ρ9Α3469ΗΞΩ-1ΥΕ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

# 20. ENABLE 6G - κόστος δημιουργίας ιστοσελίδας
    {
        "question": "Ποιο ποσό εγκρίθηκε για τη δημιουργία ιστοσελίδας του έργου ENABLE 6G το 2025;",
        "expected_answer": "6.200,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["Ψ1ΕΚ469ΗΞΩ-ΤΗΝ"],
        "expected_source_ids": [],
        "category": "web_services",
        "answerable": True,
    },
    
    # 21. SMS-CBA - κόστος
    {"question": "Πόσο κόστισε ο ηλεκτρονικός εξοπλισμός και το λογισμικό για το SMS-CBA;", 
     "expected_answer": "17.371,24 Ευρώ, πλέον ΦΠΑ.", 
     "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
     "expected_source_ids": [], 
     "category": "procurement", 
     "answerable": True},

    # 22. SMS-CBA - ανάδοχος
    {"question": "Ποια εταιρεία ανέλαβε την προμήθεια για το SMS-CBA;", 
     "expected_answer": "COSMOS BUSINESS SYSTEMS AEBE.", 
     "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
     "expected_source_ids": [], 
     "category": "procurement", 
     "answerable": True},

    # 23. SMS-CBA - αντικείμενο
    {"question": "Τι αγοράστηκε για το έργο SMS-CBA;", 
     "expected_answer": "Ηλεκτρονικός εξοπλισμός και ειδικό λογισμικό.", 
     "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
     "expected_source_ids": [], 
     "category": "procurement", 
     "answerable": True},

    # 24. SMS-CBA - λήξη σύμβασης
    {"question": "Μέχρι πότε διαρκούσε η σύμβαση προμήθειας για το SMS-CBA;", 
    "expected_answer": "Μέχρι 30/08/2021.", 
    "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
    "expected_source_ids": [], 
    "category": "procurement", 
    "answerable": True},

    # 25. SMS-CBA - έναρξη σύμβασης
    {"question": "Πότε ξεκινούσε η σύμβαση προμήθειας για το SMS-CBA;", 
     "expected_answer": "Στις 15/07/2021.", 
     "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
     "expected_source_ids": [], 
     "category": "procurement", 
     "answerable": True},

    # 26. BEHAVE - διάρκεια μετακίνησης
    {"question": "Πόσες μέρες θα διαρκούσε η μετακίνηση του συνεργάτη για το BEHAVE;", 
     "expected_answer": "36 ημέρες.", 
     "expected_adas": ["Ψ640469ΗΞΩ-30Ο"], 
     "expected_source_ids": [], 
     "category": "travel", 
     "answerable": True},

    # 27. BEHAVE - προορισμός
    {"question": "Πού θα ταξίδευε ο συνεργάτης για το έργο BEHAVE;", 
     "expected_answer": "Από την Αθήνα στο Λος Άντζελες.", 
     "expected_adas": ["Ψ640469ΗΞΩ-30Ο"], 
     "expected_source_ids": [], 
     "category": "travel", 
     "answerable": True},

    # 28. BEHAVE - αφετηρία
    {"question": "Από ποια πόλη θα ξεκινούσε η μετακίνηση για το BEHAVE;", 
     "expected_answer": "Από την Αθήνα.", 
     "expected_adas": ["Ψ640469ΗΞΩ-30Ο"], 
     "expected_source_ids": [], 
     "category": "travel", 
     "answerable": True},

    # 29. TRUSTEE - τροποποίηση σύμβασης
    {"question": "Τι άλλαξε στη σύμβαση του έργου TRUSTEE;", 
     "expected_answer": "Τροποποιήθηκε το οικονομικό αντικείμενο της σύμβασης.", 
     "expected_adas": ["67ΖΙ469ΗΞΩ-1ΧΧ"], 
     "expected_source_ids": [], 
     "category": "contract_modification", 
     "answerable": True},

    # 30. AutoFAIR - υποτροφία
    {"question": "Με τι αντικείμενο σχετιζόταν η υποτροφία AutoFAIR;", 
     "expected_answer": "Με έρευνα στον χώρο της δικαιοσύνης και της επεξηγησιμότητας αλγορίθμων μηχανικής μάθησης.", 
     "expected_adas": ["ΡΖΑΟ469ΗΞΩ-ΒΤΑ"], 
     "expected_source_ids": [], 
     "category": "scholarship", 
     "answerable": True},

    # 31. Οριζόντιο ΙΠΣΥ - Αικατερίνη
    {"question": "Πόσο ήταν το συνολικό κόστος της συνεργασίας της Αικατερίνης στο Οριζόντιο ΙΠΣΥ;", 
     "expected_answer": "7.350,00 Ευρώ.", 
     "expected_adas": ["6Θ5Β469ΗΞΩ-ΣΧΛ"], 
     "expected_source_ids": [], 
     "category": "contract", 
     "answerable": True},

    # 32. Οριζόντιο ΙΠΣΥ - Αντωνία
    {"question": "Πόσο ήταν το συνολικό κόστος της συνεργασίας της Αντωνίας στο Οριζόντιο ΙΠΣΥ;", 
     "expected_answer": "9.990,00 Ευρώ.", 
     "expected_adas": ["6Θ5Β469ΗΞΩ-ΣΧΛ"], 
     "expected_source_ids": [], 
     "category": "contract", 
     "answerable": True},

    # 33. ARIA - συνέντευξη
    {"question": "Πόσα μόρια μπορεί να δώσει η συνέντευξη στην πρόσκληση ARIA;", 
     "expected_answer": "Από 0 έως 10 μόρια.", 
     "expected_adas": ["9Ζ87469ΗΞΩ-ΕΩΟ"], 
     "expected_source_ids": [], 
     "category": "recruitment", 
     "answerable": True},

    # 34. ARIA - συνολική βαθμολογία
    {"question": "Ποια είναι η μέγιστη συνολική βαθμολογία στην αξιολόγηση ARIA;", 
     "expected_answer": "100 μόρια.", 
     "expected_adas": ["9Ζ87469ΗΞΩ-ΕΩΟ"], 
     "expected_source_ids": [], 
     "category": "recruitment", 
     "answerable": True},

    # 35. SMS-CBA - χρονικό διάστημα σύμβασης
    {"question": "Ποιο ήταν το χρονικό διάστημα της σύμβασης προμήθειας για το SMS-CBA;", 
     "expected_answer": "Από 15/07/2021 έως 30/08/2021.", 
     "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
     "expected_source_ids": [], 
     "category": "procurement", 
     "answerable": True},

    # 36. SMS-CBA - ανάδοχος και αντικείμενο
    {"question": "Τι προμήθευσε η COSMOS BUSINESS SYSTEMS AEBE για το SMS-CBA;", 
     "expected_answer": "Ηλεκτρονικό εξοπλισμό και ειδικό λογισμικό.", 
     "expected_adas": ["ΩΤΑΜ469ΗΞΩ-ΤΛ3"], 
     "expected_source_ids": [], 
     "category": "procurement", 
     "answerable": True},

    # 37. BEHAVE - διαδρομή και διάρκεια
    {"question": "Ποια ήταν η διαδρομή και η διάρκεια της μετακίνησης για το έργο BEHAVE;", 
     "expected_answer": "Από την Αθήνα στο Λος Άντζελες, για 36 ημέρες.", 
     "expected_adas": ["Ψ640469ΗΞΩ-30Ο"], 
     "expected_source_ids": [], 
     "category": "travel", 
     "answerable": True},

    # 38. ARIA - συνέντευξη σε σχέση με συνολική βαθμολογία
    {"question": "Πόσα μόρια μπορεί να δώσει η συνέντευξη στην ARIA και ποια είναι η μέγιστη συνολική βαθμολογία;", 
     "expected_answer": "Η συνέντευξη μπορεί να δώσει από 0 έως 10 μόρια και η μέγιστη συνολική βαθμολογία είναι 100 μόρια.", 
     "expected_adas": ["9Ζ87469ΗΞΩ-ΕΩΟ"], 
     "expected_source_ids": [], 
     "category": "recruitment", 
     "answerable": True},

# 39. ΕΚ Αθηνά - αναμόρφωση προϋπολογισμού
{
    "question": "Τι αφορούσε η δεύτερη αναμόρφωση του προϋπολογισμού του ΕΚ Αθηνά για το 2026;",
    "expected_answer": "Αφορούσε τη δεύτερη αναμόρφωση του προϋπολογισμού του ΕΚ Αθηνά για το οικονομικό έτος 2026.",
    "expected_adas": ["ΕΤ52469ΗΞΩ-ΕΤ0"],
    "expected_source_ids": [],
    "category": "budget",
    "answerable": True,
},

# UNANSWERABLE CASES
    

    # 40. Άδεια μητρότητας
    {
        "question": "Πόσες μέρες άδεια μητρότητας δικαιούμαι;",
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 41. Άδεια πατρότητας
    {
        "question":"Πόσες μέρες άδεια πατρότητας δικαιούμαι;",
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 42. Κανονική άδεια
    {
        "question":"Πόσες μέρες κανονική άδεια δικαιούμαι;",
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 43. Άδεια ασθενείας
    {
        "question":"Πόσες μέρες άδεια ασθενείας επί πληρωμή δικαιούμαι;",
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },
    # 44. Άδεια γάμου
    {
        "question": "Πόσες μέρες άδεια γάμου δικαιούμαι;",
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 45. Άδεια πένθους
    {"question": "Πόσες μέρες άδεια πένθους δικαιούμαι;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [], 
     "expected_source_ids": [], 
     "category": "unanswerable", 
     "answerable": False
     },

    # 46. Άδεια αιμοδοσίας
    {"question": "Πόσες μέρες άδεια αιμοδοσίας δικαιούμαι;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [],
    "expected_source_ids": [], 
    "category": "unanswerable", 
    "answerable": False
    },

    # 47. Κυριακή
    {
        "question":"Τι προσαύξηση παίρνω αν δουλέψω Κυριακή;",
        "expected_answer": UNANSWERABLE_RESPONSE,
        "expected_adas": [],
        "expected_source_ids": [],
        "category": "unanswerable",
        "answerable": False,
    },

    # 48. Διάλειμμα εργασίας
    {"question": "Πόσο διάλειμμα δικαιούμαι κατά τη διάρκεια της εργασίας μου;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [], 
     "expected_source_ids": [], 
     "category": "unanswerable", 
     "answerable": False},

    # 49. Εργατικό ατύχημα
    {"question": "Τι θεωρείται εργατικό ατύχημα;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [], 
     "expected_source_ids": [], 
     "category": "unanswerable", 
     "answerable": False},

    # 50. Διάλειμμα κατά την τηλεργασία
    {"question": "Πόσο διάλειμμα δικαιούμαι αν δουλεύω τηλεργασία;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [], 
     "expected_source_ids": [], 
     "category": "unanswerable", 
     "answerable": False},
]

# Βοηθητικά subsets


ANSWERABLE_CASES = [case for case in EVAL_DATASET if case["answerable"]]


UNANSWERABLE_CASES = [case for case in EVAL_DATASET if not case["answerable"]]


# Validation


def validate_dataset():
    """
    Βασικός έλεγχος της δομής του evaluation dataset.
    """

    assert len(EVAL_DATASET) == 50, (f"Αναμένονταν 50 cases, βρέθηκαν {len(EVAL_DATASET)}.")

    assert len(ANSWERABLE_CASES) == 39, (f"Αναμένονταν 39 answerable cases, βρέθηκαν {len(ANSWERABLE_CASES)}.")

    assert len(UNANSWERABLE_CASES) == 11, (f"Αναμένονταν 11 unanswerable cases, βρέθηκαν {len(UNANSWERABLE_CASES)}.")

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