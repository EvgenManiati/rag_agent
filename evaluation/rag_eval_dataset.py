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

    # 1. AdVENt: κόστος προμήθειας δίσκων

    {
        "question": "Πόσο κόστισε η προμήθεια 4 δίσκων για το έργο AdVENt;",
        "expected_answer": "967,60 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["69ΑΟ469ΗΞΩ-4ΔΥ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 2. ΔΙΟΙΚΗΣΗ: προμηθευτής ηλεκτρονικών ειδών
    {
        "question": "Ποια εταιρεία ανέλαβε την προμήθεια ηλεκτρονικών ειδών για το έργο ΔΙΟΙΚΗΣΗ το 2021;",
        "expected_answer": "Creative Minds M. ΕΠΕ.",
        "expected_adas": ["Ψ57Χ469ΗΞΩ-Ν74"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },
     # 3. MORE: κόστος ηλεκτρονικού εξοπλισμού
    {
        "question": "Ποιο ποσό εγκρίθηκε για την προμήθεια ηλεκτρονικού εξοπλισμού στο έργο MORE το 2021;",
        "expected_answer": "5.592,93 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["ΩΞ2Ε469ΗΞΩ-ΜΤ5"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

   # 4. Ανάπτυξη και Λειτουργία: προμήθεια φαρμακευτικού υλικού
    {
        "question": "Τι προμηθεύτηκε το έργο Ανάπτυξη και Λειτουργία με δαπάνη 74,02 ευρώ;",
        "expected_answer": "Φαρμακευτικό υλικό.",
        "expected_adas": ["ΨΕΜΓ469ΗΞΩ-ΝΓΒ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

   # 5. Ανάπτυξη και Λειτουργία: προμηθευτής γραμματοσήμων
    {
        "question": "Από ποια εταιρεία έγινε η προμήθεια γραμματοσήμων για το έργο Ανάπτυξη και Λειτουργία το 2021;",
        "expected_answer": "ΕΛΛΗΝΙΚΑ ΤΑΧΥΔΡΟΜΕΙΑ Α.Ε.",
        "expected_adas": ["ΨΩΘ3469ΗΞΩ-9ΙΡ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    # 6. Visual Facts: κόστος δημοσίευσης επιστημονικού άρθρου
    {
        "question": "Ποιο ποσό εγκρίθηκε στο έργο Visual Facts για τη δημοσίευση επιστημονικού άρθρου;",
        "expected_answer": "2.178,00 ευρώ.",
        "expected_adas": ["ΨΒΙΓ469ΗΞΩ-Ζ3Ζ"],
        "expected_source_ids": [],
        "category": "publication",
        "answerable": True,
    },

    # 7. NEANIAS: εκτύπωση προωθητικού υλικού
    {
        "question": "Για ποιο σκοπό εγκρίθηκε δαπάνη 620,00 ευρώ στο έργο NEANIAS το 2022;",
        "expected_answer": "Για την εκτύπωση προωθητικού υλικού του έργου.",
        "expected_adas": ["93ΑΗ469ΗΞΩ-ΦΝΖ"],
        "expected_source_ids": [],
        "category": "promotion",
        "answerable": True,
    },

    # 8. STELAR: κόστος φιλοξενίας εναρκτήριας συνάντησης
    {
        "question": "Ποιο ποσό εγκρίθηκε για τη φιλοξενία στο πλαίσιο της εναρκτήριας συνάντησης του έργου STELAR to 2022;",
        "expected_answer": "1.008,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["9Β8Φ469ΗΞΩ-02Σ"],
        "expected_source_ids": [],
        "category": "event",
        "answerable": True,
    },

    # 9. ΑΡΧΙΜΗΔΗΣ: γραφική ύλη και είδη γραφείου
    {
        "question": "Ποιο ποσό εγκρίθηκε στο έργο ΑΡΧΙΜΗΔΗΣ για γραφική ύλη και είδη γραφείου;",
        "expected_answer": "1.160,50 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["Ω5ΑΡ469ΗΞΩ-4ΜΓ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

    
# 10. ERA4TB: προμήθεια σκληρού δίσκου
    {
        "question": "Τι αγοράστηκε για το έργο ERA4TB με εγκεκριμένη δαπάνη 90,00 ευρώ;",
        "expected_answer": "Ένας σκληρός δίσκος.",
        "expected_adas": ["6Ρ4Γ469ΗΞΩ-ΑΕΗ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },
# 11. LAZARUS: εγγραφή σε συνέδριο
    {
        "question": "Σε ποιο συνέδριο αφορούσε η εγγραφή μέλους ΔΕΠ στο πλαίσιο του έργου «LAZARUS - pLatform for Analysis of Resilientand secUre Software» το 2023;",
        "expected_answer": "Στο συνέδριο IEEE DAPPS 2023.",
        "expected_adas": ["6400469ΗΞΩ-9Δ0"],
        "expected_source_ids": [],
        "category": "conference",
        "answerable": True,
    },
# 12. EASIER: κόστος προωθητικού υλικού
    {
        "question": "Ποιο ποσό εγκρίθηκε για την προμήθεια προωθητικού υλικού στο έργο EASIER;",
        "expected_answer": "50,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["980Μ469ΗΞΩ-1ΑΔ"],
        "expected_source_ids": [],
        "category": "promotion",
        "answerable": True,
    },


# 13. SciLake: προμήθεια μνήμης RAM
    {
        "question": "Τι είδους εξοπλισμός αγοράστηκε για το έργο SciLake το 2023;",
        "expected_answer": "Μνήμη τυχαίας προσπέλασης.",
        "expected_adas": ["9Ν91469ΗΞΩ-ΠΚΨ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

# 14. EDITH: κόστος αναλώσιμων Η/Υ
    {
        "question": "Ποιο ποσό εγκρίθηκε για αναλώσιμα είδη Η/Υ στο έργο EDITH το 2024;",
        "expected_answer": "818,40 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["9ΒΟΩ469ΗΞΩ-ΑΤ5"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

# 15. GRAPES: συμμετοχή στο Summer School HYPATIA 2024
    {
        "question": "Σε ποια διοργάνωση αφορούσε η εγγραφή συνεργάτη του ΙΠΣΥ στο έργο GRAPES το 2024;",
        "expected_answer": "Στο Summer School HYPATIA 2024.",
        "expected_adas": ["96Β6469ΗΞΩ-97Λ"],
        "expected_source_ids": [],
        "category": "training",
        "answerable": True,
    },

  # 16. HDMS2024: προωθητικό υλικό και ενοικίαση εξοπλισμού
    {
        "question": "Τι κάλυπτε η δαπάνη των 2.000,00 ευρώ στο έργο HDMS2024;",
        "expected_answer": "Την προμήθεια προωθητικού υλικού και την ενοικίαση εξοπλισμού για τη διοργάνωση του HDMS 2024.",
        "expected_adas": ["ΨΧΔ7469ΗΞΩ-122"],
        "expected_source_ids": [],
        "category": "event",
        "answerable": True,
    },

# 17. ALGEBRA: προμήθεια εργαστηριακού αναλώσιμου
    {
        "question": "Τι ύψους ήταν το ποσό της δαπάνης για το εργαστηριακό αναλώσιμο εγκρίθηκε για προμήθεια στο έργο ALGEBRA;",
        "expected_answer": "310,00 Ευρώ",
        "expected_adas": ["9ΠΚΦ469ΗΞΩ-ΙΣ2"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },


# 18. EBRAINS 2.0: ηλεκτρονικός εξοπλισμός και άδειες λογισμικού
    {
        "question": "Τι αφορούσε η προμήθεια ύψους 1.016,80 ευρώ στο έργο EBRAINS 2.0 το 2025;",
        "expected_answer": "Ηλεκτρονικό εξοπλισμό και άδειες χρήσης λογισμικού.",
        "expected_adas": ["9ΒΑΨ469ΗΞΩ-ΘΜ2"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },


# 19. EU BabyRobot+: προμηθευτής αναλώσιμων Η/Υ

    {
        "question": "Ποιος ήταν ο προμηθευτής αναλώσιμων ειδών Η/Υ για το έργο EU BabyRobot+;",
        "expected_answer": "ΠΛΑΙΣΙΟ COMPUTERS AEBE.",
        "expected_adas": ["Ρ9Α3469ΗΞΩ-1ΥΕ"],
        "expected_source_ids": [],
        "category": "procurement",
        "answerable": True,
    },

# 20. ENABLE 6G: κόστος δημιουργίας ιστοσελίδας
    {
        "question": "Ποιο ποσό εγκρίθηκε για τη δημιουργία ιστοσελίδας του έργου ENABLE 6G το 2025;",
        "expected_answer": "6.200,00 ευρώ, συμπεριλαμβανομένου ΦΠΑ και λοιπών νόμιμων κρατήσεων.",
        "expected_adas": ["Ψ1ΕΚ469ΗΞΩ-ΤΗΝ"],
        "expected_source_ids": [],
        "category": "web_services",
        "answerable": True,
    },
    
#21 AdVENt: Προμηθευτής
{
    "question": "Ποια εταιρεία ανέλαβε την προμήθεια αναλωσίμων και περιφερειακών ειδών για το έργο AdVENt;",
    "expected_answer": "Η easyPC – ΝΙΚΟΛΑΪΔΗΣ ΔΗΜΗΤΡΙΟΣ ΚΑΙ ΣΙΑ Ο.Ε.",
    "expected_adas": ["Ω8ΟΤ469ΗΞΩ-2ΣΙ"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},

#22 AdVENt: Ποσό προμήθειας
{
    "question": "Ποιο ήταν το ποσό της προμήθειας αναλώσιμων-περιφερειακών ειδών για το έργο AdVENt;",
    "expected_answer": "9.794,00 ευρώ, πλέον του αναλογούντος ΦΠΑ.",
    "expected_adas": ["Ω8ΟΤ469ΗΞΩ-2ΣΙ"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},

#23 AdVENt: Χρονικό διάστημα προμήθειας
{
    "question": "Ποιο ήταν το χρονικό διάστημα της προμήθειας αναλώσιμων- περιφεριακών ειδών για το έργο AdVENt;",
    "expected_answer": "Από 10/01/2022 έως 17/01/2022.",
    "expected_adas": ["Ω8ΟΤ469ΗΞΩ-2ΣΙ"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},

#24 IntelComp: Προμηθευτής ηλεκτρονικού εξοπλισμού
{
    "question": "Ποια εταιρεία ανέλαβε την προμήθεια ηλεκτρονικού εξοπλισμού για το IntelComp;",
    "expected_answer": "Η MSYSTEMS I.K.E. - ΕΜΠΟΡΙΑ ΣΥΣΤΗΜΑΤΩΝ ΠΛΗΡΟΟΦΟΡΙΚΗΣ ΔΙΚΤΥΑ‐SERVICE‐ΥΠΟΣΤΗΡΙΞΗΠΑΡΑΓΩΓΗ ΗΛΕΚΤΡΟΝΙΚΩΝ ΥΠΟΛΟΓΙΣΤΩΝ",
    "expected_adas": ["ΨΘ08469ΗΞΩ-Λ28"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},

#25 IntelComp: Ποσό προμήθειας
{
    "question": "Πόσο κόστισε η προμήθεια ηλεκτρονικού εξοπλισμού για το IntelComp;",
    "expected_answer": "4.397,1 ευρώ, πλέον του αναλογούντος ΦΠΑ.",
    "expected_adas": ["ΨΘ08469ΗΞΩ-Λ28"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},


#26 PREMIERE: Διαδρομή μετακίνησης
{
    "question": "Από πού προς πού θα μετακινούνταν ο συνεργάτης του έργου PREMIERE;",
    "expected_answer": "Από την Αθήνα στο Πόρτο.",
    "expected_adas": ["ΨΝΤΥ469ΗΞΩ-ΒΣ4"],
    "expected_source_ids": [],
    "category": "travel",
    "answerable": True,
},


#27 PREMIERE: Χρονικό διάστημα μετακίνησης
{
    "question": "Πόσο ήταν το χρονικό διάστημα της μετακίνησης του συνεργάτη ID-8816 για το έργο PREMIERE;",
    "expected_answer": "6 ημέρες",
    "expected_adas": ["ΨΝΤΥ469ΗΞΩ-ΒΣ4"],
    "expected_source_ids": [],
    "category": "travel",
    "answerable": True,
},


#28 IEEE Smart Cities Conference: Προορισμός μετακίνησης
{
    "question": "Πού θα ταξίδευε ο Διευθυντής για το συνέδριο 10th IEEE International Smart Cities Conference;",
    "expected_answer": "Από την Πάτρα στην Pataya City της Ταϊλάνδης.",
    "expected_adas": ["9Υ7Γ469ΗΞΩ-9ΞΞ"],
    "expected_source_ids": [],
    "category": "travel",
    "answerable": True,
},

#29 TRUSTEE: Τροποποίηση σύμβασης
{
    "question": "Τι άλλαξε στη σύμβαση του έργου TRUSTEE;",
    "expected_answer": "Τροποποιήθηκε το οικονομικό αντικείμενο της σύμβασης.",
    "expected_adas": ["67ΖΙ469ΗΞΩ-1ΧΧ"],
    "expected_source_ids": [],
    "category": "contract_modification",
    "answerable": True,
},

#30 Ανάπτυξη Αυτόνομων Ρομποτικών Συστημάτων: Υποτροφία
{
    "question": "Ποιο ήταν το μηνιαίο ποσό της υποτροφίας στο έργο Ανάπτυξη Αυτόνομων Ρομποτικών Συστημάτων;",
    "expected_answer": "1.300,00 ευρώ τον μήνα.",
    "expected_adas": ["9Ε7Β469ΗΞΩ-4ΟΞ"],
    "expected_source_ids": [],
    "category": "scholarship",
    "answerable": True,
},
# 31 - ΟΡΙΖΟΝΤΙΟ ΙΠΣΥ - Αικατερίνη
{
    "question": "Πόσο ήταν το συνολικό κόστος της συνεργασίας της Αικατερίνης Ταχριλτζίδου στο Οριζόντιο ΙΠΣΥ;",
    "expected_answer": "7.350,00 Ευρώ.",
    "expected_adas": ["6Θ5Β469ΗΞΩ-ΣΧΛ"],
    "expected_source_ids": [],
    "category": "contract",
    "answerable": True,
},

# 32 - ΟΡΙΖΟΝΤΙΟ ΙΠΣΥ - Αντωνία
{
    "question": "Πόσο ήταν το συνολικό κόστος της συνεργασίας της Αντωνίας Ηλιοπούλου στο Οριζόντιο ΙΠΣΥ;",
    "expected_answer": "9.990,00 Ευρώ.",
    "expected_adas": ["6Θ5Β469ΗΞΩ-ΣΧΛ"],
    "expected_source_ids": [],
    "category": "contract",
    "answerable": True,
},

#33 MINOR-MED: Μόρια συνέντευξης
{
    "question": "Πόσα μόρια μπορεί να δώσει η συνέντευξη στην πρόσκληση MINOR-MED;",
    "expected_answer": "Έως 10 μόρια.",
    "expected_adas": ["921Ε469ΗΞΩ-Λ7Δ"],
    "expected_source_ids": [],
    "category": "recruitment",
    "answerable": True,
},

#34 MINOR-MED: Ανώτατη συνολική βαθμολογία
{
    "question": "Ποια είναι η ανώτατη συνολική βαθμολογία στην αξιολόγηση της πρόσκλησης MINOR-MED;",
    "expected_answer": "100 μόρια.",
    "expected_adas": ["921Ε469ΗΞΩ-Λ7Δ"],
    "expected_source_ids": [],
    "category": "recruitment",
    "answerable": True,
},


#35 ΤΜΗΜΑ 3: Προμηθευτής φορητού συστήματος φασματοσκoπίας
{
    "question": "Ποια εταιρεία ανέλαβε την προμήθεια ηλεκτρονικού και εργαστηριακού εξοπλισμού για ΤΜΗΜΑ 3 –Φορητό σύστημα φασματοσκοπίας φθορισμού ακτίνων-Χ;",
    "expected_answer": "Η ΑΝΑΛΥΤΙΚΕΣ ΣΥΣΚΕΥΕΣ ΑΕ.",
    "expected_adas": ["92ΧΔ469ΗΞΩ-ΦΒ8"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},


#36 ΑΝΑΛΥΤΙΕΣ ΣΥΣΚΕΥΕΣ ΑΕ.: Ποσό σύμβασης
{
    "question": "Ποιο ήταν το ποσό της σύμβασης προμήθειας ηλεκτρονικού και εργαστηριακού εξοπλισμού με την ΑΝΑΛΥΤΙΚΕΣ ΣΥΣΚΕΥΕΣ ΑΕ;",
    "expected_answer": "94.866,00 ευρώ, πλέον του αναλογούντος ΦΠΑ.",
    "expected_adas": ["92ΧΔ469ΗΞΩ-ΦΒ8"],
    "expected_source_ids": [],
    "category": "procurement",
    "answerable": True,
},


#37 IEEE Smart Cities Conference: Διάρκεια μετακίνησης
{
    "question": "Πόσες ημέρες θα διαρκούσε η μετακίνηση στην Ταϊλάνδη για το 10th IEEE International Smart Cities Conference;",
    "expected_answer": "8 ημέρες.",
    "expected_adas": ["9Υ7Γ469ΗΞΩ-9ΞΞ"],
    "expected_source_ids": [],
    "category": "travel",
    "answerable": True,
},

#38 RegulAIze: Ελάχιστη βαθμολογία επιτυχίας
{
    "question": "Πόσα μόρια πρέπει να συγκεντρώσει τουλάχιστον ένας υποψήφιος για να θεωρηθεί επιτυχών στην πρόσκληση RegulAIze;",
    "expected_answer": "Τουλάχιστον 40 μόρια.",
    "expected_adas": ["63Η3469ΗΞΩ-1Ρ6"],
    "expected_source_ids": [],
    "category": "recruitment",
    "answerable": True,
},

# 39. ΑΝΑΛΥΤΙΚΕΣ ΣΥΣΚΕΥΕΣ : Τμήματα σύμβασης
{
    "question": "Ποια ήταν τα τμήματα που ανατέθηκαν στην εταιρεία ΑΝΑΛΥΤΙΚΕΣ ΣΥΣΚΕΥΕΣ ΑΕ;",
    "expected_answer": "ΤΜΗΜΑ 3 – Φορητό σύστημα φασματοσκοπίας φθορισμού ακτίνων-Χ,"
     "ΤΜΗΜΑ 7 – Συστήματα μέτρησης, ΤΜΗΜΑ 8 –Εργαστηριακή φυγόκεντρος και ΤΜΗΜΑ 12 Στερεοσκόπιο",
    "expected_adas": ["92ΧΔ469ΗΞΩ-ΦΒ8"],
    "expected_source_ids": [],
    "category": "procurement",
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
     "answerable": False
},

    # 49. Εργατικό ατύχημα
{"question": "Τι θεωρείται εργατικό ατύχημα;", 
     "expected_answer": UNANSWERABLE_RESPONSE, 
     "expected_adas": [], 
     "expected_source_ids": [], 
     "category": "unanswerable", 
     "answerable": False
},

# 50. Επίδομα τηλεργασίας
{
    "question": "Πόσο επίδομα τηλεργασίας δικαιούμαι κάθε μήνα;",
    "expected_answer": UNANSWERABLE_RESPONSE,
    "expected_adas": [],
    "expected_source_ids": [],
    "category": "unanswerable",
    "answerable": False,
},

]

# Βοηθητικά subsets


ANSWERABLE_CASES = [case for case in EVAL_DATASET if case["answerable"]]


UNANSWERABLE_CASES = [case for case in EVAL_DATASET if not case["answerable"]]


# Validation


def validate_dataset():
    """
    Main check of the evaluation dataset.
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

        assert not missing_fields, (f"Case {index}: λείπουν fields {sorted(missing_fields)}")

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