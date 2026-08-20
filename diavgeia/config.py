from datetime import date
from pathlib import Path


# Φορέας

# Ερευνητικό Κέντρο «Αθηνά»
ORGANIZATION_UID = "99221083"


# Χρονικό διάστημα

DATE_FROM = date(2021, 1, 1)
DATE_TO = date(2026, 12, 31)


# API


DIAVGEIA_API_ROOT = "https://diavgeia.gov.gr/luminapi/opendata"

SEARCH_ENDPOINT = f"{DIAVGEIA_API_ROOT}/search"

DOCUMENT_URL_TEMPLATE = "https://diavgeia.gov.gr/doc/{ada}"

# Πλήθος εγγραφών ανά σελίδα.
# Για την τελική συλλογή βάλε 100.
PAGE_SIZE = 20

# None = ανάκτηση όλων των σελίδων.
# Για δοκιμή μπορείς να βάλεις 1, 3, 5 κ.λπ.
MAX_PAGES = 3

REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 4

# Exponential backoff:
# 1s, 2s, 4s, 8s
RETRY_BASE_SECONDS = 2


# Φίλτρα

# Κρατάμε μόνο αναρτημένες και ενεργές πράξεις.
DECISION_STATUS = "PUBLISHED"

# Η σειρά μπορεί να είναι recent ή άλλο υποστηριζόμενο
# sort value της υπηρεσίας.
SORT_ORDER = "recent"


# Debug / logging


DEBUG_MODE = False
LOG_LEVEL = "INFO"

# Αρχεία εξόδου

DATA_DIRECTORY = Path("data/diavgeia")
LOG_DIRECTORY = DATA_DIRECTORY / "logs"

METADATA_FILE = DATA_DIRECTORY / "metadata.jsonl"
REJECTED_FILE = DATA_DIRECTORY / "rejected_metadata.jsonl"

LOG_FILE = LOG_DIRECTORY / "crawler.log"


# Dataset builder

PDF_DIRECTORY = DATA_DIRECTORY / "pdfs"

DATASET_FILE = DATA_DIRECTORY / "dataset.jsonl"
FAILED_FILE = DATA_DIRECTORY / "failed_documents.jsonl"

BUILDER_LOG_FILE = LOG_DIRECTORY / "dataset_builder.log"

# True:
# Αποθηκεύει και τα πρωτότυπα PDF στον φάκελο pdfs.
#
# False:
# Τα PDF μεταφέρονται προσωρινά στη RAM, εξάγεται το κείμενο
# και μετά απορρίπτονται.
SAVE_PDFS = False

# None = επεξεργασία όλων των metadata records.
# Αριθμός, π.χ. 10 = μόνο τα πρώτα 10 για δοκιμή.
MAX_DOCUMENTS = None

# Ελάχιστος αριθμός χαρακτήρων ώστε ένα PDF να θεωρείται
# ότι περιέχει αξιοποιήσιμο κείμενο.
MIN_DOCUMENT_CHARACTERS = 100

# Ελάχιστος αριθμός χαρακτήρων ανά σελίδα.
# Σελίδες με λιγότερους χαρακτήρες θεωρούνται κενές.
MIN_PAGE_CHARACTERS = 10


# Suspicious document repair

SUSPICIOUS_FILE = (
    DATA_DIRECTORY / "suspicious_documents.jsonl"
)

REPAIRED_FILE = (
    DATA_DIRECTORY / "repaired_documents.jsonl"
)

REPAIR_FAILED_FILE = (
    DATA_DIRECTORY / "repair_failed.jsonl"
)

REPAIR_LOG_FILE = (
    LOG_DIRECTORY / "repair_suspicious.log"
)

# OCR settings
OCR_ENABLED = True
OCR_LANGUAGES = "ell+eng"
OCR_DPI = 250

# Quality threshold after which extracted text is accepted.
MIN_REPAIR_QUALITY_SCORE = 0.70