from datetime import date
from pathlib import Path


# Organization

#Research Center Athena
ORGANIZATION_UID = "99221083"


# Issue date range

DATE_FROM = date(2021, 1, 1)
DATE_TO = date(2026, 12, 31)


# API

DIAVGEIA_API_ROOT = "https://diavgeia.gov.gr/luminapi/opendata"

SEARCH_ENDPOINT = f"{DIAVGEIA_API_ROOT}/search"

DOCUMENT_URL_TEMPLATE = "https://diavgeia.gov.gr/doc/{ada}"

# Number of records per page for the final collection, set to 100.

PAGE_SIZE = 100

# None = retrieve all pages, otherwise set a number to limit the number of pages retrieved.
MAX_PAGES = None


#range of days to crawl for new documents. The crawler will look for documents issued in the last CRAWL_WINDOW_DAYS days.

CRAWL_WINDOW_DAYS = 180

REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 4

# Exponential backoff:
# 1s, 2s, 4s, 8s
RETRY_BASE_SECONDS = 2


# Filters

# We keep only published and active decisions.
DECISION_STATUS = "PUBLISHED"

# The sort order can be 'recent' or another supported value
SORT_ORDER = "recent"


# Debug / logging
DEBUG_MODE = False
LOG_LEVEL = "INFO"

# Output files

DATA_DIRECTORY = Path("data/diavgeia")
LOG_DIRECTORY = DATA_DIRECTORY / "logs"

METADATA_FILE = DATA_DIRECTORY / "metadata.jsonl"
SELECTED_METADATA_FILE = DATA_DIRECTORY / "selected_metadata.jsonl"
REJECTED_FILE = DATA_DIRECTORY / "rejected_metadata.jsonl"

LOG_FILE = LOG_DIRECTORY / "crawler.log"


# Dataset builder

PDF_DIRECTORY = DATA_DIRECTORY / "pdfs"

DATASET_FILE = DATA_DIRECTORY / "dataset.jsonl"
FAILED_FILE = DATA_DIRECTORY / "failed_documents.jsonl"

BUILDER_LOG_FILE = LOG_DIRECTORY / "dataset_builder.log"

# True:
#Stores the original PDFs in the pdfs folder.
# False:
#PDFs are temporarily loaded into RAM, text is extracted, and then discarded.
SAVE_PDFS = False

# None = process all documents.
# Number, e.g., 10 = only the first 10 for testing.
MAX_DOCUMENTS = None

# Minimum number of characters for a PDF to be considered as containing usable text.
MIN_DOCUMENT_CHARACTERS = 100

# Minimum number of characters per page.  Pages with fewer characters are considered empty.
MIN_PAGE_CHARACTERS = 10


# Suspicious document repair

NEW_SUSPICIOUS_FILE = DATA_DIRECTORY / "new_suspicious_documents.jsonl"
NEW_REPAIRED_FILE = DATA_DIRECTORY / "new_repaired_documents.jsonl"
NEW_REPAIR_FAILED_FILE = DATA_DIRECTORY / "new_repair_failed.jsonl"
NEW_REPAIR_LOG_FILE = LOG_DIRECTORY / "new_repair_suspicious.log"

# OCR settings

OCR_ENABLED = True
OCR_LANGUAGES = "ell+eng"
OCR_DPI = 250

# Quality threshold after which extracted text is accepted.

MIN_REPAIR_QUALITY_SCORE = 0.70

# New dataset pipeline

NEW_SELECTED_METADATA_FILE = DATA_DIRECTORY / "new_selected_metadata.jsonl"
NEW_DATASET_FILE = DATA_DIRECTORY / "dataset_new.jsonl"
NEW_FAILED_FILE = DATA_DIRECTORY / "new_failed_documents.jsonl"
NEW_QUALITY_REPORT_FILE = DATA_DIRECTORY / "new_quality_report.jsonl"
NEW_SUSPICIOUS_FILE = DATA_DIRECTORY / "new_suspicious_documents.jsonl"
NEW_REPAIRED_FILE = DATA_DIRECTORY / "new_repaired_documents.jsonl"
NEW_REPAIR_FAILED_FILE = DATA_DIRECTORY / "new_repair_failed.jsonl"
NEW_FINAL_DATASET_FILE = DATA_DIRECTORY / "final_dataset_new.jsonl"