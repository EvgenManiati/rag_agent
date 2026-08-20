from pathlib import Path
import hashlib
import json
import re

import pymupdf


# CONFIGURATION


DATASET_FILE = Path("data/diavgeia/final_dataset.jsonl")

EXTERNAL_PDF_DIR = Path("data/external_documents")


EXTERNAL_DOCUMENTS = [
    {
        "filename": "kanonismos_ehde_athina.pdf",
        "source_id": "athena_ehde_regulation",
        "title": (
            "Κανονισμός Αρχών και Λειτουργίας "
            "της Επιτροπής Ηθικής και "
            "Δεοντολογίας της Έρευνας"
        ),
        "organization": "Ερευνητικό Κέντρο Αθηνά",
    },

    {
        "filename": "odigos_xrimatodotisis_ekpa_2024.pdf",
        "source_id": "ekpa_funding_guide_2024",
        "title": (
            "Οδηγός Χρηματοδότησης και "
            "Διαχείρισης - Νοέμβριος 2024"
        ),
        "organization": (
            "Εθνικό και Καποδιστριακό "
            "Πανεπιστήμιο Αθηνών"
        ),
    },
]



# TEXT CLEANING

def clean_text(text: str) -> str:
    """
    Conservative PDF text cleaning.
    """

    if not text:
        return ""

    text = text.replace("\x00"," ",)

    text = (text.replace("\r\n", "\n").replace("\r", "\n"))

    # Join words broken by hyphen + newline.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+"," ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *","\n",text)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}","\n\n",text)

    return text.strip()



# PDF EXTRACTION


def extract_pdf(pdf_path: Path) -> tuple[list[str], str]:
    """
    Extract text page by page from a PDF.

    Returns:
        pages:
            Clean text of every usable page.

        full_text:
            All pages combined.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    try:

        for page_index in range(
            len(document)):

            page = document.load_page(page_index)

            raw_text = page.get_text("text", sort=True)

            cleaned_text = clean_text(raw_text)

            if cleaned_text:

                pages.append(cleaned_text)

    finally:

        document.close()


    full_text = "\n\n".join(
        pages
    ).strip()

    return pages, full_text



# CHECKSUM

def calculate_sha256(pdf_path: Path) -> str:

    return hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()



# LOAD EXISTING SOURCE IDs


def load_existing_source_ids() -> set[str]:
    """
    Prevent duplicate insertion if the script
    is accidentally executed more than once.
    """

    existing_ids = set()

    if not DATASET_FILE.exists():
        return existing_ids

    with DATASET_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                continue

            source_id = record.get("source_id")

            if source_id:

                existing_ids.add(str(source_id))

    return existing_ids


# APPEND JSONL


def append_record(
    record: dict,
) -> None:

    DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DATASET_FILE.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        file.write("\n")



# MAIN

def main():

    print(
        "EXTERNAL DOCUMENT IMPORT"
    )

    print(
        f"Dataset: "
        f"{DATASET_FILE.resolve()}"
    )

    print(
        f"PDF directory: "
        f"{EXTERNAL_PDF_DIR.resolve()}"
    )


    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Δεν βρέθηκε το dataset:\n"
            f"{DATASET_FILE.resolve()}"
        )


    existing_source_ids = (load_existing_source_ids())


    added = 0
    skipped = 0


    for document_info in (EXTERNAL_DOCUMENTS):

        source_id = (document_info["source_id"])

        pdf_path = (EXTERNAL_PDF_DIR / document_info["filename"])

        print(
            f"Έγγραφο: "
            f"{document_info['title']}"
        )


        
        # Already imported?
    

        if source_id in (
            existing_source_ids
        ):

            print(
                "Υπάρχει ήδη στο dataset. "
                "Παράλειψη."
            )

            skipped += 1

            continue


    
        # File exists?
    

        if not pdf_path.exists():

            print(
                f"ΔΕΝ ΒΡΕΘΗΚΕ:\n"
                f"{pdf_path.resolve()}"
            )

            skipped += 1

            continue


    
        # Extract
    

        pages, full_text = (extract_pdf(pdf_path))


        if not full_text:

            print(
                "Δεν εξήχθη κείμενο. "
                "Παράλειψη."
            )

            skipped += 1

            continue


        pdf_sha256 = calculate_sha256(pdf_path)


        # Build normalized record

        record = {

            # Δεν υπάρχει ΑΔΑ, γιατί δεν είναι
            # απόφαση Διαύγειας.
            "ada": None,

            "source_id": source_id,

            "subject": (
                document_info["title"]),

            "document_title": (document_info["title"]),

            "organization": (document_info["organization"]),

            "source": "external_pdf",

            "document_url": None,

            "issue_date": None,

            "organization_id": None,

            "decision_type_id": None,

            "total_page_count": len(pages),

            "text_page_count": len(pages),

            "pdf_size_bytes": (pdf_path.stat().st_size),

            "pdf_sha256": (pdf_sha256),

            "text_character_count": len(full_text),

            "pages": pages,

            "text": full_text,
        }


        append_record(record)


        existing_source_ids.add(source_id)

        added += 1


        print(f"Προστέθηκε επιτυχώς.")

        print(
            f"Σελίδες: "
            f"{len(pages)}"
        )

        print(
            f"Χαρακτήρες: "
            f"{len(full_text)}"
        )


    print("IMPORT COMPLETE")

    print(
        f"Νέα documents: "
        f"{added}"
    )

    print(
        f"Παραλείφθηκαν: "
        f"{skipped}"
    )


if __name__ == "__main__":
    main()