import json
import re
from functools import lru_cache

from langchain_core.documents import Document

from config import DIAVGEIA_DATASET_FILE


DECISION_TYPE_NAMES = {
    "Β.2.2": "Οριστικοποίηση Πληρωμής",
    "Β.2.1": "Έγκριση Δαπάνης",
    "Β.1.3": "Ανάληψη Υποχρέωσης",
    "Γ.3.4": "Σύμβαση",
    "2.4.7.1": "Λοιπές Ατομικές Διοικητικές Πράξεις",
    "Δ.1": "Ανάθεση Έργων Προμηθειών Υπηρεσιών Μελετών",
    "Γ.3.1": "Προκήρυξη Πλήρωσης Θέσεων",
    "Α.2": "Κανονιστική Πράξη",
    "Γ.3.2": "Πίνακες Επιτυχόντων Διοριστέων Επιλαχόντων",
    "Δ.2.1": "Περίληψη Διακήρυξης",
    "Γ.2": "Συλλογικό Όργανο Επιτροπή Ομάδα Εργασίας",
    "Β.3": "Ισολογισμός Απολογισμός",
    "Β.1.1": "Έγκριση Προϋπολογισμού",
    "Β.4": "Δωρεά Επιχορήγηση",
    "Γ.3.3": "Διορισμός",
    "Γ.3.5": "Υπηρεσιακή Μεταβολή",
}


def safe_folder_name(value: str) -> str:
    """
    Convert a string into a filesystem-safe folder name.

    Characters that are not allowed in Windows file and folder names
    are removed, while leading and trailing whitespace is stripped.

    Args:
        value:
            Original folder name.

    Returns:
        str:
            Sanitized folder name that can safely be used in a path.
    """
    return re.sub(r'[<>:"/\\|?*]', "", value).strip()



def get_folder_name(record: dict) -> str:

    """
    Build the Google Drive folder name for a Diavgeia decision.

    The folder name is derived from the decision type identifier stored
    in the dataset record and the corresponding human-readable decision
    type name.

    Args:
        record:
            One record loaded from final_dataset.jsonl.

    Returns:
        str:
            Folder name in the form
            "<decision_type_id> - <decision_type_name>".
    """
    
    decision_type_id = str(record.get("decision_type_id", "UNKNOWN")).strip()
    decision_type_name = DECISION_TYPE_NAMES.get(decision_type_id, "Άγνωστος Τύπος")
    return safe_folder_name(f"{decision_type_id} - {decision_type_name}")


@lru_cache(maxsize=1)
def load_documents_from_final_dataset() -> tuple[Document, ...]:
   
    """
    Load the quality-controlled Diavgeia corpus from final_dataset.jsonl.

    Each JSONL record represents one Diavgeia administrative decision.
    The record contains the text of the document separated into pages.

    Every page is converted into an independent LangChain Document so
    that page-level source information is preserved throughout the RAG
    pipeline.

    The generated metadata includes:
        - ADA identifier
        - PDF file name
        - decision type
        - folder name
        - page number
        - issue date and year
        - original Diavgeia document URL
        - logical Google Drive path
        - text extraction method
        - repair status
        - corpus and text source identifiers

    The loader also validates the page structure of every record.
    If the number of entries in the "pages" field differs from
    "total_page_count", execution stops to prevent incorrect PDF page
    mappings from propagating into the retrieval index.

    The function is cached because the dataset is static during a
    single application execution and loading thousands of page-level
    Documents repeatedly would be unnecessary.

    Returns:
        tuple[Document, ...]:
            Immutable collection of page-level LangChain Documents.

    Raises:
        FileNotFoundError:
            If final_dataset.jsonl cannot be found.

        RuntimeError:
            If a JSON record is malformed, an ADA is missing, the page
            structure is invalid, a page is empty, or no usable
            Documents are produced.
    """
    if not DIAVGEIA_DATASET_FILE.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε το dataset: {DIAVGEIA_DATASET_FILE.resolve()}")

    documents = []
    record_count = 0

    with DIAVGEIA_DATASET_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Μη έγκυρο JSON στη γραμμή {line_number}: {error}") from error

            record_count += 1

            ada = str(record.get("ada", "")).strip()
            pages = record.get("pages", [])
            total_page_count = record.get("total_page_count")

            if not ada:
                raise RuntimeError(f"Λείπει ADA στη γραμμή {line_number}.")

            if not isinstance(pages, list):
                raise RuntimeError(f"Μη έγκυρο pages για ADA {ada}.")

            if total_page_count is not None and len(pages) != total_page_count:
                raise RuntimeError(f"Page mapping mismatch για ADA {ada}: total_page_count={total_page_count}, pages={len(pages)}")

            decision_type_id = str(record.get("decision_type_id", "")).strip()
            decision_type_name = DECISION_TYPE_NAMES.get(decision_type_id, "Άγνωστος Τύπος")
            folder_name = get_folder_name(record)

            issue_date = str(record.get("issue_date", "") or record.get("publish_date", "")).strip()
            issue_year = issue_date[:4] if len(issue_date) >= 4 else ""

            for page_number, page_text in enumerate(pages, start=1):
                page_text = str(page_text).strip()

                if not page_text:
                    raise RuntimeError(f"Κενή σελίδα στο final_dataset για ADA {ada}, page {page_number}.")

                metadata = {
                    "source": "diavgeia",
                    "corpus": "diavgeia_5000",
                    "ada": ada,
                    "file_name": f"{ada}.pdf",
                    "folder_name": folder_name,
                    "decision_type_id": decision_type_id,
                    "decision_type_name": decision_type_name,
                    "decision_type_folder": folder_name,
                    "page": page_number,
                    "page_scope": "single_page",
                    "issue_date": issue_date,
                    "issue_year": issue_year,
                    "document_url": record.get("document_url", ""),
                    "drive_path": f"Diavgeia_5000/{folder_name}/{ada}.pdf",
                    "extraction_method": record.get("extraction_method", ""),
                    "repaired": record.get("repaired", False),
                    "text_source": "final_dataset",
                }

                documents.append(Document(page_content=page_text, metadata=metadata))

    if not documents:
        raise RuntimeError("Το final_dataset.jsonl δεν περιείχε αξιοποιήσιμα documents.")

    print(f"Dataset records: {record_count}")
    print(f"Page-level Documents: {len(documents)}")

    return tuple(documents)

if __name__ == "__main__":
    documents = load_documents_from_final_dataset()

    print("\nTEST COMPLETE")
    print(f"Total Documents: {len(documents)}")

    if documents:
        print("\nFirst Document metadata:")
        print(documents[0].metadata)

        print("\nFirst Document text preview:")
        print(documents[0].page_content[:500])