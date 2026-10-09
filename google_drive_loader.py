from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from langchain_core.documents import Document
from config import DIAVGEIA_DATASET_FILE


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = ("application/vnd.google-apps.folder")

DIAVGEIA_ROOT_FOLDER_NAME = "Diavgeia_5000"


def authenticate_google_drive(
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
):
    """
    Connect to Google Drive through OAuth.

    If the existing token is expired or invalid,
    a new login flow is started automatically.
    """

    credentials_path = Path(credentials_file)
    token_path = Path(token_file)

    if not credentials_path.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το credentials.json στο: "
            f"{credentials_path.resolve()}"
        )

    creds = None

    # Try to load an existing token.
    if token_path.exists():
        try:
            creds = (
                Credentials
                .from_authorized_user_file(
                    str(token_path),
                    SCOPES,
                )
            )

        except Exception:
            token_path.unlink(
                missing_ok=True
            )
            creds = None

    # Refresh expired token when possible.
    if creds and not creds.valid:

        if creds.expired and creds.refresh_token:

            try:
                creds.refresh(Request())

            except RefreshError:
                print(
                    "Το Google token έληξε "
                    "ή ανακλήθηκε. "
                    "Θα πραγματοποιηθεί νέα "
                    "σύνδεση."
                )

                token_path.unlink(missing_ok=True)

                creds = None

        else:
            token_path.unlink(missing_ok=True)

            creds = None

    # Start new login if necessary.
    if not creds or not creds.valid:

        flow = (
            InstalledAppFlow
            .from_client_secrets_file(
                str(credentials_path),
                SCOPES,
            )
        )

        creds = flow.run_local_server(
            port=0,
            open_browser=True,
            access_type="offline",
            prompt="select_account consent",
            success_message=(
                "Η σύνδεση ολοκληρώθηκε. "
                "Μπορείς να επιστρέψεις "
                "στην εφαρμογή."
            ),
        )

        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def load_diavgeia_metadata() -> dict[str, dict]:
    """
    Load Diavgeia metadata from final_dataset.jsonl.

    Returns:
        ADA -> original dataset record
    """

    if not DIAVGEIA_DATASET_FILE.exists():
        print(
            "WARNING: Δεν βρέθηκε "
            f"{DIAVGEIA_DATASET_FILE}. "
            "Τα Diavgeia documents θα φορτωθούν "
            "χωρίς issue_date / issue_year metadata."
        )

        return {}

    metadata_by_ada: dict[str, dict] = {}

    with DIAVGEIA_DATASET_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            if (record.get("source") != "diavgeia"):
                continue

            ada = str(record.get("ada", "")).strip()

            if not ada:
                continue

            metadata_by_ada[ada] = record

    print(f"Loaded Diavgeia metadata: {len(metadata_by_ada)} records.")

    return metadata_by_ada


def list_folder_items(service: Resource, folder_id: str) -> list[dict]:
    """
    Return all direct children of a Drive folder.
    """

    items: list[dict] = []

    page_token = None

    while True:

        response = (
            service.files()
            .list(
                q=(
                    f"'{folder_id}' in parents "
                    "and trashed = false"
                ),
                spaces="drive",
                fields=(
                    "nextPageToken, "
                    "files("
                    "id, "
                    "name, "
                    "mimeType, "
                    "modifiedTime, "
                    "parents"
                    ")"
                ),
                pageToken=page_token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        items.extend(response.get("files",[]))

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return items


def collect_pdf_files(
    service: Resource,
    folder_id: str,
    recursive: bool = True,
    parent_path: str = "",
) -> list[dict]:
    """
    Recursively collect PDF files.

    virtual_path stores the full logical path
    starting from the supplied root folder.
    """

    collected: list[dict] = []

    for item in list_folder_items(service, folder_id,):

        name = item.get("name", "")

        mime_type = item.get("mimeType", "")

        current_path = (f"{parent_path}/{name}" if parent_path else name)

        if mime_type == FOLDER_MIME_TYPE:

            if recursive:

                collected.extend(
                    collect_pdf_files(
                        service=service,
                        folder_id=item["id"],
                        recursive=True,
                        parent_path=current_path,
                    )
                )

            continue

        if (
            mime_type == PDF_MIME_TYPE
            or name.lower().endswith(".pdf")
        ):

            item["virtual_path"] = (current_path)

            collected.append(item)

    return collected

def parse_decision_type_folder(folder_name: str) -> tuple[str, str]:
    """
    Parse a folder name such as:

    Β.2.2 - Οριστικοποίηση Πληρωμής

    Returns:
        ("Β.2.2", "Οριστικοποίηση Πληρωμής")
    """

    if " - " not in folder_name:
        return "", folder_name.strip()

    decision_type_id, decision_type_name = (folder_name.split(" - ", 1))

    return (decision_type_id.strip(), decision_type_name.strip())

def drive_file_to_documents(file_info: dict, diavgeia_metadata: dict[str, dict]) -> list[Document]:
    """
    Convert one Google Drive PDF reference into page-level LangChain
    Documents using quality-controlled text from final_dataset.jsonl.

    Google Drive remains the authoritative source for file identity,
    folder structure and cloud path. The textual content used for
    retrieval is obtained from the corresponding dataset record,
    identified through the ADA contained in the PDF file name.

    Args:
        file_info:
            Google Drive metadata for one PDF file.

        diavgeia_metadata:
            Mapping from ADA identifiers to records loaded from
            final_dataset.jsonl.

    Returns:
        list[Document]:
            One LangChain Document for every original PDF page.

    Raises:
        RuntimeError:
            If the Drive PDF does not have a matching dataset record,
            if its page structure is invalid or if an expected page
            contains no text.
    """

    file_name = str(file_info.get("name", "")).strip()
    drive_path = str(file_info.get("virtual_path", file_name)).strip()
    path_parts = [part for part in drive_path.split("/") if part]

    root_folder_name = path_parts[0] if path_parts else ""
    folder_name = path_parts[-2] if len(path_parts) >= 2 else ""

    if root_folder_name.casefold() != DIAVGEIA_ROOT_FOLDER_NAME.casefold():
        return []

    ada = Path(file_name).stem.strip()

    if not ada:
        raise RuntimeError(f"Δεν ήταν δυνατό να εξαχθεί ADA από το Drive αρχείο: {file_name}")

    dataset_record = diavgeia_metadata.get(ada)

    if dataset_record is None:
        raise RuntimeError(f"Δεν βρέθηκε record στο final_dataset.jsonl για Drive PDF με ADA {ada}.")

    pages = dataset_record.get("pages", [])
    total_page_count = dataset_record.get("total_page_count")

    if not isinstance(pages, list):
        raise RuntimeError(f"Μη έγκυρο pages field για ADA {ada}.")

    if total_page_count is not None and len(pages) != total_page_count:
        raise RuntimeError(f"Page mapping mismatch για ADA {ada}: total_page_count={total_page_count}, pages={len(pages)}")

    folder_decision_type_id, folder_decision_type_name = parse_decision_type_folder(folder_name)
    dataset_decision_type_id = str(dataset_record.get("decision_type_id", "")).strip()

    decision_type_id = dataset_decision_type_id or folder_decision_type_id
    decision_type_name = folder_decision_type_name

    issue_date = str(dataset_record.get("issue_date", "") or dataset_record.get("publish_date", "")).strip()
    issue_year = issue_date[:4] if len(issue_date) >= 4 else ""

    base_metadata = {
        "source": "google_drive",
        "corpus": "diavgeia_5000",
        "file_name": file_name,
        "drive_file_id": file_info.get("id", ""),
        "root_folder": root_folder_name,
        "folder_name": folder_name,
        "drive_path": drive_path,
        "modified_time": file_info.get("modifiedTime", ""),
        "ada": ada,
        "decision_type_id": decision_type_id,
        "decision_type_name": decision_type_name,
        "decision_type_folder": folder_name,
        "issue_date": issue_date,
        "issue_year": issue_year,
        "document_url": dataset_record.get("document_url", ""),
        "extraction_method": dataset_record.get("extraction_method", ""),
        "repaired": dataset_record.get("repaired", False),
        "text_source": "final_dataset",
    }

    documents = []

    for page_number, page_text in enumerate(pages, start=1):
        page_text = str(page_text).strip()

        if not page_text:
            raise RuntimeError(f"Κενή σελίδα στο final_dataset για ADA {ada}, page {page_number}.")

        metadata = {**base_metadata, "page": page_number, "page_scope": "single_page"}

        documents.append(Document(page_content=page_text, metadata=metadata))

    return documents

def load_documents_from_drive_folders(
    folder_ids: Iterable[str],
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
    recursive: bool = True,
) -> list[Document]:
    """
    Load PDFs from multiple Google Drive folders.

    PDFs are downloaded only to memory.
    Nothing is written locally.
    """

    cleaned_folder_ids = [
        folder_id.strip()
        for folder_id in folder_ids
        if (folder_id and folder_id.strip())
    ]

    if not cleaned_folder_ids:

        raise ValueError("Δεν έχουν οριστεί Google Drive folder IDs.")

    service = authenticate_google_drive(
        credentials_file=credentials_file,
        token_file=token_file,
    )

    diavgeia_metadata = (load_diavgeia_metadata())

    all_documents: list[Document] = []

    seen_file_ids: set[str] = set()

    for folder_id in cleaned_folder_ids:

        folder_info = (
            service.files()
            .get(
                fileId=folder_id,
                fields="id, name",
                supportsAllDrives=True,
            )
            .execute()
        )

        root_folder_name = (folder_info.get("name", ""))

        if (root_folder_name.casefold()!= DIAVGEIA_ROOT_FOLDER_NAME.casefold()):
            raise ValueError(
                f"Μη επιτρεπόμενο Drive corpus: {root_folder_name}. "
                f"Ο loader επιτρέπεται να φορτώνει μόνο {DIAVGEIA_ROOT_FOLDER_NAME}."
            )

        print(f"\nLoading Drive folder: {root_folder_name}")

        pdf_files = collect_pdf_files(
            service=service,
            folder_id=folder_id,
            recursive=recursive,
            parent_path=root_folder_name,
        )

        print(f"PDF files found: {len(pdf_files)}")

        for index, file_info in enumerate(pdf_files, start=1):

            file_id = file_info["id"]

            # Avoid processing the same Drive file more than once.

            if file_id in seen_file_ids:
                continue

            seen_file_ids.add(file_id)

            print(f"[{index}/{len(pdf_files)}] {file_info.get('virtual_path', '')}")

            file_documents = drive_file_to_documents(file_info=file_info, diavgeia_metadata=diavgeia_metadata)

            all_documents.extend(file_documents)

    print(f"\nΣυνολικά δημιουργήθηκαν {len(all_documents)} Document objects.")

    print(f"Συνολικά μοναδικά PDF files: {len(seen_file_ids)}")

    return all_documents

if __name__ == "__main__":
    from config import DIAVGEIA_5000_FOLDER_ID

    documents = load_documents_from_drive_folders(
        folder_ids=[DIAVGEIA_5000_FOLDER_ID],
        recursive=True,
    )

    print("\nDRIVE LOADER TEST COMPLETE")
    print(f"Total Documents: {len(documents)}")

    if documents:
        print("\nFirst Document metadata:")
        print(documents[0].metadata)

        print("\nFirst Document text preview:")
        print(documents[0].page_content[:500])