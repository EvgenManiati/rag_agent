from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Iterable

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload
from langchain_core.documents import Document
from pypdf import PdfReader


SCOPES = [
    "https://www.googleapis.com/auth/drive"
]

PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = (
    "application/vnd.google-apps.folder"
)

DIAVGEIA_DATASET_FILE = Path(
    "data/diavgeia/final_dataset.jsonl"
)


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
                creds.refresh(
                    Request()
                )

            except RefreshError:
                print(
                    "Το Google token έληξε "
                    "ή ανακλήθηκε. "
                    "Θα πραγματοποιηθεί νέα "
                    "σύνδεση."
                )

                token_path.unlink(
                    missing_ok=True
                )

                creds = None

        else:
            token_path.unlink(
                missing_ok=True
            )

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

        token_path.write_text(
            creds.to_json(),
            encoding="utf-8",
        )

    return build(
        "drive",
        "v3",
        credentials=creds,
    )


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

    with DIAVGEIA_DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            if (
                record.get("source")
                != "diavgeia"
            ):
                continue

            ada = str(
                record.get(
                    "ada",
                    "",
                )
            ).strip()

            if not ada:
                continue

            metadata_by_ada[ada] = record

    print(
        "Loaded Diavgeia metadata: "
        f"{len(metadata_by_ada)} records."
    )

    return metadata_by_ada


def list_folder_items(
    service: Resource,
    folder_id: str,
) -> list[dict]:
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

        items.extend(
            response.get(
                "files",
                [],
            )
        )

        page_token = response.get(
            "nextPageToken"
        )

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

    for item in list_folder_items(
        service,
        folder_id,
    ):

        name = item.get(
            "name",
            "",
        )

        mime_type = item.get(
            "mimeType",
            "",
        )

        current_path = (
            f"{parent_path}/{name}"
            if parent_path
            else name
        )

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

            item["virtual_path"] = (
                current_path
            )

            collected.append(
                item
            )

    return collected


def download_file_to_memory(
    service: Resource,
    file_id: str,
    max_retries: int = 5,
) -> bytes:
    """
    Download a Drive PDF into memory.

    Retries are used for temporary
    network failures and timeouts.
    """

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            request = (
                service.files()
                .get_media(
                    fileId=file_id,
                    supportsAllDrives=True,
                )
            )

            buffer = io.BytesIO()

            downloader = (
                MediaIoBaseDownload(
                    buffer,
                    request,
                )
            )

            done = False

            while not done:

                _, done = (
                    downloader
                    .next_chunk(
                        num_retries=3
                    )
                )

            return buffer.getvalue()

        except (
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:

            print(
                f"DOWNLOAD ERROR "
                f"{attempt}/{max_retries} "
                f"| file_id={file_id} "
                f"| {exc}"
            )

            if attempt < max_retries:
                time.sleep(5)

    raise RuntimeError(
        "Αποτυχία λήψης Drive αρχείου "
        f"μετά από {max_retries} "
        f"προσπάθειες: {file_id}"
    )


def pdf_bytes_to_documents(
    pdf_bytes: bytes,
    file_info: dict,
    diavgeia_metadata: dict[str, dict],
) -> list[Document]:
    """
    Convert one PDF into LangChain Documents.

    One Document is created per page.
    """

    reader = PdfReader(
        io.BytesIO(
            pdf_bytes
        )
    )

    documents: list[Document] = []

    file_name = file_info.get(
        "name",
        "",
    )

    drive_path = file_info.get(
        "virtual_path",
        file_name,
    )

    path_parts = [
        part
        for part in drive_path.split("/")
        if part
    ]

    # Direct parent folder.
    folder_name = (
        path_parts[-2]
        if len(path_parts) >= 2
        else ""
    )

    # Root folder.
    root_folder_name = (
        path_parts[0]
        if path_parts
        else ""
    )

    is_diavgeia = (
        root_folder_name.lower()
        == "diavgeia"
    )

    ada = (
        Path(file_name).stem
        if is_diavgeia
        else ""
    )

    dataset_record = (
        diavgeia_metadata.get(
            ada,
            {},
        )
        if ada
        else {}
    )

    issue_date = str(
        dataset_record.get(
            "issue_date",
            "",
        )
    )

    issue_year = (
        issue_date[:4]
        if len(issue_date) >= 4
        else ""
    )

    decision_type_id = str(
        dataset_record.get(
            "decision_type_id",
            "",
        )
    )

    # For the current Drive structure,
    # the parent folder is the official
    # Diavgeia decision-type category.
    decision_type = (
        folder_name
        if is_diavgeia
        else ""
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = (
            page.extract_text()
            or ""
        )

        if not text.strip():
            continue

        metadata = {
            "source": "google_drive",

            "file_name": file_name,

            "drive_file_id": (
                file_info.get(
                    "id",
                    "",
                )
            ),

            "folder_name": folder_name,

            "drive_path": drive_path,

            "page": page_number,

            "modified_time": (
                file_info.get(
                    "modifiedTime",
                    "",
                )
            ),

            "ada": ada,

            "decision_type": (
                decision_type
            ),

            "decision_type_id": (
                decision_type_id
            ),

            "issue_date": issue_date,

            "issue_year": issue_year,
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

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
        if (
            folder_id
            and folder_id.strip()
        )
    ]

    if not cleaned_folder_ids:

        raise ValueError(
            "Δεν έχουν οριστεί "
            "Google Drive folder IDs."
        )

    service = authenticate_google_drive(
        credentials_file=credentials_file,
        token_file=token_file,
    )

    diavgeia_metadata = (
        load_diavgeia_metadata()
    )

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

        root_folder_name = (
            folder_info.get(
                "name",
                "",
            )
        )

        print(
            "\nLoading Drive folder: "
            f"{root_folder_name}"
        )

        pdf_files = collect_pdf_files(
            service=service,
            folder_id=folder_id,
            recursive=recursive,
            parent_path=root_folder_name,
        )

        print(
            f"PDF files found: "
            f"{len(pdf_files)}"
        )

        for index, file_info in enumerate(
            pdf_files,
            start=1,
        ):

            file_id = file_info["id"]

            # Avoid processing the same Drive
            # file more than once.
            if file_id in seen_file_ids:
                continue

            seen_file_ids.add(
                file_id
            )

            print(
                f"[{index}/{len(pdf_files)}] "
                f"{file_info.get('virtual_path', '')}"
            )

            pdf_bytes = (
                download_file_to_memory(
                    service=service,
                    file_id=file_id,
                )
            )

            file_documents = (
                pdf_bytes_to_documents(
                    pdf_bytes=pdf_bytes,
                    file_info=file_info,
                    diavgeia_metadata=(
                        diavgeia_metadata
                    ),
                )
            )

            all_documents.extend(
                file_documents
            )

    print(
        "f\nΣυνολικά δημιουργήθηκαν {len(all_documents)} Document objects.")

    print(
        "Συνολικά μοναδικά PDF files: "
        f"{len(seen_file_ids)}"
    )

    return all_documents