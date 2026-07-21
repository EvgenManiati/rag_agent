import io
from pathlib import Path
from typing import Iterable
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload
from langchain_core.documents import Document
from pypdf import PdfReader


# Read-only δικαίωμα πρόσβασης στο Google Drive.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def authenticate_google_drive(
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
) -> Resource:
    """
    Συνδέεται στο Google Drive μέσω OAuth.

    Το credentials.json περιγράφει την εφαρμογή.
    Το token.json αποθηκεύει την έγκριση του χρήστη και
    δημιουργείται αυτόματα μετά το πρώτο επιτυχημένο login.
    """

    credentials_path = Path(credentials_file)
    token_path = Path(token_file)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το credentials.json στο: "
            f"{credentials_path.resolve()}"
        )

    creds = None

    # Χρησιμοποιούμε υπάρχον token, αν υπάρχει.
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(
            str(token_path),
            SCOPES,
        )

    # Αν το token λείπει ή δεν είναι έγκυρο,
    # γίνεται refresh ή ανοίγει νέο OAuth login.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                SCOPES,
            )

            creds = flow.run_local_server(port=0)

        token_path.write_text(
            creds.to_json(),
            encoding="utf-8",
        )

    return build(
        "drive",
        "v3",
        credentials=creds,
    )


def list_folder_items(
    service: Resource,
    folder_id: str,
) -> list[dict]:
    """
    Επιστρέφει όλα τα άμεσα παιδιά ενός Google Drive φακέλου.
    """

    items: list[dict] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields=(
                    "nextPageToken, "
                    "files(id, name, mimeType, modifiedTime, parents)"
                ),
                pageToken=page_token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        items.extend(response.get("files", []))
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
    Βρίσκει όλα τα PDF ενός φακέλου.

    Αν recursive=True, διαβάζει και τους υποφακέλους.
    Το virtual_path κρατά τη διαδρομή του αρχείου μέσα στο Drive.
    """

    collected: list[dict] = []

    for item in list_folder_items(service, folder_id):
        name = item.get("name", "")
        mime_type = item.get("mimeType", "")

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

        if mime_type == PDF_MIME_TYPE or name.lower().endswith(".pdf"):
            item["virtual_path"] = current_path
            collected.append(item)

    return collected


def download_file_to_memory(
    service: Resource,
    file_id: str,
) -> bytes:
    """
    Κατεβάζει ένα PDF σε προσωρινό buffer στη RAM.

    Δεν δημιουργείται κανένα αρχείο στον δίσκο.
    """

    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request,
    )

    done = False

    while not done:
        _, done = downloader.next_chunk()

    # Επιστρέφουμε το περιεχόμενο του buffer ως bytes.
    return buffer.getvalue()


def pdf_bytes_to_documents(
    pdf_bytes: bytes,
    file_info: dict,
) -> list[Document]:
    """
    Μετατρέπει PDF bytes σε LangChain Document objects.

    Δημιουργείται ένα Document ανά σελίδα.
    """

    reader = PdfReader(io.BytesIO(pdf_bytes))

    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Αγνοούμε εντελώς κενές σελίδες.
        if not text.strip():
            continue

        metadata = {
            "source": file_info.get("name", ""),
            "drive_file_id": file_info.get("id", ""),
            "drive_path": file_info.get(
                "virtual_path",
                file_info.get("name", ""),
            ),
            "page": page_number,
            "modified_time": file_info.get("modifiedTime", ""),
            "document_source": "google_drive",
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
    Διαβάζει PDF από πολλούς Google Drive φακέλους
    και τα επιστρέφει ως LangChain Documents.

    Δεν αποθηκεύει τίποτα τοπικά.
    """

    cleaned_folder_ids = [
        folder_id.strip()
        for folder_id in folder_ids
        if folder_id and folder_id.strip()
    ]

    if not cleaned_folder_ids:
        raise ValueError(
            "Δεν έχουν οριστεί Google Drive folder IDs."
        )

    service = authenticate_google_drive(
        credentials_file=credentials_file,
        token_file=token_file,
    )

    all_documents: list[Document] = []
    seen_file_ids: set[str] = set()

    for folder_id in cleaned_folder_ids:

        pdf_files = collect_pdf_files(
            service=service,
            folder_id=folder_id,
            recursive=recursive,
        )


        for file_info in pdf_files:
            file_id = file_info["id"]
            file_name = file_info["name"]

            # Αποφεύγουμε διπλή επεξεργασία του ίδιου Drive αρχείου.
            if file_id in seen_file_ids:
                continue

            seen_file_ids.add(file_id)

            pdf_bytes = download_file_to_memory(
                service=service,
                file_id=file_id,
            )

            file_documents = pdf_bytes_to_documents(
                pdf_bytes=pdf_bytes,
                file_info=file_info,
            )


            all_documents.extend(file_documents)

    print(
        f"Συνολικά δημιουργήθηκαν "
        f"{len(all_documents)} Document objects."
    )

    return all_documents