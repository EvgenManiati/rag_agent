import json
from pathlib import Path

from googleapiclient.http import MediaFileUpload

from google_drive_loader import authenticate_google_drive


DATASET_FILE = Path("data/diavgeia/final_dataset.jsonl")
DIAVGEIA_DIR = Path("data/drive_upload/diavgeia")
EXTERNAL_DIR = Path("data/drive_upload/external")

ROOT_FOLDER_NAME = "RAG_KNOWLEDGE_BASE"


def find_or_create_folder(service, name, parent_id=None):
    """
    Βρίσκει έναν φάκελο στο Drive.
    Αν δεν υπάρχει, τον δημιουργεί.
    """

    escaped_name = name.replace("'", "\\'")

    query = (
        f"name = '{escaped_name}' "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )

    if parent_id:
        query += f" and '{parent_id}' in parents"

    result = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
    ).execute()

    folders = result.get("files", [])

    if folders:
        return folders[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }

    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(
        body=metadata,
        fields="id",
    ).execute()

    print(f"CREATED FOLDER: {name}")

    return folder["id"]


import time
from googleapiclient.errors import HttpError


def upload_pdf(service, file_path, parent_id):
    """
    Ανεβάζει PDF στο Drive.
    Αν υπάρχει ήδη στον ίδιο φάκελο, το προσπερνά.
    Κάνει retries σε προσωρινά network errors.
    """

    escaped_name = file_path.name.replace("'", "\\'")

    query = (
        f"name = '{escaped_name}' "
        f"and '{parent_id}' in parents "
        "and trashed = false"
    )

    existing = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
    ).execute()

    files = existing.get("files", [])

    if files:
        print(f"EXISTS IN DRIVE: {file_path.name}")
        return files[0]["id"]

    metadata = {
        "name": file_path.name,
        "parents": [parent_id],
    }

    for attempt in range(1, 4):

        try:
            media = MediaFileUpload(
                str(file_path),
                mimetype="application/pdf",
                resumable=True,
            )

            request = service.files().create(
                body=metadata,
                media_body=media,
                fields="id, name",
            )

            uploaded = request.execute(
                num_retries=3
            )

            print(f"UPLOADED: {uploaded['name']}")
            return uploaded["id"]

        except (
            HttpError,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as exc:

            print(
                f"UPLOAD ERROR {attempt}/3: "
                f"{file_path.name} -> {exc}"
            )

            if attempt < 3:
                time.sleep(5)

    print(f"FAILED: {file_path.name}")
    return None


def main():

    service = authenticate_google_drive()

    # Root
    root_id = find_or_create_folder(
        service,
        ROOT_FOLDER_NAME,
    )

    # Main folders
    diavgeia_root_id = find_or_create_folder(
        service,
        "Diavgeia",
        root_id,
    )

    external_root_id = find_or_create_folder(
        service,
        "External",
        root_id,
    )

    # -------------------------------
    # DIAVGEIA
    # -------------------------------

    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("source") != "diavgeia":
                continue

            ada = record.get("ada")
            issue_date = record.get("issue_date", "")

            if not ada:
                continue

            # π.χ. 2023-03-09 -> 2023
            year = (
                issue_date[:4]
                if len(issue_date) >= 4
                else "Unknown"
            )

            year_folder_id = find_or_create_folder(
                service,
                year,
                diavgeia_root_id,
            )

            pdf_path = DIAVGEIA_DIR / f"{ada}.pdf"

            if not pdf_path.exists():
                print(f"MISSING LOCAL PDF: {ada}")
                continue

            upload_pdf(
                service,
                pdf_path,
                year_folder_id,
            )



print("\nUPLOAD COMPLETE")

def test_upload():
    service = authenticate_google_drive()

    # Δημιουργία root folder
    root_id = find_or_create_folder(
        service,
        ROOT_FOLDER_NAME,
    )

    # Test folder
    test_folder_id = find_or_create_folder(
        service,
        "TEST_UPLOAD",
        root_id,
    )

    # Χρησιμοποιούμε το external PDF της ΕΗΔΕ
    test_pdf = (
        EXTERNAL_DIR
        / "kanonismos_ehde_athina.pdf"
    )

    if not test_pdf.exists():
        print(f"Δεν βρέθηκε: {test_pdf}")
        return

    upload_pdf(
        service,
        test_pdf,
        test_folder_id,
    )

    print("\nTEST UPLOAD COMPLETE")

if __name__ == "__main__":
    main()

    