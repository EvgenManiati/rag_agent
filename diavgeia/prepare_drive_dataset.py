from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import DIAVGEIA_DATASET_FILE


OUTPUT_ROOT = Path("data/drive_upload/Diavgeia_5000")

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
REQUEST_DELAY = 0.10


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


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def safe_folder_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", value).strip()


def get_folder(record: dict) -> Path:
    decision_type = str(record.get("decision_type_id", "UNKNOWN")).strip()

    name = DECISION_TYPE_NAMES.get(decision_type, "Άγνωστος Τύπος")

    folder_name = safe_folder_name(f"{decision_type} - {name}")

    return OUTPUT_ROOT / folder_name


def download_pdf(session: requests.Session, url: str) -> bytes:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Accept": "application/pdf,*/*",
                    "User-Agent": "rag-agent-drive-export/1.0",
                },
            )

            response.raise_for_status()

            content = response.content

            if not content.startswith(b"%PDF"):
                raise RuntimeError(
                    "Downloaded content is not a PDF."
                )

            return content

        except Exception as error:
            last_error = error

            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {last_error}")


def prepare_drive_dataset() -> None:
    records = load_jsonl(DIAVGEIA_DATASET_FILE)

    print(f"Dataset records: {len(records)}")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    completed = 0
    skipped = 0
    failed = []

    try:
        for record in tqdm(
            records,
            desc="Preparing Drive dataset",
            unit="document",
        ):
            ada = str(record.get("ada", "")).strip()

            document_url = str(
                record.get("document_url", "")
            ).strip()

            if not ada or not document_url:
                failed.append(
                    {
                        "ada": ada,
                        "error": "Missing ADA or document URL",
                    }
                )
                continue

            folder = get_folder(record)

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            pdf_path = folder / f"{ada}.pdf"

            # Safe resume.
            if pdf_path.exists():
                skipped += 1
                continue

            try:
                pdf_bytes = download_pdf(
                    session,
                    document_url,
                )

                pdf_path.write_bytes(pdf_bytes)

                completed += 1

            except Exception as error:
                failed.append(
                    {
                        "ada": ada,
                        "error": str(error),
                    }
                )

            time.sleep(REQUEST_DELAY)

    finally:
        session.close()

    print()
    print("DRIVE DATASET PREPARATION COMPLETE")
    print(f"Downloaded: {completed}")
    print(f"Already existing: {skipped}")
    print(f"Failed: {len(failed)}")

    if failed:
        failed_file = OUTPUT_ROOT / "_failed.json"

        failed_file.write_text(
            json.dumps(
                failed,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Failures: {failed_file.resolve()}")


if __name__ == "__main__":
    prepare_drive_dataset()