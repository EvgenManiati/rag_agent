import json
import re
from pathlib import Path

import requests


DATASET_FILE = Path("data/diavgeia/final_dataset.jsonl")
OUTPUT_DIR = Path("data/drive_upload/diavgeia")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(value: str) -> str:
    """Αφαιρεί χαρακτήρες που δεν επιτρέπονται σε Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", value).strip()


def download_pdf(record: dict) -> bool:
    if record.get("source") != "diavgeia":
        return False

    ada = record.get("ada")
    document_url = record.get("document_url")

    if not ada or not document_url:
        print("SKIP: λείπει ADA ή document_url")
        return False

    filename = safe_filename(f"{ada}.pdf")
    output_path = OUTPUT_DIR / filename

    # Δεν το ξανακατεβάζουμε αν υπάρχει ήδη.
    if output_path.exists():
        print(f"EXISTS: {filename}")
        return True

    try:
        response = requests.get(
            document_url,
            timeout=60,
        )

        response.raise_for_status()

        # Απλός έλεγχος ότι πήραμε PDF και όχι HTML error page.
        if not response.content.startswith(b"%PDF"):
            print(f"NOT PDF: {ada}")
            return False

        output_path.write_bytes(response.content)

        print(f"OK: {filename}")
        return True

    except Exception as exc:
        print(f"ERROR: {ada} -> {exc}")
        return False


def main():
    total = 0
    downloaded = 0
    skipped_external = 0

    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("source") != "diavgeia":
                skipped_external += 1
                continue

            total += 1

            if download_pdf(record):
                downloaded += 1

    print("DOWNLOAD COMPLETE")
    print(f"Diavgeia records: {total}")
    print(f"PDFs available:   {downloaded}")
    print(f"External skipped: {skipped_external}")
    print(f"Folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()