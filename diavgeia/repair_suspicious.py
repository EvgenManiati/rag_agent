import io
import json
import time

import pymupdf
import pytesseract
import requests
from PIL import Image
from tqdm import tqdm
import unicodedata

from diavgeia.config import (
    NEW_DATASET_FILE,
    NEW_SUSPICIOUS_FILE,
    NEW_REPAIRED_FILE,
    NEW_REPAIR_FAILED_FILE,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    OCR_DPI,
    OCR_LANGUAGES,
)

from diavgeia.quality_check import evaluate_text_quality


def normalize_unicode(text: str) -> str:
    if not text:
        return ""

    decomposed = unicodedata.normalize("NFD", text)

    result = []
    seen_combining = set()

    for char in decomposed:
        if unicodedata.combining(char):
            if char in seen_combining:
                continue

            seen_combining.add(char)
            result.append(char)

        else:
            seen_combining.clear()
            result.append(char)

    return unicodedata.normalize("NFC", "".join(result))

def load_jsonl(path):
    """
    Load JSON records from a JSONL file.
    """

    records = []

    with path.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def save_jsonl(path, record):
    """
    Append one record to a JSONL file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:

        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dataset_by_ada():
    """
    Create a dictionary:
    ADA -> dataset record.
    """

    records = load_jsonl(NEW_DATASET_FILE)

    return {
        str(record["ada"]): record
        for record in records
        if record.get("ada")
    }


def download_pdf(url):
    """
    Download PDF bytes from Diavgeia.
    """

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent":
                "rag-agent-diavgeia-repair/1.0"
        },
    )

    response.raise_for_status()

    return response.content


def clean_text(text):
    """
    Apply minimal whitespace cleaning to OCR output.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def ocr_pdf(pdf_bytes):
    """
    Render every PDF page as an image and run
    Greek + English Tesseract OCR.
    """

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    # 250 DPI
    zoom = OCR_DPI / 72

    matrix = pymupdf.Matrix(zoom, zoom)

    try:

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            image = Image.open(io.BytesIO(pixmap.tobytes("png")))

            text = pytesseract.image_to_string(
                image,
                lang=OCR_LANGUAGES,
                config="--psm 6",
            )

            text = clean_text(text)

            if text:
                pages.append(text)

    finally:
        document.close()

    return pages


def repair_suspicious():
    """
    OCR only the documents previously classified
    as suspicious.
    """

    suspicious = load_jsonl(NEW_SUSPICIOUS_FILE)

    dataset = load_dataset_by_ada()

    # Καθαρίζουμε παλιό αποτέλεσμα repair.
    NEW_REPAIRED_FILE.unlink(missing_ok=True)

    NEW_REPAIR_FAILED_FILE.unlink(missing_ok=True)

    print(f"Suspicious documents: {len(suspicious)}")

    for item in tqdm(suspicious, desc="OCR repair", unit="document"):

        ada = str(item.get("ada", "")).strip()

        original = dataset.get(ada)

        if original is None:

            save_jsonl(NEW_REPAIR_FAILED_FILE,
                    {
                    "ada": ada,
                    "error":
                        "Original dataset record not found",
                },
            )

            continue

        url = original.get("document_url")

        if not url:

            save_jsonl(NEW_REPAIR_FAILED_FILE,
                {
                    "ada": ada,
                    "error":
                        "Document URL not found",
                },
            )

            continue

        try:

            normalized_pages = [normalize_unicode(page) for page in original.get("pages", [])]

            normalized_text = "\n\n".join(normalized_pages).strip()

            quality = evaluate_text_quality(normalized_text)

            if quality["status"] == "good":
                repaired = dict(original)
                repaired["pages"] = normalized_pages
                repaired["text"] = normalized_text
                repaired["text_character_count"] = len(normalized_text)
                repaired["extraction_method"] = "pymupdf_unicode_nfc"
                repaired["repaired"] = True
                save_jsonl(NEW_REPAIRED_FILE, repaired)

                continue

            pdf_bytes = download_pdf(url)

            pages = ocr_pdf(pdf_bytes)

            full_text = "\n\n".join(pages).strip()

            if not full_text:

                raise RuntimeError("OCR produced no text")

            repaired = dict(original)

            repaired["pages"] = pages

            repaired["text"] = full_text

            repaired["text_page_count"] = len(pages)

            repaired["text_character_count"] = len(full_text)

            repaired["extraction_method"] = "tesseract_ocr"

            repaired["repaired"] = True

            save_jsonl(NEW_REPAIRED_FILE, repaired)

        except Exception as error:

            save_jsonl(NEW_REPAIR_FAILED_FILE, {"ada": ada, "document_url": url, "error": str(error)})

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nRepair finished.")

    if NEW_REPAIRED_FILE.exists():

        repaired_count = len(load_jsonl(NEW_REPAIRED_FILE))

        print(f"Repaired documents: {repaired_count}")

    if NEW_REPAIR_FAILED_FILE.exists():

        failed_count = len(load_jsonl(NEW_REPAIR_FAILED_FILE))

        print(f"Failed documents: {failed_count}")


if __name__ == "__main__":
    repair_suspicious()
    