from __future__ import annotations

import io
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf
import requests
import pikepdf
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from diavgeia.config import (
    BUILDER_LOG_FILE,
    DATASET_FILE,
    FAILED_FILE,
    LOG_DIRECTORY,
    LOG_LEVEL,
    MAX_DOCUMENTS,
    MAX_RETRIES,
    METADATA_FILE,
    MIN_DOCUMENT_CHARACTERS,
    MIN_PAGE_CHARACTERS,
    PDF_DIRECTORY,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BASE_SECONDS,
    SAVE_PDFS,
)


# Dataset builder statistics

@dataclass
class BuilderStats:
    """
    Store statistics collected during a dataset-builder execution.

    Attributes:
        metadata_records:
            Number of metadata records loaded from metadata.jsonl.

        documents_considered:
            Number of records examined during the current execution.

        documents_completed:
            Number of new documents successfully written to
            dataset.jsonl.

        already_completed:
            Number of records skipped because their ADA already
            existed in the dataset.

        downloads_attempted:
            Number of document-download operations attempted.

        download_failures:
            Number of documents that could not be downloaded.

        invalid_pdf_files:
            Number of responses that were not valid PDF files.

        encrypted_pdf_files:
            Number of encrypted PDF files that could not be read.

        scanned_or_empty_documents:
            Number of PDFs from which insufficient text was extracted.

        parsing_failures:
            Number of PDF parsing operations that failed unexpectedly.

        failed_documents:
            Total number of records written to failed_documents.jsonl.

        total_pages:
            Number of PDF pages examined successfully.

        pages_with_text:
            Number of pages containing usable extracted text.

        extracted_characters:
            Total number of text characters added to the dataset.
    """

    metadata_records: int = 0
    documents_considered: int = 0
    documents_completed: int = 0
    already_completed: int = 0
    downloads_attempted: int = 0
    download_failures: int = 0
    invalid_pdf_files: int = 0
    encrypted_pdf_files: int = 0
    scanned_or_empty_documents: int = 0
    parsing_failures: int = 0
    failed_documents: int = 0
    total_pages: int = 0
    pages_with_text: int = 0
    extracted_characters: int = 0


# Logging

def configure_logging() -> logging.Logger:
    """
    Configure console and file logging for the dataset builder.

    The logger writes messages to both the terminal and the configured
    builder log file. Existing handlers are reused to avoid duplicate
    output when the module is imported multiple times.

    Returns:
        logging.Logger:
            Configured dataset-builder logger.
    """

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "diavgeia_dataset_builder"
    )

    logger.setLevel(
        getattr(
            logging,
            LOG_LEVEL.upper(),
            logging.INFO,
        )
    )

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        BUILDER_LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logging()


# JSONL helpers

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load valid JSON objects from a JSON Lines file.

    Invalid or empty lines are ignored and reported through the
    logger.

    Args:
        path:
            Source JSONL file.

    Returns:
        list[dict[str, Any]]:
            Successfully decoded JSON records.

    Raises:
        FileNotFoundError:
            If the requested file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(
                    stripped_line
                )

            except json.JSONDecodeError as error:
                LOGGER.warning(
                    "Invalid JSON at line %s of %s: %s",
                    line_number,
                    path,
                    error,
                )
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def append_jsonl_record(
    path: Path,
    record: dict[str, Any],
) -> None:
    """
    Append one JSON object to a JSON Lines file.

    The destination directory is created automatically when needed.

    Args:
        path:
            Destination JSONL file.

        record:
            JSON-serializable record to append.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def load_completed_adas(
    path: Path,
) -> set[str]:
    """
    Load ΑΔΑ identifiers already present in dataset.jsonl.

    This enables safe resume behaviour and prevents the same decision
    from being processed more than once.

    Args:
        path:
            Existing dataset JSONL file.

    Returns:
        set[str]:
            ADA identifiers already completed successfully.
    """

    if not path.exists():
        return set()

    completed_adas: set[str] = set()

    for record in load_jsonl(path):
        ada = record.get("ada")

        if ada:
            completed_adas.add(
                str(ada).strip()
            )

    return completed_adas


def load_failed_adas(
    path: Path,
) -> set[str]:
    """
    Load ΑΔΑ identifiers already recorded as failed.

    The set is used only for reporting. Failed documents are retried
    on later executions unless explicit skip logic is added.

    Args:
        path:
            Failed-document JSONL file.

    Returns:
        set[str]:
            ADA identifiers previously recorded as failed.
    """

    if not path.exists():
        return set()

    failed_adas: set[str] = set()

    for record in load_jsonl(path):
        ada = record.get("ada")

        if ada:
            failed_adas.add(
                str(ada).strip()
            )

    return failed_adas


# Text cleaning


    """
    Apply conservative cleaning to extracted PDF page text.

    The function preserves the semantic content while normalizing
    excessive whitespace, repeated blank lines and null characters.
    It does not remove legal headers, signatures or repeated page
    content, because those may still contain useful administrative
    information.

    Args:
        text:
            Raw text extracted from one PDF page.

    Returns:
        str:
            Cleaned page text.
    """
def clean_page_text(
    text: str,
) -> str:
    if not text:
        return ""

    cleaned = text.replace(
        "\x00",
        " ",
    )

    # Normalize Windows and old Mac line endings.
    cleaned = cleaned.replace("\r\n", "\n",
        ).replace("\r","\n",
    )
    
    # Remove common Diavgeia digital-signature artefacts.
    cleaned = re.sub(
        r"Ministry of Digitally.*?Location:\s*Athens",
        " ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Join words split by line-break hyphenation.
    cleaned = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        cleaned,
    )

    # Replace repeated spaces and tabs while retaining line breaks.
    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    # Remove spaces at line boundaries.
    cleaned = re.sub(
        r" *\n *",
        "\n",
        cleaned,
    )

    # Limit excessive empty lines.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()

# Download helpers


def looks_like_pdf(
    content: bytes,
    content_type: str,
) -> bool:
    """
    Check whether a response appears to contain a PDF document.

    Both the HTTP Content-Type header and the PDF binary signature are
    examined because some servers return generic MIME types.

    Args:
        content:
            Downloaded response body.

        content_type:
            HTTP Content-Type header.

    Returns:
        bool:
            True when the response is likely a PDF.
    """

    normalized_content_type = (
        content_type.lower().strip()
    )

    has_pdf_header = content.startswith(
        b"%PDF"
    )

    has_pdf_content_type = (
        "application/pdf"
        in normalized_content_type
    )

    return (
        has_pdf_header
        or has_pdf_content_type
    )


def download_pdf(
    session: requests.Session,
    document_url: str,
) -> tuple[bytes, str]:
    """
    Download one PDF document with retry and exponential backoff.

    Args:
        session:
            Reusable HTTP session.

        document_url:
            Public Diavgeia document URL.

    Returns:
        tuple[bytes, str]:
            Downloaded bytes and returned Content-Type header.

    Raises:
        RuntimeError:
            If all retry attempts fail or the response body is empty.
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = session.get(
                document_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
                headers={
                    "Accept": "application/pdf,*/*",
                    "User-Agent": (
                        "academic-rag-diavgeia-"
                        "dataset-builder/1.0"
                    ),
                },
            )

            response.raise_for_status()

            content = response.content

            if not content:
                raise RuntimeError(
                    "The downloaded response body is empty."
                )

            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            return content, content_type

        except (
            requests.RequestException,
            RuntimeError,
        ) as error:
            last_error = error

            LOGGER.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                MAX_RETRIES,
                document_url,
                error,
            )

            if attempt >= MAX_RETRIES:
                break

            wait_seconds = (
                RETRY_BASE_SECONDS
                ** (attempt - 1)
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"PDF download failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# PDF parsing

def repair_pdf_with_pikepdf(
    pdf_bytes: bytes,
) -> bytes:
    """
    Rewrite a PDF using pikepdf before text extraction.

    The rewrite may repair malformed PDF objects, streams or structural
    inconsistencies. It does not perform OCR and does not guarantee that
    broken font-to-Unicode mappings will be recovered.

    Args:
        pdf_bytes:
            Original PDF content.

    Returns:
        bytes:
            Rewritten PDF content.

    Raises:
        RuntimeError:
            If pikepdf cannot open or rewrite the document.
    """

    input_buffer = BytesIO(pdf_bytes)
    output_buffer = BytesIO()

    try:
        with pikepdf.open(
            input_buffer
        ) as pdf:
            pdf.save(
                output_buffer,
            )

    except Exception as error:
        raise RuntimeError(
            f"pikepdf could not repair the PDF: {error}"
        ) from error

    return output_buffer.getvalue()

def extract_pdf_pages(
    pdf_bytes: bytes,
) -> tuple[list[str], int]:
    """
    Repair a PDF with pikepdf and extract text using PyMuPDF.

    The PDF is first rewritten by pikepdf. The repaired binary is then
    opened with PyMuPDF, which extracts text page by page.

    Args:
        pdf_bytes:
            Original PDF content stored in memory.

    Returns:
        tuple[list[str], int]:
            A list containing usable page texts and the total number of
            pages in the PDF.

    Raises:
        RuntimeError:
            If the PDF cannot be repaired or opened.
    """


    # 1. Rewrite / repair PDF
    

    repaired_pdf_bytes = repair_pdf_with_pikepdf(
        pdf_bytes
    )

    
    # 2. Open repaired PDF with PyMuPDF
    

    try:
        document = pymupdf.open(
            stream=repaired_pdf_bytes,
            filetype="pdf",
        )

    except Exception as error:
        raise RuntimeError(
            f"PyMuPDF could not open the repaired PDF: {error}"
        ) from error

    total_pages = len(document)

    extracted_pages: list[str] = []

    try:
        for page_number in range(
            total_pages
        ):
            page = document.load_page(
                page_number
            )

            # sort=True attempts to reconstruct a more natural
            # reading order from the PDF text blocks.
            raw_text = page.get_text(
                "text",
                sort=True,
            )

            cleaned_text = clean_page_text(
                raw_text
            )

            if page_number == 0:
                print(cleaned_text[:1000])

            LOGGER.debug(
                "Page %s: extracted %s characters.",
                page_number + 1,
                len(cleaned_text),
            )

            if (
                len(cleaned_text)
                >= MIN_PAGE_CHARACTERS
            ):
                extracted_pages.append(
                    cleaned_text
                )

            else:
                LOGGER.warning(
                    "Page %s produced insufficient text.",
                    page_number + 1,
                )

    finally:
        document.close()

    return extracted_pages, total_pages


def calculate_sha256(
    content: bytes,
) -> str:
    """
    Calculate the SHA-256 checksum of downloaded content.

    The checksum can later be used to identify changed or duplicated
    document binaries.

    Args:
        content:
            Binary document content.

    Returns:
        str:
            Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(
        content
    ).hexdigest()


# Record construction

def build_dataset_record(
    metadata: dict[str, Any],
    pages: list[str],
    total_page_count: int,
    pdf_checksum: str,
    pdf_size_bytes: int,
) -> dict[str, Any]:
    """
    Build one normalized dataset record from metadata and PDF text.

    Args:
        metadata:
            Decision metadata produced by crawler.py.

        pages:
            Cleaned page texts extracted from the PDF.

        total_page_count:
            Total number of pages found in the PDF.

        pdf_checksum:
            SHA-256 checksum of the PDF binary.

        pdf_size_bytes:
            Size of the downloaded PDF in bytes.

    Returns:
        dict[str, Any]:
            Final JSON-serializable dataset record.
    """

    full_text = "\n\n".join(
        pages
    ).strip()

    return {
        "ada": metadata.get("ada"),
        "protocol_number": metadata.get(
            "protocol_number"
        ),
        "subject": metadata.get("subject"),
        "issue_date": metadata.get(
            "issue_date"
        ),
        "publish_date": metadata.get(
            "publish_date"
        ),
        "submission_date": metadata.get(
            "submission_date"
        ),
        "organization_id": metadata.get(
            "organization_id"
        ),
        "unit_ids": metadata.get(
            "unit_ids",
            [],
        ),
        "signer_ids": metadata.get(
            "signer_ids",
            [],
        ),
        "decision_type_id": metadata.get(
            "decision_type_id"
        ),
        "thematic_category_ids": metadata.get(
            "thematic_category_ids",
            [],
        ),
        "status": metadata.get("status"),
        "private_data": metadata.get(
            "private_data",
            False,
        ),
        "version_id": metadata.get(
            "version_id"
        ),
        "document_url": metadata.get(
            "document_url"
        ),
        "api_url": metadata.get(
            "api_url"
        ),
        "total_page_count": total_page_count,
        "text_page_count": len(pages),
        "pdf_size_bytes": pdf_size_bytes,
        "pdf_sha256": pdf_checksum,
        "text_character_count": len(
            full_text
        ),
        "pages": pages,
        "text": full_text,
        "source": "diavgeia",
    }


def build_failed_record(
    metadata: dict[str, Any],
    reason: str,
    error: str,
) -> dict[str, Any]:
    """
    Build a compact failed-document record.

    Args:
        metadata:
            Source decision metadata.

        reason:
            Stable machine-readable failure category.

        error:
            Human-readable error description.

    Returns:
        dict[str, Any]:
            Failed-document JSON record.
    """

    return {
        "ada": metadata.get("ada"),
        "subject": metadata.get("subject"),
        "issue_date": metadata.get(
            "issue_date"
        ),
        "document_url": metadata.get(
            "document_url"
        ),
        "reason": reason,
        "error": error,
        "source": "diavgeia",
    }


# Main dataset-builder workflow


def build_dataset() -> BuilderStats:
    """
    Download Diavgeia PDFs and build a text dataset.

    Metadata records are read from metadata.jsonl. Each corresponding
    PDF is downloaded into memory, validated, parsed and converted into
    a normalized dataset record. Successfully processed decisions are
    appended to dataset.jsonl.

    Existing ADA identifiers are skipped, allowing the builder to
    resume safely after interruption. Failed documents are written to
    failed_documents.jsonl without terminating the full execution.

    Returns:
        BuilderStats:
            Statistics collected during the current builder execution.
    """

    stats = BuilderStats()

    metadata_records = load_jsonl(
        METADATA_FILE
    )

    stats.metadata_records = len(
        metadata_records
    )

    completed_adas = load_completed_adas(
        DATASET_FILE
    )

    previous_failed_adas = load_failed_adas(
        FAILED_FILE
    )

    if MAX_DOCUMENTS is not None:
        metadata_records = metadata_records[
            :MAX_DOCUMENTS
        ]

    
    LOGGER.info("DIAVGEIA DATASET BUILDER")
    LOGGER.info(
        "Metadata records available: %s",
        stats.metadata_records,
    )
    LOGGER.info(
        "Records selected for this run: %s",
        len(metadata_records),
    )
    LOGGER.info(
        "Already completed: %s",
        len(completed_adas),
    )
    LOGGER.info(
        "Previously failed: %s",
        len(previous_failed_adas),
    )
    LOGGER.info(
        "Save PDF files: %s",
        SAVE_PDFS,
    )
    LOGGER.info(
        "Minimum document characters: %s",
        MIN_DOCUMENT_CHARACTERS,
    )

    if SAVE_PDFS:
        PDF_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

    session = requests.Session()

    try:
        with logging_redirect_tqdm(
            loggers=[LOGGER]
        ):
            with tqdm(
                total=len(metadata_records),
                desc="Building dataset",
                unit="document",
                dynamic_ncols=True,
            ) as progress:

                for metadata in metadata_records:
                    stats.documents_considered += 1

                    ada = str(
                        metadata.get("ada", "")
                    ).strip()

                    if not ada:
                        failed_record = build_failed_record(
                            metadata=metadata,
                            reason="missing_ada",
                            error=(
                                "The metadata record does not "
                                "contain an ADA identifier."
                            ),
                        )

                        append_jsonl_record(
                            FAILED_FILE,
                            failed_record,
                        )

                        stats.failed_documents += 1
                        progress.update(1)
                        continue

                    if ada in completed_adas:
                        stats.already_completed += 1

                        progress.set_postfix(
                            {
                                "saved": (
                                    stats.documents_completed
                                ),
                                "existing": (
                                    stats.already_completed
                                ),
                                "failed": (
                                    stats.failed_documents
                                ),
                            },
                            refresh=False,
                        )

                        progress.update(1)
                        continue

                    document_url = str(
                        metadata.get(
                            "document_url",
                            "",
                        )
                    ).strip()

                    if not document_url:
                        failed_record = build_failed_record(
                            metadata=metadata,
                            reason="missing_document_url",
                            error=(
                                "The metadata record does not "
                                "contain a document URL."
                            ),
                        )

                        append_jsonl_record(
                            FAILED_FILE,
                            failed_record,
                        )

                        stats.failed_documents += 1
                        progress.update(1)
                        continue

                    try:
                        stats.downloads_attempted += 1

                        pdf_bytes, content_type = (
                            download_pdf(
                                session=session,
                                document_url=document_url,
                            )
                        )

                    except Exception as error:
                        stats.download_failures += 1
                        stats.failed_documents += 1

                        append_jsonl_record(
                            FAILED_FILE,
                            build_failed_record(
                                metadata=metadata,
                                reason="download_failed",
                                error=str(error),
                            ),
                        )

                        LOGGER.error(
                            "Download failed for ADA %s: %s",
                            ada,
                            error,
                        )

                        progress.update(1)
                        continue

                    if not looks_like_pdf(
                        content=pdf_bytes,
                        content_type=content_type,
                    ):
                        stats.invalid_pdf_files += 1
                        stats.failed_documents += 1

                        append_jsonl_record(
                            FAILED_FILE,
                            build_failed_record(
                                metadata=metadata,
                                reason="invalid_pdf_response",
                                error=(
                                    "The downloaded response did "
                                    "not appear to be a PDF. "
                                    f"Content-Type: {content_type}"
                                ),
                            ),
                        )

                        LOGGER.warning(
                            "Invalid PDF response for ADA %s.",
                            ada,
                        )

                        progress.update(1)
                        continue

                    if SAVE_PDFS:
                        pdf_path = (
                            PDF_DIRECTORY
                            / f"{ada}.pdf"
                        )

                        pdf_path.write_bytes(
                            pdf_bytes
                        )

                    try:
                        pages, total_page_count = (
                            extract_pdf_pages(
                                pdf_bytes
                            )
                        )

                    except RuntimeError as error:
                        stats.encrypted_pdf_files += 1
                        stats.failed_documents += 1

                        append_jsonl_record(
                            FAILED_FILE,
                            build_failed_record(
                                metadata=metadata,
                                reason="encrypted_pdf",
                                error=str(error),
                            ),
                        )

                        LOGGER.warning(
                            "Encrypted PDF for ADA %s: %s",
                            ada,
                            error,
                        )

                        progress.update(1)
                        continue

                    except PyMuPDF as error:
                        stats.parsing_failures += 1
                        stats.failed_documents += 1

                        append_jsonl_record(
                            FAILED_FILE,
                            build_failed_record(
                                metadata=metadata,
                                reason="pdf_read_error",
                                error=str(error),
                            ),
                        )

                        LOGGER.warning(
                            "PDF read error for ADA %s: %s",
                            ada,
                            error,
                        )

                        progress.update(1)
                        continue

                    except Exception as error:
                        stats.parsing_failures += 1
                        stats.failed_documents += 1

                        append_jsonl_record(
                            FAILED_FILE,
                            build_failed_record(
                                metadata=metadata,
                                reason="unexpected_parsing_error",
                                error=str(error),
                            ),
                        )

                        LOGGER.exception(
                            "Unexpected parsing error for ADA %s.",
                            ada,
                        )

                        progress.update(1)
                        continue

                    full_text = "\n\n".join(
                        pages
                    ).strip()

                    stats.total_pages += (
                        total_page_count
                    )

                    stats.pages_with_text += len(
                        pages
                    )

                    if (
                        len(full_text)
                        < MIN_DOCUMENT_CHARACTERS
                    ):
                        stats.scanned_or_empty_documents += 1
                        stats.failed_documents += 1

                        append_jsonl_record(
                            FAILED_FILE,
                            build_failed_record(
                                metadata=metadata,
                                reason="insufficient_extracted_text",
                                error=(
                                    "The PDF contained less than "
                                    f"{MIN_DOCUMENT_CHARACTERS} "
                                    "usable text characters. It may "
                                    "be scanned or image-based."
                                ),
                            ),
                        )

                        LOGGER.warning(
                            "Insufficient text for ADA %s: "
                            "%s characters.",
                            ada,
                            len(full_text),
                        )

                        progress.update(1)
                        continue

                    pdf_checksum = calculate_sha256(
                        pdf_bytes
                    )

                    dataset_record = build_dataset_record(
                        metadata=metadata,
                        pages=pages,
                        total_page_count=total_page_count,
                        pdf_checksum=pdf_checksum,
                        pdf_size_bytes=len(pdf_bytes),
                    )

                    append_jsonl_record(
                        DATASET_FILE,
                        dataset_record,
                    )

                    completed_adas.add(ada)

                    stats.documents_completed += 1
                    stats.extracted_characters += len(
                        full_text
                    )

                    progress.set_postfix(
                        {
                            "saved": (
                                stats.documents_completed
                            ),
                            "existing": (
                                stats.already_completed
                            ),
                            "failed": (
                                stats.failed_documents
                            ),
                        },
                        refresh=False,
                    )

                    progress.update(1)

                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

    except KeyboardInterrupt:
        LOGGER.warning(
            "The dataset builder was interrupted by the user. "
            "Successfully stored records have been preserved."
        )

    finally:
        session.close()

    LOGGER.info("DATASET BUILD SUMMARY")
    LOGGER.info(
        "Metadata records available: %s",
        stats.metadata_records,
    )
    LOGGER.info(
        "Documents considered: %s",
        stats.documents_considered,
    )
    LOGGER.info(
        "Documents completed: %s",
        stats.documents_completed,
    )
    LOGGER.info(
        "Already completed: %s",
        stats.already_completed,
    )
    LOGGER.info(
        "Downloads attempted: %s",
        stats.downloads_attempted,
    )
    LOGGER.info(
        "Download failures: %s",
        stats.download_failures,
    )
    LOGGER.info(
        "Invalid PDF responses: %s",
        stats.invalid_pdf_files,
    )
    LOGGER.info(
        "Encrypted PDFs: %s",
        stats.encrypted_pdf_files,
    )
    LOGGER.info(
        "Scanned or empty documents: %s",
        stats.scanned_or_empty_documents,
    )
    LOGGER.info(
        "PDF parsing failures: %s",
        stats.parsing_failures,
    )
    LOGGER.info(
        "Total failed documents: %s",
        stats.failed_documents,
    )
    LOGGER.info(
        "Total PDF pages: %s",
        stats.total_pages,
    )
    LOGGER.info(
        "Pages containing text: %s",
        stats.pages_with_text,
    )
    LOGGER.info(
        "Extracted characters: %s",
        stats.extracted_characters,
    )
    LOGGER.info(
        "Dataset file: %s",
        DATASET_FILE.resolve(),
    )
    LOGGER.info(
        "Failed file: %s",
        FAILED_FILE.resolve(),
    )
    LOGGER.info(
        "Builder log: %s",
        BUILDER_LOG_FILE.resolve(),
    )

    return stats


if __name__ == "__main__":
    build_dataset()