from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from diavgeia.config import (
    DATE_FROM,
    DATE_TO,
    DEBUG_MODE,
    DECISION_STATUS,
    DOCUMENT_URL_TEMPLATE,
    LOG_DIRECTORY,
    LOG_FILE,
    LOG_LEVEL,
    MAX_PAGES,
    MAX_RETRIES,
    METADATA_FILE,
    ORGANIZATION_UID,
    PAGE_SIZE,
    REJECTED_FILE,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BASE_SECONDS,
    SEARCH_ENDPOINT,
    SORT_ORDER,
)

# Στατιστικά εκτέλεσης

@dataclass
class CrawlStats:

    """
    Store statistics collected during a crawler execution.

    Attributes:
        pages_requested:
            Number of API pages successfully requested.

        api_records_received:
            Total number of raw records returned by the API.

        records_saved:
            Number of new and valid records written to metadata.jsonl.

        duplicates_skipped:
            Number of records skipped because their ADA identifier
            was already present in the dataset.

        invalid_date_records:
            Number of records rejected because their date was missing,
            invalid or outside the configured date range.

        missing_ada_records:
            Number of records rejected because no ADA was available.

        wrong_organization_records:
            Number of records rejected because they belonged to a
            different organization.

        rejected_records:
            Total number of records written to the rejected metadata
            file.
    """

    pages_requested: int = 0
    api_records_received: int = 0
    records_saved: int = 0
    duplicates_skipped: int = 0
    invalid_date_records: int = 0
    missing_ada_records: int = 0
    wrong_organization_records: int = 0
    rejected_records: int = 0


# Logging

def configure_logging() -> logging.Logger:
    """
    Configure console and file logging for the crawler.

    The logger writes messages both to the terminal and to the
    configured crawler log file. Existing handlers are reused to
    prevent duplicate log messages when the module is imported more
    than once.

    Returns:
        logging.Logger:
            The configured crawler logger.
    """

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "diavgeia_crawler"
    )

    logger.setLevel(
        getattr(
            logging,
            LOG_LEVEL.upper(),
            logging.INFO,
        )
    )

    # Αποφεύγει duplicate handlers όταν το module
    # φορτώνεται περισσότερες από μία φορές.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logging()


# JSONL helpers

def append_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    Append records to a JSON Lines file.

    Each record is serialized as an independent JSON object on a
    separate line. The destination directory is created automatically
    when it does not already exist.

    Args:
        path:
            Destination JSONL file.

        records:
            Records that will be appended to the file.
    """

    if not records:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_existing_adas(
    path: Path,
) -> set[str]:
    """
    Load all ADA identifiers already stored in a JSONL file.

    Invalid JSON lines are ignored and reported through the logger.
    The returned set is used to prevent duplicate decisions from being
    stored during later executions.

    Args:
        path:
            Metadata JSONL file.

    Returns:
        set[str]:
            Unique ADA identifiers already stored in the file.
    """

    if not path.exists():
        return set()

    existing_adas: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Μη έγκυρο JSON στη γραμμή %s του %s.",
                    line_number,
                    path,
                )
                continue

            ada = record.get("ada")

            if ada:
                existing_adas.add(str(ada))

    return existing_adas


# Date handling

def normalize_date(
    value: Any,
) -> str | None:
    """
    Convert a Diavgeia date value to ISO format.

    Supported formats include Unix timestamps in seconds or
    milliseconds, numeric timestamps represented as strings and ISO
    date or datetime strings.

    Args:
        value:
            Raw date value returned by the API.

    Returns:
        str | None:
            Date in YYYY-MM-DD format, or None when conversion fails.
    """

    if value is None:
        return None

    if isinstance(value, (int, float)):
        timestamp = float(value)

        # Milliseconds → seconds
        if timestamp > 10_000_000_000:
            timestamp /= 1000

        try:
            return datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            ).date().isoformat()
        except (
            ValueError,
            OSError,
            OverflowError,
        ):
            return None

    text = str(value).strip()

    if not text:
        return None

    if text.isdigit():
        return normalize_date(float(text))

    normalized_text = text.replace(
        "Z",
        "+00:00",
    )

    try:
        return datetime.fromisoformat(
            normalized_text
        ).date().isoformat()
    except ValueError:
        pass

    try:
        return date.fromisoformat(
            text[:10]
        ).isoformat()
    except ValueError:
        return None


def extract_record_date(
    raw_record: dict[str, Any],
) -> str | None:
    """
    Extract and normalize the most appropriate decision date.

    The issue date is preferred. Submission timestamps are used only
    as a fallback when an issue date is not available.

    Args:
        raw_record:
            Raw decision object returned by the API.

    Returns:
        str | None:
            Normalized date in YYYY-MM-DD format.
    """

    raw_date = (
        raw_record.get("issueDate")
        or raw_record.get("issue_date")
        or raw_record.get("submissionTimestamp")
        or raw_record.get("submission_timestamp")
    )

    return normalize_date(raw_date)


def is_inside_date_range(
    normalized_date: str | None,
) -> bool:
    """
    Check whether a normalized date is inside the configured range.

    Args:
        normalized_date:
            Date in YYYY-MM-DD format.

    Returns:
        bool:
            True when the date is valid and inside the configured
            interval; otherwise False.
    """

    if not normalized_date:
        return False

    try:
        record_date = date.fromisoformat(
            normalized_date
        )
    except ValueError:
        return False

    return DATE_FROM <= record_date <= DATE_TO


# API response parsing

def extract_decisions(
    payload: Any,
) -> list[dict[str, Any]]:
    """
    Extract decision records from an API response payload.

    The function supports direct lists and common response wrapper
    fields such as decisions, results, items, documents, data and
    response.

    Args:
        payload:
            Parsed JSON response returned by the API.

    Returns:
        list[dict[str, Any]]:
            Raw decision records found in the response.
    """

    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if not isinstance(payload, dict):
        return []

    for key in (
        "decisions",
        "results",
        "items",
        "documents",
    ):
        value = payload.get(key)

        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    # Μερικά APIs επιστρέφουν nested result object.
    for wrapper_key in (
        "data",
        "response",
    ):
        nested = payload.get(wrapper_key)

        if isinstance(nested, dict):
            results = extract_decisions(nested)

            if results:
                return results

    return []


def normalize_record(
    raw_record: dict[str, Any],
    normalized_issue_date: str,
) -> dict[str, Any] | None:
    """
    Convert a raw Diavgeia decision into the metadata schema.

    Only fields required for document downloading, filtering,
    traceability and later RAG processing are retained.

    Args:
        raw_record:
            Raw decision returned by the API.

        normalized_issue_date:
            Decision date already converted to YYYY-MM-DD.

    Returns:
        dict[str, Any] | None:
            Normalized metadata record, or None if the decision does
            not contain an ADA identifier.
    """

    ada = (
        raw_record.get("ada")
        or raw_record.get("ADA")
        or raw_record.get("iun")
    )

    if not ada:
        return None

    ada = str(ada).strip()

    organization_id = (
        raw_record.get("organizationId")
        or raw_record.get("organizationUid")
        or ORGANIZATION_UID
    )

    decision_type_id = (
        raw_record.get("decisionTypeId")
        or raw_record.get("decisionTypeUid")
    )

    document_url = (
        raw_record.get("documentUrl")
        or DOCUMENT_URL_TEMPLATE.format(
            ada=ada
        )
    )

    publish_timestamp = normalize_date(
        raw_record.get("publishTimestamp")
    )

    submission_timestamp = normalize_date(
        raw_record.get("submissionTimestamp")
    )

    return {
        "ada": ada,
        "protocol_number": raw_record.get(
            "protocolNumber"
        ),
        "subject": raw_record.get("subject"),
        "issue_date": normalized_issue_date,
        "publish_date": publish_timestamp,
        "submission_date": submission_timestamp,
        "organization_id": str(
            organization_id
        ),
        "unit_ids": raw_record.get(
            "unitIds",
            [],
        ),
        "signer_ids": raw_record.get(
            "signerIds",
            [],
        ),
        "decision_type_id": decision_type_id,
        "thematic_category_ids": raw_record.get(
            "thematicCategoryIds",
            [],
        ),
        "status": raw_record.get("status"),
        "private_data": raw_record.get(
            "privateData",
            False,
        ),
        "version_id": raw_record.get(
            "versionId"
        ),
        "document_url": document_url,
        "api_url": raw_record.get("url"),
        "source": "diavgeia",
    }

# HTTP request

def request_search_page(
    session: requests.Session,
    page: int,
) -> list[dict[str, Any]]:
    """
        Retrieve one paginated result set from the Diavgeia API.

    Requests are filtered by organization, issue-date range and
    publication status. Temporary failures are retried using
    exponential backoff.

    Args:
        session:
            Reusable HTTP session.

        page:
            Zero-based page number to retrieve.

    Returns:
        list[dict[str, Any]]:
            Raw decision records returned for the requested page.

    Raises:
        RuntimeError:
            If all retry attempts fail or the response is not valid
            JSON.
    """

    params = {
        "org": ORGANIZATION_UID,
        "from_issue_date": DATE_FROM.isoformat(),
        "to_issue_date": DATE_TO.isoformat(),
        "status": DECISION_STATUS,
        "page": page,
        "size": PAGE_SIZE,
        "sort": SORT_ORDER,
    }

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = session.get(
                SEARCH_ENDPOINT,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "academic-rag-diavgeia-"
                        "crawler/1.0"
                    ),
                },
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            if (
                "json" not in content_type
                and not response.text.strip().startswith(
                    ("{", "[")
                )
            ):
                raise RuntimeError(
                    "Το API δεν επέστρεψε JSON. "
                    f"Content-Type: {content_type}"
                )

            payload = response.json()

            if DEBUG_MODE:
                LOGGER.debug(
                    "Page %s response keys: %s",
                    page,
                    list(payload.keys())
                    if isinstance(payload, dict)
                    else type(payload).__name__,
                )

            return extract_decisions(payload)

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as error:
            last_error = error

            LOGGER.warning(
                "Αποτυχία σελίδας %s, προσπάθεια %s/%s: %s",
                page,
                attempt,
                MAX_RETRIES,
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
        f"Η σελίδα {page} απέτυχε μετά από "
        f"{MAX_RETRIES} προσπάθειες: {last_error}"
    )


# Record validation

def validate_raw_record(
    raw_record: dict[str, Any],
    stats: CrawlStats,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
Validate and normalize a raw decision record.

    The function verifies the decision date, organization identifier
    and ADA. Invalid records are converted into compact rejected-record
    objects containing the rejection reason.

    Args:
        raw_record:
            Raw decision returned by the API.

        stats:
            Statistics object updated when validation fails.

    Returns:
        tuple:
            A pair containing:

            - the normalized decision, or None;
            - the rejected-record description, or None.
    """

    normalized_date = extract_record_date(
        raw_record
    )

    if not is_inside_date_range(
        normalized_date
    ):
        stats.invalid_date_records += 1

        return None, {
            "reason": "invalid_or_out_of_range_date",
            "ada": raw_record.get("ada"),
            "raw_date": raw_record.get(
                "issueDate"
            ),
        }

    raw_organization_id = str(
        raw_record.get("organizationId")
        or raw_record.get("organizationUid")
        or ""
    )

    if (
        raw_organization_id
        and raw_organization_id
        != ORGANIZATION_UID
    ):
        stats.wrong_organization_records += 1

        return None, {
            "reason": "wrong_organization",
            "ada": raw_record.get("ada"),
            "organization_id": raw_organization_id,
        }

    normalized_record = normalize_record(
        raw_record=raw_record,
        normalized_issue_date=normalized_date,
    )

    if normalized_record is None:
        stats.missing_ada_records += 1

        return None, {
            "reason": "missing_ada",
            "subject": raw_record.get(
                "subject"
            ),
        }

    return normalized_record, None


# Main crawler

def crawl_organization() -> CrawlStats:
    """
    Collect all published decisions of the configured organization.

    The crawler retrieves paginated decision metadata from the
    Diavgeia Open Data API, validates each record, removes duplicates
    based on the ADA identifier and appends valid records to a JSONL
    metadata file.

    Existing ADA identifiers are loaded before crawling, allowing the
    process to resume safely after an interruption without storing the
    same decision twice.

    Returns:
        CrawlStats:
            Statistics collected during the current crawler execution,
            including requested pages, received records, saved records,
            duplicates and rejected records.
    """

    stats = CrawlStats()

    existing_adas = load_existing_adas(
        METADATA_FILE
    )

    LOGGER.info("DIAVGEIA ORGANIZATION CRAWLER")
    LOGGER.info(
        "Organization UID: %s",
        ORGANIZATION_UID,
    )
    LOGGER.info(
        "Date range: %s to %s",
        DATE_FROM,
        DATE_TO,
    )
    LOGGER.info(
        "Page size: %s",
        PAGE_SIZE,
    )
    LOGGER.info(
        "Maximum pages: %s",
        MAX_PAGES,
    )
    LOGGER.info(
        "Existing unique decisions: %s",
        len(existing_adas),
    )

    session = requests.Session()
    page = 0

    # When MAX_PAGES is known, tqdm can display a percentage.
    # Otherwise, it displays the number of processed API records.
    progress_total = (
        MAX_PAGES * PAGE_SIZE
        if MAX_PAGES is not None
        else None
    )

    try:
        # Redirect logger output through tqdm so that log messages
        # do not visually overwrite the progress bar.
        with logging_redirect_tqdm(
            loggers=[LOGGER]
        ):
            with tqdm(
                total=progress_total,
                desc="Diavgeia records",
                unit="record",
                dynamic_ncols=True,
            ) as progress:

                while True:
                    if (
                        MAX_PAGES is not None
                        and page >= MAX_PAGES
                    ):
                        LOGGER.info(
                            "Reached the configured limit "
                            "of %s pages.",
                            MAX_PAGES,
                        )
                        break

                    LOGGER.info(
                        "Retrieving API page %s...",
                        page,
                    )

                    raw_results = request_search_page(
                        session=session,
                        page=page,
                    )

                    stats.pages_requested += 1
                    stats.api_records_received += len(
                        raw_results
                    )

                    if not raw_results:
                        LOGGER.info(
                            "No additional results were returned."
                        )
                        break

                    new_records: list[
                        dict[str, Any]
                    ] = []

                    rejected_records: list[
                        dict[str, Any]
                    ] = []

                    for raw_record in raw_results:
                        normalized, rejected = (
                            validate_raw_record(
                                raw_record=raw_record,
                                stats=stats,
                            )
                        )

                        if rejected is not None:
                            rejected_records.append(
                                rejected
                            )
                            stats.rejected_records += 1
                            continue

                        if normalized is None:
                            continue

                        ada = normalized["ada"]

                        if ada in existing_adas:
                            stats.duplicates_skipped += 1
                            continue

                        existing_adas.add(ada)
                        new_records.append(normalized)

                    append_jsonl(
                        path=METADATA_FILE,
                        records=new_records,
                    )

                    append_jsonl(
                        path=REJECTED_FILE,
                        records=rejected_records,
                    )

                    stats.records_saved += len(
                        new_records
                    )

                    # Update the progress bar according to the number
                    # of API records processed on the current page.
                    progress.update(
                        len(raw_results)
                    )

                    progress.set_postfix(
                        {
                            "page": page,
                            "saved": stats.records_saved,
                            "duplicates": (
                                stats.duplicates_skipped
                            ),
                            "rejected": (
                                stats.rejected_records
                            ),
                        },
                        refresh=True,
                    )

                    LOGGER.info(
                        "Page %s: API=%s, new=%s, "
                        "duplicates=%s, rejected=%s",
                        page,
                        len(raw_results),
                        len(new_records),
                        stats.duplicates_skipped,
                        len(rejected_records),
                    )

                    # Fewer records than PAGE_SIZE normally indicates
                    # that the final API page has been reached.
                    if len(raw_results) < PAGE_SIZE:
                        LOGGER.info(
                            "The final API page was reached."
                        )
                        break

                    page += 1

                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

    except KeyboardInterrupt:
        LOGGER.warning(
            "The crawler was interrupted by the user. "
            "Previously stored metadata has been preserved."
        )

    finally:
        session.close()

    LOGGER.info("CRAWL SUMMARY")
    LOGGER.info(
        "API pages requested: %s",
        stats.pages_requested,
    )
    LOGGER.info(
        "API records received: %s",
        stats.api_records_received,
    )
    LOGGER.info(
        "New decisions saved: %s",
        stats.records_saved,
    )
    LOGGER.info(
        "Duplicates skipped: %s",
        stats.duplicates_skipped,
    )
    LOGGER.info(
        "Invalid dates: %s",
        stats.invalid_date_records,
    )
    LOGGER.info(
        "Records without ADA: %s",
        stats.missing_ada_records,
    )
    LOGGER.info(
        "Records from another organization: %s",
        stats.wrong_organization_records,
    )
    LOGGER.info(
        "Total rejected records: %s",
        stats.rejected_records,
    )
    LOGGER.info(
        "Metadata file: %s",
        METADATA_FILE.resolve(),
    )
    LOGGER.info(
        "Log file: %s",
        LOG_FILE.resolve(),
    )
    

    return stats


if __name__ == "__main__":
    crawl_organization()