from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from diavgeia.config import (
    CRAWL_WINDOW_DAYS,
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


@dataclass
class CrawlStats:
    """
    Statistics collected during a crawler execution.
    """

    windows_processed: int = 0
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
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("diavgeia_crawler")

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

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logging()


# JSONL helpers


def append_jsonl(path: Path, records: list[dict[str, Any]],) -> None:

    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as output_file:

        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_adas(path: Path) -> set[str]:
    if not path.exists():
        return set()

    existing_adas: set[str] = set()

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Invalid JSON at line %s of %s.",
                    line_number,
                    path,
                )
                continue

            ada = record.get("ada")

            if ada:
                existing_adas.add(
                    str(ada)
                )

    return existing_adas


# Date handling

def normalize_date(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        timestamp = float(value)

        # Milliseconds -> seconds
        if timestamp > 10_000_000_000:
            timestamp /= 1000

        try:
            return datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            ).date().isoformat()
        
        except (ValueError, OSError,  OverflowError):
            return None

    text = str(value).strip()

    if not text:
        return None

    if text.isdigit():
        return normalize_date(float(text))

    normalized_text = text.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(normalized_text).date().isoformat()
    
    except ValueError:
        pass

    try:
        return date.fromisoformat(text[:10]).isoformat()
    
    except ValueError:
        return None


def extract_record_date(raw_record: dict[str, Any]) -> str | None:

    raw_date = (
        raw_record.get("issueDate")
        or raw_record.get("issue_date")
        or raw_record.get("submissionTimestamp")
        or raw_record.get("submission_timestamp")
    )

    return normalize_date(raw_date)


def is_inside_date_range(normalized_date: str | None) -> bool:

    if not normalized_date:
        return False

    try:
        record_date = date.fromisoformat(normalized_date)

    except ValueError:
        return False

    return DATE_FROM <= record_date <= DATE_TO


def generate_date_windows(
    start_date: date,
    end_date: date,
    window_days: int,
) -> Iterator[tuple[date, date]]:
    """
    Split the total crawl period into smaller inclusive date windows.
    """

    if window_days <= 0:
        raise ValueError("CRAWL_WINDOW_DAYS must be greater than zero.")

    current_start = start_date

    while current_start <= end_date:
        current_end = min(
            current_start
            + timedelta(days=window_days - 1),
            end_date)

        yield current_start, current_end

        current_start = (current_end + timedelta(days=1))


# API response parsing


def extract_decisions(payload: Any) -> list[dict[str, Any]]:

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

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
            return [item for item in value if isinstance(item, dict)]

    for wrapper_key in ("data", "response"):
        nested = payload.get(wrapper_key)

        if isinstance(nested, dict):
            results = extract_decisions(
                nested
            )

            if results:
                return results

    return []


def normalize_record(raw_record: dict[str, Any], normalized_issue_date: str) -> dict[str, Any] | None:
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
        or raw_record.get("decisionTypeUid"))

    document_url = (
        raw_record.get("documentUrl")
        or DOCUMENT_URL_TEMPLATE.format(
            ada=ada))

    publish_timestamp = normalize_date(
        raw_record.get("publishTimestamp"))

    submission_timestamp = normalize_date(
        raw_record.get("submissionTimestamp"))

    return {
        "ada": ada,
        "protocol_number": raw_record.get(
            "protocolNumber"
        ),
        "issue_date": normalized_issue_date,
        "publish_date": publish_timestamp,
        "submission_date": submission_timestamp,
        "organization_id": str(
            organization_id
        ),
        "unit_ids": raw_record.get("unitIds",[]),

        "signer_ids": raw_record.get("signerIds",[]),

        "decision_type_id": decision_type_id,

        "thematic_category_ids": raw_record.get("thematicCategoryIds", []),

        "status": raw_record.get("status"),
         
        "private_data": raw_record.get("privateData", False),

        "version_id": raw_record.get("versionId"),

        "document_url": document_url,

        "api_url": raw_record.get("url"),

        "source": "diavgeia",
    }



# HTTP request


def request_search_page(
    session: requests.Session,
    page: int,
    window_start: date,
    window_end: date,
) -> list[dict[str, Any]]:
    """
    Retrieve one API page for one specific date window.
    """

    params = {
        "org": ORGANIZATION_UID,
        "from_issue_date": window_start.isoformat(),
        "to_issue_date": window_end.isoformat(),
        "status": DECISION_STATUS,
        "page": page,
        "size": PAGE_SIZE,
        "sort": SORT_ORDER,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                SEARCH_ENDPOINT,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "academic-rag-diavgeia-"
                        "crawler/2.0"
                    ),
                },
            )

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()

            if (
                "json" not in content_type
                and not response.text.strip().startswith(
                    ("{", "[")
                )):
                
                raise RuntimeError(f"API response is not JSON. Content-Type: {content_type}")

            payload = response.json()

            if DEBUG_MODE:
                LOGGER.debug(
                    "Window %s -> %s | page %s | keys=%s",
                    window_start,
                    window_end,
                    page,
                    list(payload.keys())
                    if isinstance(payload, dict)
                    else type(payload).__name__,
                )

            return extract_decisions(
                payload
            )

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as error:
            last_error = error

            LOGGER.warning(
                "Request failed for %s -> %s, "
                "page %s, attempt %s/%s: %s",
                window_start,
                window_end,
                page,
                attempt,
                MAX_RETRIES,
                error,
            )

            if attempt >= MAX_RETRIES:
                break

            wait_seconds = (RETRY_BASE_SECONDS ** (attempt - 1))

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Window {window_start} -> {window_end}, "
        f"page {page} failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )



# Record validation


def validate_raw_record(
    raw_record: dict[str, Any],
    stats: CrawlStats,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    normalized_date = extract_record_date(raw_record)

    if not is_inside_date_range(normalized_date):

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

    if (raw_organization_id and raw_organization_id != ORGANIZATION_UID):

        stats.wrong_organization_records += 1

        return None, {
            "reason": "wrong_organization",
            "ada": raw_record.get("ada"),
            "organization_id": (
                raw_organization_id
            ),
        }

    normalized_record = normalize_record(
        raw_record=raw_record,
        normalized_issue_date=normalized_date,
    )

    if normalized_record is None:
        stats.missing_ada_records += 1

        return None, {"reason": "missing_ada"}

    return normalized_record, None


# Main crawler

def crawl_organization() -> CrawlStats:
    stats = CrawlStats()

    existing_adas = load_existing_adas(METADATA_FILE)

    # Avoid querying dates in the future.
    effective_date_to = min(DATE_TO, date.today())

    windows = list(
        generate_date_windows(
            start_date=DATE_FROM,
            end_date=effective_date_to,
            window_days=CRAWL_WINDOW_DAYS,
        )
    )

    LOGGER.info("DIAVGEIA ORGANIZATION CRAWLER")

    LOGGER.info("Organization UID: %s", ORGANIZATION_UID)

    LOGGER.info("Configured date range: %s to %s", DATE_FROM, DATE_TO)
    
    LOGGER.info("Effective crawl range: %s to %s", DATE_FROM, effective_date_to)

    LOGGER.info("Window size: %s days", CRAWL_WINDOW_DAYS)

    LOGGER.info("Total windows: %s", len(windows))
    
    LOGGER.info("Page size: %s", PAGE_SIZE)

    LOGGER.info("Maximum pages per window: %s", MAX_PAGES)

    LOGGER.info("Existing unique decisions: %s", len(existing_adas))

    session = requests.Session()

    try:
        with logging_redirect_tqdm(loggers=[LOGGER]):

            with tqdm(
                total=None,
                desc="Diavgeia records",
                unit="record",
                dynamic_ncols=True,
            ) as progress:

                for window_number, (window_start, window_end) in enumerate(windows, start=1):

                    LOGGER.info("WINDOW %s/%s: %s -> %s",  window_number, len(windows), window_start, window_end)

                    page = 0

                    while True:
                        if (MAX_PAGES is not None and page >= MAX_PAGES):
                            LOGGER.info(
                                "Reached MAX_PAGES=%s "
                                "for window %s -> %s.",
                                MAX_PAGES,
                                window_start,
                                window_end,
                            )
                            break

                        raw_results = request_search_page(
                            session=session,
                            page=page,
                            window_start=window_start,
                            window_end=window_end,
                        )

                        stats.pages_requested += 1
                        stats.api_records_received += (len(raw_results))

                        if not raw_results:
                            LOGGER.info(
                                "No additional results for "
                                "%s -> %s.",
                                window_start,
                                window_end,
                            )
                            break

                        new_records: list[dict[str, Any]] = []

                        rejected_records: list[dict[str, Any]] = []

                        for raw_record in raw_results:
                            normalized, rejected = (validate_raw_record(raw_record=raw_record, stats=stats))
                               

                            if rejected is not None:
                                rejected_records.append(rejected)

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

                        append_jsonl(path=METADATA_FILE, records=new_records)
                            

                        append_jsonl(path=REJECTED_FILE, records=rejected_records)
                           
                        stats.records_saved += len(new_records)

                        progress.update(len(raw_results))

                        progress.set_postfix(
                            {
                                "window": (
                                    f"{window_number}/"
                                    f"{len(windows)}"
                                ),
                                "year": window_start.year,
                                "saved": stats.records_saved,
                                "dup": stats.duplicates_skipped,
                            },
                            refresh=True,
                        )

                        LOGGER.info(
                            "Window %s -> %s | "
                            "page=%s | API=%s | "
                            "new=%s | duplicates=%s | "
                            "rejected=%s",
                            window_start,
                            window_end,
                            page,
                            len(raw_results),
                            len(new_records),
                            stats.duplicates_skipped,
                            len(rejected_records),
                        )

                        if len(raw_results) < PAGE_SIZE:
                            LOGGER.info(
                                "Final page reached for "
                                "%s -> %s.",
                                window_start,
                                window_end,
                            )
                            break

                        page += 1

                        time.sleep(REQUEST_DELAY_SECONDS)

                    stats.windows_processed += 1

    except KeyboardInterrupt:
        LOGGER.warning( "Crawler interrupted by user.Previously stored metadata was preserved.")

    finally:
        session.close()

    LOGGER.info("CRAWL SUMMARY")
    
    LOGGER.info("Windows processed: %s", stats.windows_processed)

    LOGGER.info("API pages requested: %s", stats.pages_requested)

    LOGGER.info("API records received: %s", stats.api_records_received)

    LOGGER.info("New decisions saved: %s", stats.records_saved)

    LOGGER.info("Duplicates skipped: %s", stats.duplicates_skipped)

    LOGGER.info("Invalid dates: %s", stats.invalid_date_records)
    
    LOGGER.info("Records without ADA: %s", stats.missing_ada_records)
    
    LOGGER.info("Records from another organization: %s", stats.wrong_organization_records)

    LOGGER.info("Total rejected records: %s", stats.rejected_records)

    LOGGER.info("Total unique decisions now stored: %s", len(existing_adas))
    
    LOGGER.info("Metadata file: %s", METADATA_FILE.resolve())

    LOGGER.info("Log file: %s", LOG_FILE.resolve())

    return stats


if __name__ == "__main__":
    crawl_organization()