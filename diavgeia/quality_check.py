from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pathlib import Path 

DATASET_FILE = Path(
    "data/diavgeia/dataset.jsonl"
)

QUALITY_REPORT_FILE = Path(
    "data/diavgeia/quality_report.jsonl"
)

SUSPICIOUS_FILE = Path(
    "data/diavgeia/suspicious_documents.jsonl"
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load JSON objects from a JSON Lines file.
    """

    records = []

    with path.open(
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

            if isinstance(record, dict):
                records.append(record)

    return records


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    """
    Append one record to a JSON Lines file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def calculate_letter_ratio(
    text: str,
) -> float:
    """
    Calculate the ratio of alphabetic characters
    to all non-whitespace characters.
    """

    if not text:
        return 0.0

    visible_chars = [
        char
        for char in text
        if not char.isspace()
    ]

    if not visible_chars:
        return 0.0

    letters = sum(
        1
        for char in visible_chars
        if char.isalpha()
    )

    return letters / len(visible_chars)


def calculate_word_count(
    text: str,
) -> int:
    """
    Estimate the number of normal words.
    """

    words = re.findall(
        r"[A-Za-zΑ-Ωα-ωΆ-Ώά-ώϊϋΐΰ]{2,}",
        text,
    )

    return len(words)


def calculate_corrupted_accent_count(
    text: str,
) -> int:
    """
    Count suspicious Greek combining-accent patterns,
    such as ό́, ά́, ή etc.
    """

    return len(
        re.findall(
            r"[Α-Ωα-ωάέήίόύώϊϋΐΰ][\u0300-\u036f]",
            text,
        )
    )


def calculate_symbol_ratio(
    text: str,
) -> float:
    """
    Calculate how much of the visible text consists
    of punctuation or symbols rather than letters/digits.
    """

    if not text:
        return 1.0

    visible_chars = [
        char
        for char in text
        if not char.isspace()
    ]

    if not visible_chars:
        return 1.0

    symbol_count = sum(
        1
        for char in visible_chars
        if not char.isalnum()
    )

    return symbol_count / len(visible_chars)


def evaluate_text_quality(
    text: str,
) -> dict[str, Any]:
    """
    Calculate text-quality indicators and classify
    the extracted document as good or suspicious.
    """

    text_length = len(text)

    letter_ratio = calculate_letter_ratio(
        text
    )

    word_count = calculate_word_count(
        text
    )

    corrupted_accent_count = (
        calculate_corrupted_accent_count(
            text
        )
    )

    symbol_ratio = calculate_symbol_ratio(
        text
    )

    reasons = []

    if text_length < 100:
        reasons.append(
            "very_short_text"
        )

    if letter_ratio < 0.55:
        reasons.append(
            "low_letter_ratio"
        )

    if word_count < 20:
        reasons.append(
            "very_few_words"
        )

    if corrupted_accent_count >= 10:
        reasons.append(
            "suspicious_combining_accents"
        )

    if symbol_ratio > 0.25:
        reasons.append(
            "high_symbol_ratio"
        )

    status = (
        "suspicious"
        if reasons
        else "good"
    )

    return {
        "status": status,
        "text_length": text_length,
        "letter_ratio": round(
            letter_ratio,
            4,
        ),
        "word_count": word_count,
        "corrupted_accent_count": (
            corrupted_accent_count
        ),
        "symbol_ratio": round(
            symbol_ratio,
            4,
        ),
        "reasons": reasons,
    }


def run_quality_check() -> None:
    """
    Evaluate all extracted documents in dataset.jsonl
    and generate quality reports.
    """

    records = load_jsonl(
        DATASET_FILE
    )

    if not records:
        print(
            "Δεν βρέθηκαν records στο dataset."
        )
        return

    # Καθαρίζουμε προηγούμενα reports.
    QUALITY_REPORT_FILE.unlink(
        missing_ok=True
    )

    SUSPICIOUS_FILE.unlink(
        missing_ok=True
    )

    good_count = 0
    suspicious_count = 0

    reason_counts: dict[str, int] = {}

    for record in records:
        text = str(
            record.get(
                "text",
                "",
            )
        )

        quality = evaluate_text_quality(
            text
        )

        report = {
            "ada": record.get("ada"),
            "subject": record.get(
                "subject"
            ),
            **quality,
        }

        append_jsonl(
            QUALITY_REPORT_FILE,
            report,
        )

        if (
            quality["status"]
            == "suspicious"
        ):
            suspicious_count += 1

            append_jsonl(
                SUSPICIOUS_FILE,
                report,
            )

            for reason in quality[
                "reasons"
            ]:
                reason_counts[reason] = (
                    reason_counts.get(
                        reason,
                        0,
                    )
                    + 1
                )

        else:
            good_count += 1

    
    print("DATASET QUALITY CHECK")
   

    print(
        f"Total documents: {len(records)}"
    )

    print(
        f"Good documents: {good_count}"
    )

    print(
        f"Suspicious documents: "
        f"{suspicious_count}"
    )

    if reason_counts:
        print("\nSuspicious reasons:")

        for reason, count in sorted(
            reason_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"  {reason}: {count}"
            )

    print(
        f"\nQuality report: "
        f"{QUALITY_REPORT_FILE.resolve()}"
    )

    print(
        f"Suspicious documents: "
        f"{SUSPICIOUS_FILE.resolve()}"
    )


if __name__ == "__main__":
    run_quality_check()