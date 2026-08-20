from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diavgeia.config import DATASET_FILE


REPAIRED_FILE = Path(
    "data/diavgeia/repaired_documents.jsonl"
)

FINAL_DATASET_FILE = Path(
    "data/diavgeia/final_dataset.jsonl"
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load all JSON objects from a JSONL file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON at line "
                    f"{line_number} in {path}"
                ) from error

            records.append(record)

    return records


def save_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    Write all records to a JSONL file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def merge_repaired_documents() -> None:
    """
    Replace suspicious dataset records with their
    repaired OCR versions.

    The original dataset remains unchanged.
    A new final_dataset.jsonl file is created.
    """

    
    print("DIAVGEIA DATASET MERGE")
    

    
    # Load files
    

    original_records = load_jsonl(
        DATASET_FILE
    )

    repaired_records = load_jsonl(
        REPAIRED_FILE
    )

    print(
        f"Original dataset records: "
        f"{len(original_records)}"
    )

    print(
        f"Repaired records: "
        f"{len(repaired_records)}"
    )

    
    # Index repaired records by ADA
    

    repaired_by_ada = {}

    for record in repaired_records:

        ada = str(
            record.get(
                "ada",
                "",
            )
        ).strip()

        if not ada:
            continue

        repaired_by_ada[ada] = record

    # Merge
    

    final_records = []

    replaced_count = 0

    for original_record in original_records:

        ada = str(
            original_record.get(
                "ada",
                "",
            )
        ).strip()

        if ada in repaired_by_ada:

            final_records.append(
                repaired_by_ada[ada]
            )

            replaced_count += 1

        else:

            final_records.append(
                original_record
            )

    
    # Safety checks
    

    original_adas = {
        str(record.get("ada", "")).strip()
        for record in original_records
        if record.get("ada")
    }

    final_adas = {
        str(record.get("ada", "")).strip()
        for record in final_records
        if record.get("ada")
    }

    repaired_adas = set(
        repaired_by_ada.keys()
    )

    missing_repaired_adas = (
        repaired_adas
        - original_adas
    )

    if missing_repaired_adas:

        print(
            "\nWARNING:"
            " Some repaired ADA values were not "
            "found in the original dataset:"
        )

        for ada in sorted(
            missing_repaired_adas
        ):
            print(
                f"  {ada}"
            )

    if len(final_records) != len(
        original_records
    ):
        raise RuntimeError(
            "Final dataset record count does not "
            "match original dataset record count."
        )

    if final_adas != original_adas:
        raise RuntimeError(
            "ADA integrity check failed."
        )

    
    # Save final dataset
    

    save_jsonl(
        FINAL_DATASET_FILE,
        final_records,
    )

    # Summary
   
    print("MERGE SUMMARY")


    print(
        f"Original records: "
        f"{len(original_records)}"
    )

    print(
        f"Repaired records available: "
        f"{len(repaired_records)}"
    )

    print(
        f"Records replaced: "
        f"{replaced_count}"
    )

    print(
        f"Final records: "
        f"{len(final_records)}"
    )

    print(
        f"Unique ADA values: "
        f"{len(final_adas)}"
    )

    print(
        f"Final dataset: "
        f"{FINAL_DATASET_FILE.resolve()}"
    )




if __name__ == "__main__":
    merge_repaired_documents()