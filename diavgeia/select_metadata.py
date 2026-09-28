import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from diavgeia.config import NEW_SELECTED_METADATA_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "diavgeia"

METADATA_FILE = DATA_DIR / "metadata.jsonl"
FINAL_DATASET_FILE = DATA_DIR / "final_dataset.jsonl"

TARGET_TOTAL = 5000
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)

RANDOM_SEED = 42
MIN_PER_TYPE = 3


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

    return rows


def get_year(record: dict) -> int | None:
    date_value = (
        record.get("issue_date")
        or record.get("publish_date")
        or record.get("submission_date")
    )

    if not date_value:
        return None

    try:
        return int(str(date_value)[:4])
    except ValueError:
        return None


def get_decision_type(record: dict) -> str:
    decision_type = record.get("decision_type_id")

    if decision_type is None:
        return "__UNKNOWN__"

    return str(decision_type)


def deduplicate_by_ada(rows: list[dict]) -> list[dict]:
    unique_rows = []
    seen_adas = set()

    for row in rows:
        ada = row.get("ada")

        if not ada:
            continue

        if ada in seen_adas:
            continue

        seen_adas.add(ada)
        unique_rows.append(row)

    return unique_rows


def calculate_year_targets(existing_counts: Counter, target_total: int) -> dict[int, int]:
    existing_total = sum(existing_counts[year] for year in YEARS)

    if existing_total == 0:
        raise ValueError("Existing dataset contains no valid yearly records.")

    raw_targets = {}

    for year in YEARS:
        proportion = existing_counts[year] / existing_total
        raw_targets[year] = proportion * target_total

    targets = {year: int(raw_targets[year]) for year in YEARS}

    remaining = target_total - sum(targets.values())

    years_by_remainder = sorted(
        YEARS,
        key=lambda year: raw_targets[year] - int(raw_targets[year]),
        reverse=True,
    )

    for year in years_by_remainder[:remaining]:
        targets[year] += 1

    return targets


def diverse_sample(
    candidates: list[dict],
    amount: int,
    rng: random.Random
    ) -> list[dict]:
    if amount <= 0:
        return []

    if len(candidates) < amount:
        raise ValueError(f"Not enough candidates: requested {amount}, available {len(candidates)}")

    by_type = defaultdict(list)

    for record in candidates:
        by_type[get_decision_type(record)].append(record)

    for records in by_type.values():
        rng.shuffle(records)

    selected = []
    selected_adas = set()

    type_keys = list(by_type.keys())
    rng.shuffle(type_keys)

    # Phase 1:
    # Give representation to as many decision types as possible.
    for _ in range(MIN_PER_TYPE):
        for decision_type in type_keys:
            if len(selected) >= amount:
                break

            bucket = by_type[decision_type]

            while bucket:
                record = bucket.pop()
                ada = record["ada"]

                if ada not in selected_adas:
                    selected.append(record)
                    selected_adas.add(ada)
                    break

        if len(selected) >= amount:
            break

    # Phase 2:
    # Fill the remaining positions randomly from all remaining records.
    remaining_pool = []

    for bucket in by_type.values():
        remaining_pool.extend(bucket)

    rng.shuffle(remaining_pool)

    for record in remaining_pool:
        if len(selected) >= amount:
            break

        ada = record["ada"]

        if ada in selected_adas:
            continue

        selected.append(record)
        selected_adas.add(ada)

    if len(selected) != amount:
        raise RuntimeError(
            f"Selection failed: expected {amount}, selected {len(selected)}")

    return selected


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    print("Loading existing final dataset...")
    existing_rows = load_jsonl(FINAL_DATASET_FILE)

    print("Loading metadata pool...")
    metadata_rows = load_jsonl(METADATA_FILE)

    existing_rows = deduplicate_by_ada(existing_rows)
    metadata_rows = deduplicate_by_ada(metadata_rows)

    existing_adas = {row["ada"] for row in existing_rows if row.get("ada")}

    existing_counts = Counter(get_year(row) for row in existing_rows if get_year(row) in YEARS)

    print()
    print(f"Existing records: {len(existing_rows)}")
    print(f"Existing unique ADA: {len(existing_adas)}")
    print(f"Metadata records: {len(metadata_rows)}")

    if len(existing_rows) >= TARGET_TOTAL:
        raise ValueError(
            f"Dataset already contains {len(existing_rows)} records, which is >= target {TARGET_TOTAL}.")

    target_counts = calculate_year_targets(existing_counts, TARGET_TOTAL,)

    needed_counts = {
        year: target_counts[year] - existing_counts[year]
        for year in YEARS
    }

    total_needed = TARGET_TOTAL - len(existing_rows)

    print()
    print("Year distribution:")

    print(
        f"{'Year':<8}"
        f"{'Existing':>10}"
        f"{'Target':>10}"
        f"{'Needed':>10}"
    )
 

    for year in YEARS:
        print(
            f"{year:<8}"
            f"{existing_counts[year]:>10}"
            f"{target_counts[year]:>10}"
            f"{needed_counts[year]:>10}"
        )

    print(
        f"{'TOTAL':<8}"
        f"{len(existing_rows):>10}"
        f"{TARGET_TOTAL:>10}"
        f"{total_needed:>10}"
    )

    candidates_by_year = defaultdict(list)

    for record in metadata_rows:
        ada = record.get("ada")

        if not ada:
            continue

        if ada in existing_adas:
            continue

        year = get_year(record)

        if year not in YEARS:
            continue

        candidates_by_year[year].append(record)

    print()
    print("Available new candidates:")
    

    for year in YEARS:
        print(
            f"{year}: "
            f"{len(candidates_by_year[year])} candidates"
        )

    selected = []

    print()
    print("Selecting new metadata...")

    for year in YEARS:
        amount = needed_counts[year]

        if amount <= 0:
            continue

        available = len(candidates_by_year[year])

        if available < amount:
            raise ValueError(
                f"{year}: need {amount} records "
                f"but only {available} candidates are available."
            )

        year_selection = diverse_sample(
            candidates_by_year[year],
            amount,
            rng,
        )

        selected.extend(year_selection)

        print(
            f"{year}: selected "
            f"{len(year_selection)} / {amount}"
        )

    selected_adas = [row["ada"] for row in selected]

    if len(selected) != total_needed:
        raise RuntimeError(f"Expected {total_needed} new records, selected {len(selected)}.")

    if len(selected_adas) != len(set(selected_adas)):
        raise RuntimeError("Duplicate ADA found inside selected metadata.")

    overlap = existing_adas.intersection(selected_adas)

    if overlap:
        raise RuntimeError(
            f"Found {len(overlap)} ADA already present in final_dataset.jsonl.")

    save_jsonl(selected, NEW_SELECTED_METADATA_FILE)

    selected_counts = Counter(get_year(row) for row in selected)

    selected_types = Counter(get_decision_type(row) for row in selected)

    print()
    print("SELECTION COMPLETE")
   

    print(f"Existing dataset: {len(existing_rows)}")
    print(f"New selected:     {len(selected)}")
    print(f"Final target:     {len(existing_rows) + len(selected)}")

    print()
    print("Selected by year:")

    for year in YEARS:
        print(f"  {year}: {selected_counts[year]}")

    print()
    print(f"Decision types represented: {len(selected_types)}")

    print(f"Unique selected ADA: {len(set(selected_adas))}")

    print(f"Overlap with existing dataset: {len(overlap)}")

    print()
    print(f"Saved to: {NEW_SELECTED_METADATA_FILE}")


if __name__ == "__main__":
    main()