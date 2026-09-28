import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET_FILE = Path("data/diavgeia/final_dataset.jsonl")
OUTPUT_FILE = Path("data/evaluation/manual_retrieval_sample_100.jsonl")
SEED = 42
YEAR_TARGETS = {2021: 16, 2022: 16, 2023: 17, 2024: 17, 2025: 17, 2026: 17}


def clean_space(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def get_year(row):
    for field in ("issue_date", "publish_date", "submission_date"):
        match = re.search(r"(20\d{2})", clean_space(row.get(field)))
        if match and int(match.group(1)) in YEAR_TARGETS:
            return int(match.group(1))

    return None


def extract_focus(text):
    text = str(text or "")
    segments = []

    subject_match = re.search(r"(?is)\bΘΕΜΑ\s*:\s*", text)

    if subject_match:
        start = subject_match.start()
        segments.append(text[start:start + 1800])

    decision_patterns = [r"(?is)\bΑΠΟΦΑΣΙΖΟΥΜΕ\b", r"(?is)\bΑ\s*Π\s*Ο\s*Φ\s*Α\s*Σ\s*Ι\s*Ζ\s*Ο\s*Υ\s*Μ\s*Ε\b", r"(?is)\bΕΓΚΡΙΝΟΥΜΕ\b", r"(?is)\bΕ\s*Γ\s*Κ\s*Ρ\s*Ι\s*Ν\s*Ο\s*Υ\s*Μ\s*Ε\b", r"(?is)\bΑΠΟΦΑΣΙΖΕΙ\b"]

    for pattern in decision_patterns:
        match = re.search(pattern, text)

        if match:
            start = max(0, match.start() - 300)
            segments.append(text[start:start + 2800])
            break

    if not segments:
        segments.append(text[-3000:])

    return clean_space("\n\n".join(segments))


def load_documents():
    documents = []

    with DATASET_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            ada = clean_space(row.get("ada"))
            year = get_year(row)
            decision_type_id = clean_space(row.get("decision_type_id"))
            text = str(row.get("text") or "")

            if not ada or year is None or not decision_type_id or len(text) < 300:
                continue

            documents.append({"ada": ada, "year": year, "decision_type_id": decision_type_id, "protocol_number": clean_space(row.get("protocol_number")), "issue_date": clean_space(row.get("issue_date")), "text": text})

    return documents


def select_documents(documents):
    rng = random.Random(SEED)
    by_type = defaultdict(list)

    for document in documents:
        by_type[document["decision_type_id"]].append(document)

    for items in by_type.values():
        rng.shuffle(items)

    selected = []
    selected_adas = set()
    remaining = dict(YEAR_TARGETS)

    for decision_type_id in sorted(by_type.keys(), key=lambda value: len(by_type[value])):
        options = [item for item in by_type[decision_type_id] if remaining[item["year"]] > 0 and item["ada"] not in selected_adas]

        if not options:
            continue

        max_capacity = max(remaining[item["year"]] for item in options)
        options = [item for item in options if remaining[item["year"]] == max_capacity]
        item = rng.choice(options)

        selected.append(item)
        selected_adas.add(item["ada"])
        remaining[item["year"]] -= 1

    for year in YEAR_TARGETS:
        need = remaining[year]
        pool = [item for item in documents if item["year"] == year and item["ada"] not in selected_adas]
        rng.shuffle(pool)

        if len(pool) < need:
            raise RuntimeError(f"Δεν υπάρχουν αρκετά έγγραφα για το {year}. Χρειάζονται {need}, υπάρχουν {len(pool)}.")

        for item in pool[:need]:
            selected.append(item)
            selected_adas.add(item["ada"])

    selected.sort(key=lambda item: (item["year"], item["decision_type_id"], item["ada"]))

    if len(selected) != 100:
        raise RuntimeError(f"Επιλέχθηκαν {len(selected)} έγγραφα αντί για 100.")

    return selected


def main():
    documents = load_documents()
    selected = select_documents(documents)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for index, item in enumerate(selected, start=1):
            record = {"id": index, "ada": item["ada"], "year": item["year"], "decision_type_id": item["decision_type_id"], "protocol_number": item["protocol_number"], "issue_date": item["issue_date"], "focus": extract_focus(item["text"])}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("DATASET DOCUMENTS:", len(documents))
    print("SELECTED:", len(selected))
    print("UNIQUE ADA:", len({item["ada"] for item in selected}))
    print("YEARS:", dict(sorted(Counter(item["year"] for item in selected).items())))
    print("DECISION TYPES:", dict(sorted(Counter(item["decision_type_id"] for item in selected).items())))
    print("DECISION TYPES COVERED:", len({item["decision_type_id"] for item in selected}))
    print("OUTPUT:", OUTPUT_FILE)


if __name__ == "__main__":
    main()