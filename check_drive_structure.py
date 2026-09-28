from collections import Counter
from pathlib import Path

from config import DIAVGEIA_5000_FOLDER_ID
from google_drive_loader import (authenticate_google_drive, collect_pdf_files)


def main():
    service = authenticate_google_drive()

    folder_info = (
        service.files()
        .get(
            fileId=DIAVGEIA_5000_FOLDER_ID,
            fields="id, name",
            supportsAllDrives=True,
        )
        .execute()
    )

    root_name = folder_info.get("name", "")

    print(f"\nRoot folder: {root_name}")

    pdf_files = collect_pdf_files(
        service=service,
        folder_id=DIAVGEIA_5000_FOLDER_ID,
        recursive=True,
        parent_path=root_name,
    )

    print(f"PDF files found: {len(pdf_files)}")

    # ADA = PDF filename without .pdf
    adas = [
        Path(file_info["name"]).stem.strip()
        for file_info in pdf_files
    ]

    unique_adas = set(adas)

    print(f"Unique ADA: {len(unique_adas)}")
    print(
        f"Duplicate ADA: "
        f"{len(adas) - len(unique_adas)}"
    )

    # Extract decision-type folder from:
    # Diavgeia_5000/category/ADA.pdf
    decision_type_folders = []

    for file_info in pdf_files:
        path = file_info.get("virtual_path", "")

        parts = [
            part
            for part in path.split("/")
            if part
        ]

        if len(parts) >= 3:
            decision_type_folders.append(parts[-2])

    counts = Counter(decision_type_folders)

    print(f"Decision-type folders: {len(counts)}")

    print("\nDistribution:")

    for folder, count in counts.most_common():
        print(f"{folder}: {count}")

    print("\nCHECKS")

    print(
        "5000 PDFs:",
        "OK" if len(pdf_files) == 5000
        else "FAIL",
    )

    print(
        "5000 unique ADA:",
        "OK" if len(unique_adas) == 5000
        else "FAIL",
    )

    print(
        "0 duplicate ADA:",
        "OK" if len(adas) == len(unique_adas)
        else "FAIL",
    )

    print(
        "16 decision-type folders:",
        "OK" if len(counts) == 16
        else "FAIL",
    )


if __name__ == "__main__":
    main()