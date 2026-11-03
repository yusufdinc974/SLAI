
import json
import hashlib

from pathlib import Path
from datetime import datetime, timezone

from document_processing.loader import load_documents
from document_processing.extractor import extract_document
from document_processing.cleaner import clean_pages


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def get_document_category(file_path):
    relative_path = file_path.relative_to(RAW_DIR)

    if len(relative_path.parts) > 1:
        return relative_path.parts[0]

    return "uncategorized"


def get_document_id(file_path):
    relative_path = file_path.relative_to(RAW_DIR)

    return hashlib.sha256(
        str(relative_path).encode("utf-8")
    ).hexdigest()[:12]


def process_document(file_path):
    print(f"\nProcessing: {file_path.name}")

    pages = extract_document(file_path)
    pages = clean_pages(pages)

    full_text = "\n\n".join(
        page["text"] for page in pages
    )

    empty_pages = [
        page["page_number"]
        for page in pages
        if not page["text"].strip()
    ]

    document = {
        "document_id": get_document_id(file_path),
        "title": file_path.stem,
        "file_name": file_path.name,
        "file_type": file_path.suffix.lower(),
        "category": get_document_category(file_path),
        "source_path": str(file_path.relative_to(RAW_DIR)),
        "file_modified_at": datetime.fromtimestamp(
            file_path.stat().st_mtime,
            tz=timezone.utc
        ).isoformat(),
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "language": "tr",
        "page_count": len(pages),
        "character_count": len(full_text),
        "empty_pages": empty_pages,
        "needs_ocr": bool(empty_pages),
        "pages": pages,
        "full_text": full_text
    }

    return document


def save_document(document):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DIR /
        f"{document['document_id']}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            document,
            file,
            ensure_ascii=False,
            indent=4
        )

    return output_path


def main():
    documents = load_documents(RAW_DIR)

    if not documents:
        print(f"No supported documents found in {RAW_DIR}")
        return

    print(f"Found {len(documents)} documents.")

    successful = 0
    failed = 0

    for file_path in documents:
        try:
            document = process_document(file_path)
            output_path = save_document(document)

            successful += 1
            print(f"Saved: {output_path.name}")

        except Exception as error:
            failed += 1
            print(f"ERROR {file_path.name}: {error}")

    print("\nDocument processing completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()