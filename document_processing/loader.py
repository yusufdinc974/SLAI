
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_documents(raw_directory):
    """Find supported documents recursively."""

    raw_path = Path(raw_directory)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {raw_path}"
        )

    documents = []

    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.name.startswith("~$"):
            continue

        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(file_path)

    return sorted(documents)
