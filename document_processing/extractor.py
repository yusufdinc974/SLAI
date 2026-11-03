
import pymupdf
from docx import Document


def extract_pdf(file_path):
    """Extract text from PDF while preserving page numbers."""

    pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def extract_docx(file_path):
    """Extract paragraphs and tables from DOCX."""

    document = Document(file_path)

    content = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            content.append(text)

    # Extract tables without losing row/column relationships
    for table_index, table in enumerate(document.tables, start=1):
        rows = []

        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))

        content.append(
            f"\n[TABLE {table_index}]\n" +
            "\n".join(rows) +
            "\n[/TABLE]"
        )

    return [{
        "page_number": None,
        "text": "\n".join(content)
    }]


def extract_document(file_path):
    """Choose the correct extraction method."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

import pymupdf  # PyMuPDF
from docx import Document


def extract_pdf(file_path):
    """Extract text from PDF while preserving page numbers."""

    pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def extract_docx(file_path):
    """Extract paragraphs and tables from DOCX."""

    document = Document(file_path)

    content = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            content.append(text)

    # Extract tables without losing row/column relationships
    for table_index, table in enumerate(document.tables, start=1):
        rows = []

        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))

        content.append(
            f"\n[TABLE {table_index}]\n" +
            "\n".join(rows) +
            "\n[/TABLE]"
        )

    return [{
        "page_number": None,
        "text": "\n".join(content)
    }]


def extract_document(file_path):
    """Choose the correct extraction method."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )
