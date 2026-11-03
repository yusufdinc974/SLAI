
import json
import re
import hashlib
import argparse

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"


EXPERIMENTS = [
    {"chunk_size": 500, "overlap": 0},
    {"chunk_size": 1000, "overlap": 0},
    {"chunk_size": 1500, "overlap": 0},
    {"chunk_size": 1000, "overlap": 100},
    {"chunk_size": 1000, "overlap": 200},
]


# Mevzuat madde ve ek başlangıçları
SECTION_PATTERN = re.compile(
    r"^(?P<marker>"
    r"(?:GEÇİCİ|EK)\s+MADDE\s+\d+(?:\s*/\s*[A-Z])?"
    r"|MADDE\s+\d+(?:\s*/\s*[A-Z])?"
    r"|EK\s*[-–—]?\s*\d+"
    r")\s*(?:[-–—:]|$)",
    re.IGNORECASE
)

PARAGRAPH_PATTERN = re.compile(r"^\((\d+)\)")
CLAUSE_PATTERN = re.compile(r"^([a-zçğıöşü])\)")


def parse_sections(text):
    """
    Mevzuat metnini MADDE, GEÇİCİ MADDE, EK MADDE ve EK
    yapılarına ayırır. Bölüm başlıklarını metadata olarak saklar.
    """

    lines = text.splitlines()

    sections = []
    current_lines = []
    current_type = "intro"
    current_number = None
    current_marker = None
    current_chapter_title = None

    chapter_pattern = re.compile(
        r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|"
        r"ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|"
        r"ONUNCU|ON BİRİNCİ|ON İKİNCİ)"
        r"\s+(BÖLÜM|KISIM)\s*$",
        re.IGNORECASE
    )

    def save_section():
        if current_lines:
            sections.append({
                "section_type": current_type,
                "section_number": current_number,
                "section_marker": current_marker,
                "chapter_title": current_chapter_title,
                "text": "\n".join(current_lines).strip()
            })

    i = 0

    while i < len(lines):

        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue

        # Bölüm başlığı kontrolü
        if chapter_pattern.match(stripped):

            save_section()
            current_lines = []

            current_chapter_title = None

            # Sonraki dolu satır bölüm başlığı olabilir.
            j = i + 1

            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                next_line = lines[j].strip()

                if (
                    not SECTION_PATTERN.match(next_line)
                    and not chapter_pattern.match(next_line)
                ):
                    current_chapter_title = next_line
                    i = j

            i += 1
            continue

        # MADDE / EK başlangıcı kontrolü
        match = SECTION_PATTERN.match(stripped)

        if match:

            heading = None

            # Önceki satır kısa bir madde başlığıysa al.
            if current_lines:
                previous = current_lines[-1].strip()

                if (
                    previous
                    and len(previous) < 100
                    and not previous.endswith((".", ":", ";"))
                    and not SECTION_PATTERN.match(previous)
                ):
                    heading = current_lines.pop()

            save_section()

            marker = match.group("marker")
            normalized_marker = marker.upper()
            if re.match(r"^GEÇİCİ\s+MADDE", marker, re.IGNORECASE):
                current_type = "temporary_article"

            elif re.match(r"^EK\s+MADDE", marker, re.IGNORECASE):
                current_type = "additional_article"

            elif re.match(r"^MADDE", marker, re.IGNORECASE):
                current_type = "article"

            else:
                current_type = "annex"

            number_match = re.search(r"\d+", marker)

            current_number = (
                number_match.group(0) if number_match else None
            )

            current_marker = marker
            current_lines = []

            if heading:
                current_lines.append(heading)

            current_lines.append(stripped)

        else:
            current_lines.append(stripped)

        i += 1

    save_section()

    return [
        section for section in sections
        if section["text"]
    ]

def split_long_unit(text, chunk_size):
    """
    Uzun metinleri önce cümle, gerekirse kelime sınırlarından böler.
    Çok kısa son parçalar oluşmasını önlemek için uzun cümleleri dengeli böler.
    """

    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)

    parts = []
    current = ""

    for sentence in sentences:

        if current and len(current) + len(sentence) + 1 > chunk_size:
            parts.append(current)
            current = sentence
        else:
            current = (
                current + " " + sentence if current else sentence
            )

    if current:
        parts.append(current)

    result = []

    for part in parts:

        while len(part) > chunk_size:

            # Normalde chunk_size sınırında kes.
            cut_position = chunk_size

            # Kalan parça çok küçük olacaksa metni dengeli böl.
            if len(part) - chunk_size < 200:
                cut_position = len(part) // 2

            # Kelimenin ortasında kesmemek için son boşluğu bul.
            whitespace_position = part.rfind(
                " ",
                max(1, cut_position - 100),
                cut_position + 1
            )

            if whitespace_position > 0:
                cut_position = whitespace_position

            result.append(part[:cut_position].strip())
            part = part[cut_position:].strip()

        if part:
            result.append(part)

    return result

def extract_units(section_text, chunk_size):

    lines = [
        line.strip()
        for line in section_text.splitlines()
        if line.strip()
    ]

    units = []
    current = ""

    for line in lines:

        # Yalnızca (1), (2), (3) gibi fıkra başlangıçlarında böl.
        starts_new_paragraph = bool(
            re.match(r"^\(\d+\)", line)
        )

        if starts_new_paragraph and current:
            units.extend(split_long_unit(current, chunk_size))
            current = line

        else:
            current = (
                current + "\n" + line if current else line
            )

    if current:
        units.extend(split_long_unit(current, chunk_size))

    return units



def build_chunks(units, chunk_size, overlap):

    chunks = []
    start = 0

    while start < len(units):

        current_units = []
        current_length = 0
        end = start

        # Chunk boyutuna sığan birimleri ekle.
        while end < len(units):

            unit = units[end]
            added_length = len(unit) + (1 if current_units else 0)

            if current_units and current_length + added_length > chunk_size:
                break

            current_units.append(unit)
            current_length += added_length
            end += 1

        if not current_units:
            break

        chunks.append("\n".join(current_units))

        if end >= len(units):
            break

        # Overlap için son birimleri sonraki chunk'a taşı.
        overlap_units = []
        overlap_length = 0

        for unit in reversed(current_units):

            if overlap_length >= overlap:
                break

            overlap_units.insert(0, unit)
            overlap_length += len(unit) + 1

        next_start = end - len(overlap_units)

        # İlerlemeyi garanti et.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def extract_paragraph_numbers(text):
    return list(dict.fromkeys(
        re.findall(r"\((\d+)\)", text)
    ))


def create_passage_id(document_id, experiment_name, index):
    raw_id = f"{document_id}_{experiment_name}_{index}"

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()[:20]


def create_document_chunks(document, chunk_size, overlap):

    sections = parse_sections(document.get("full_text", ""))

    experiment_name = (
        f"size_{chunk_size}_overlap_{overlap}"
    )

    passages = []

    for section in sections:

        units = extract_units(section["text"], chunk_size)

        section_chunks = build_chunks(
            units,
            chunk_size,
            overlap
        )

        for chunk_text in section_chunks:

            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            passage_index = len(passages)

            passages.append({
                "passage_id": create_passage_id(
                    document["document_id"],
                    experiment_name,
                    passage_index
                ),
                "document_id": document["document_id"],
                "title": document["title"],
                "category": document["category"],
                "source_path": document["source_path"],
                "section_type": section["section_type"],
                "section_number": section["section_number"],
                "section_marker": section["section_marker"],
                "chapter_title": section["chapter_title"],
                "paragraph_numbers": extract_paragraph_numbers(
                    chunk_text
                ),
                "passage_index": passage_index,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "character_count": len(chunk_text),
                "text": chunk_text
            })

    return passages


def run_experiment(chunk_size, overlap):

    experiment_name = (
        f"size_{chunk_size}_overlap_{overlap}"
    )

    output_dir = CHUNKS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(PROCESSED_DIR.glob("*.json"))

    total_passages = 0

    print(f"\n{'=' * 50}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"{'=' * 50}")

    for file_path in files:

        try:
            with file_path.open("r", encoding="utf-8") as file:
                document = json.load(file)

            passages = create_document_chunks(
                document,
                chunk_size,
                overlap
            )

            output_path = output_dir / file_path.name

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    passages,
                    file,
                    ensure_ascii=False,
                    indent=4
                )

            total_passages += len(passages)

            print(
                f"{document['title']}: "
                f"{len(passages)} passages"
            )

        except Exception as error:
            print(f"ERROR {file_path.name}: {error}")

    print(f"\nTotal documents: {len(files)}")
    print(f"Total passages: {total_passages}")
    print(f"Output: {output_dir}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk boyutu"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap miktarı"
    )

    args = parser.parse_args()

    if args.chunk_size is not None:

        if args.chunk_size <= 0:
            parser.error("Chunk size pozitif olmalıdır.")

        if args.overlap < 0 or args.overlap >= args.chunk_size:
            parser.error(
                "Overlap negatif olamaz ve chunk size değerinden küçük olmalıdır."
            )

        run_experiment(args.chunk_size, args.overlap)

    else:

        for experiment in EXPERIMENTS:
            run_experiment(
                experiment["chunk_size"],
                experiment["overlap"]
            )


if __name__ == "__main__":
    main()
