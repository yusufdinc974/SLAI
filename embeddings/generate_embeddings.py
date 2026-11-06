import json
import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_DIR = ROOT / "data" / "chunks"
OUTPUT_DIR = ROOT / "data" / "embeddings"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16

def generate_embeddings(model, chunks):
    """Chunk metinlerini embedding vektörlerine dönüştürür."""

    texts = []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"{i}. kayıt dict değil.")

        if "passage_id" not in chunk or "text" not in chunk:
            raise ValueError(
                f"Geçersiz chunk: {chunk.get('document_id', 'unknown')} "
                f"| Index: {i} "
                f"| Keys: {list(chunk.keys())}"
            )

        texts.append(chunk["text"])

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    results = []

    for chunk, embedding in zip(chunks, embeddings):
        result = chunk.copy()
        result["embedding"] = embedding.tolist()
        results.append(result)

    return results

def save_embeddings(experiment_name, chunks):
    """Embedding sonuçlarını belge bazında JSON dosyalarına kaydeder."""

    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = {}

    for chunk in chunks:
        document_id = chunk["document_id"]

        if document_id not in documents:
            documents[document_id] = []

        documents[document_id].append(chunk)

    for document_id, document_chunks in documents.items():
        output_path = output_dir / f"{document_id}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                document_chunks,
                f,
                ensure_ascii=False
            )

def load_chunks(experiment_dir):
    """Deney klasöründeki bütün JSON dosyalarını okur."""

    all_chunks = []

    json_files = sorted(experiment_dir.glob("*.json"))

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError(f"Beklenmeyen JSON yapısı: {file_path}")

        all_chunks.extend(chunks)

    return all_chunks
def process_experiment(model, experiment_name):
    print(f"\n{'=' * 50}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 50}")

    experiment_dir = CHUNKS_DIR / experiment_name

    if not experiment_dir.exists():
        raise FileNotFoundError(experiment_dir)

    chunks = load_chunks(experiment_dir)

    print(f"Toplam chunk: {len(chunks)}")

    embeddings = generate_embeddings(model, chunks)

    # Vektör boyutunu kontrol et
    assert all(len(c["embedding"]) == 1024 for c in embeddings)

    save_embeddings(experiment_name, embeddings)

    print(f"Embedding sayısı: {len(embeddings)}")
    print("Vektör boyutu: 1024")
    print(f"Çıktı: data/embeddings/{experiment_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        help="İşlenecek deneyin adı. Belirtilmezse tamamlanmamış deneyler işlenir."
    )

    args = parser.parse_args()

    # İşlenecek deneyleri belirle
    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = sorted(
            d.name
            for d in CHUNKS_DIR.iterdir()
            if d.is_dir()
            and not (OUTPUT_DIR / d.name).exists()
        )

    if not experiments:
        print("İşlenecek yeni deney bulunamadı.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BGE-M3 modeli yükleniyor (Cihaz: {device})...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    model.max_seq_length = 512

    print("Model hazır.")

    for experiment in experiments:
        process_experiment(model, experiment)

    print("\nTüm embedding işlemleri tamamlandı!")


if __name__ == "__main__":
    main()