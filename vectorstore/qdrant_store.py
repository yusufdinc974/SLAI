
import json
import math
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_DIR = (
    BASE_DIR / "data" / "embeddings" / "size_1000_overlap_0"
)

QDRANT_DIR = BASE_DIR / "data" / "qdrant"

# Qdrant configuration
COLLECTION_NAME = "isg_size_1000_overlap_0"
VECTOR_SIZE = 1024
BATCH_SIZE = 128


def create_client():
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)

    return QdrantClient(path=str(QDRANT_DIR))


def create_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' already exists. "
            "Stopping to avoid overwriting existing data."
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

    print(f"Collection created: {COLLECTION_NAME}")


def load_embedding_file(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {file_path.name}")

    return records


def convert_record(record):
    passage_id = record["passage_id"]
    embedding = record["embedding"]

    if len(embedding) != VECTOR_SIZE:
        raise ValueError(
            f"Invalid vector size for passage {passage_id}: "
            f"{len(embedding)}"
        )

    vector = [float(value) for value in embedding]

    if not all(math.isfinite(value) for value in vector):
        raise ValueError(
            f"Non-finite vector value in passage {passage_id}"
        )

    # Qdrant accepts UUID or unsigned integer point IDs.
    # Keep the original passage_id in the payload.
    point_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"isg-rag:{passage_id}")
    )

    payload = {
        key: value
        for key, value in record.items()
        if key != "embedding"
    }

    return models.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
    )


def upload_embeddings(client):
    files = sorted(EMBEDDINGS_DIR.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {EMBEDDINGS_DIR}"
        )

    total_uploaded = 0

    for file_index, file_path in enumerate(files, start=1):
        records = load_embedding_file(file_path)

        points = [
            convert_record(record)
            for record in records
        ]

        for start in range(0, len(points), BATCH_SIZE):
            batch = points[start:start + BATCH_SIZE]

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True
            )

        total_uploaded += len(points)

        print(
            f"[{file_index}/{len(files)}] "
            f"{file_path.name}: {len(points)} points uploaded"
        )

    return total_uploaded


def verify_collection(client, expected_count):
    result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    actual_count = result.count

    print("\n========== VERIFICATION ==========")
    print(f"Expected points: {expected_count}")
    print(f"Actual points:   {actual_count}")

    if actual_count != expected_count:
        raise RuntimeError("Point count verification failed!")

    print("Verification successful.")


def main():
    client = create_client()

    try:
        print("Creating Qdrant collection...")
        create_collection(client)

        print("\nUploading embeddings...")
        total_uploaded = upload_embeddings(client)

        verify_collection(client, total_uploaded)

    finally:
        client.close()


if __name__ == "__main__":
    main()
