
import json
import math
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
QDRANT_DIR = BASE_DIR / "data" / "qdrant"

VECTOR_SIZE = 1024
BATCH_SIZE = 128


def load_records(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"Invalid JSON structure: {file_path}")

    return records


def convert_record(record):
    passage_id = record["passage_id"]
    embedding = record["embedding"]

    if len(embedding) != VECTOR_SIZE:
        raise ValueError(f"Invalid vector size: {passage_id}")

    vector = [float(value) for value in embedding]

    if not all(math.isfinite(v) for v in vector):
        raise ValueError(f"Invalid vector value: {passage_id}")

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


def index_experiment(client, experiment_dir):

    experiment_name = experiment_dir.name
    collection_name = f"isg_{experiment_name}"

    print(f"\n{'=' * 55}")
    print(f"Experiment: {experiment_name}")
    print(f"Collection: {collection_name}")

    if client.collection_exists(collection_name):
        existing_count = client.count(
            collection_name=collection_name,
            exact=True
        ).count

        print(
            f"Collection already exists ({existing_count} points). "
            "Skipping to protect existing data."
        )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

    files = sorted(experiment_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {experiment_dir}"
        )

    total_uploaded = 0

    for file_index, file_path in enumerate(files, start=1):

        records = load_records(file_path)

        points = [
            convert_record(record)
            for record in records
        ]

        for start in range(0, len(points), BATCH_SIZE):
            batch = points[start:start + BATCH_SIZE]

            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )

        total_uploaded += len(points)

        print(
            f"[{file_index}/{len(files)}] "
            f"{file_path.name}: {len(points)} uploaded"
        )

    actual_count = client.count(
        collection_name=collection_name,
        exact=True
    ).count

    print(f"\nExpected: {total_uploaded}")
    print(f"Actual:   {actual_count}")

    if actual_count != total_uploaded:
        raise RuntimeError(
            f"Verification failed: {collection_name}"
        )

    print("Verification successful!")


def main():

    QDRANT_DIR.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_DIR))

    try:
        experiment_dirs = sorted(
            path for path in EMBEDDINGS_DIR.iterdir()
            if path.is_dir()
        )

        if not experiment_dirs:
            raise FileNotFoundError("No experiment directories found.")

        for experiment_dir in experiment_dirs:
            index_experiment(client, experiment_dir)

        print("\n========== ALL EXPERIMENTS ==========")

        for collection in client.get_collections().collections:
            count = client.count(
                collection_name=collection.name,
                exact=True
            ).count

            print(f"{collection.name}: {count} points")

    finally:
        client.close()


if __name__ == "__main__":
    main()
