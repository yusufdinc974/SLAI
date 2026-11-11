"""Run BM25, Dense and Hybrid retrieval over a question set and export results.

The output is meant to be consumed by the evaluation module (Recall@k, MRR).

Input question file (JSON list or JSONL), one object per question:
    {"question_id": "q001", "question": "İSG kurulu kaç çalışanda kurulur?"}
Any extra fields (e.g. gold passage IDs, category) are ignored here.

Output, written to results/retrieval/<experiment>/:
    runs.jsonl  one line per (question, method):
                {"question_id", "method", "experiment",
                 "results": [{"rank", "passage_id", "score"}, ...]}
    runs.csv    flat table: question_id, method, rank, passage_id, score

Usage:
    python -m retrieval.export_results --questions data/questions/dev.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .bm25 import RetrievalResult

METHODS = ("bm25", "dense", "hybrid")


def load_questions(path: str | Path) -> list[dict[str, str]]:
    """Load questions from a JSON list, JSONL, or CSV file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
    elif text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    questions = []
    seen = set()
    for index, record in enumerate(records):
        if "question_id" not in record or "question" not in record:
            raise ValueError(
                f"Record {index} must contain 'question_id' and 'question': {record}"
            )
        qid = str(record["question_id"])
        if qid in seen:
            raise ValueError(f"Duplicate question_id: {qid}")
        seen.add(qid)
        questions.append({"question_id": qid, "question": str(record["question"])})
    return questions


def run_questions(
    retriever: Any,
    questions: Iterable[dict[str, str]],
    *,
    experiment: str,
    top_k: int = 10,
    candidate_top_k: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve with all three methods for every question."""
    runs = []
    for question in questions:
        bm25 = retriever.search_bm25(question["question"], top_k=candidate_top_k)
        dense = retriever.search_dense(question["question"], top_k=candidate_top_k)
        hybrid = retriever.fuse(bm25, dense, top_k=top_k)

        per_method: dict[str, list[RetrievalResult]] = {
            "bm25": bm25[:top_k],
            "dense": dense[:top_k],
            "hybrid": hybrid,
        }
        for method in METHODS:
            runs.append({
                "question_id": question["question_id"],
                "method": method,
                "experiment": experiment,
                "results": [
                    {"rank": rank, "passage_id": r.passage_id, "score": r.score}
                    for rank, r in enumerate(per_method[method], start=1)
                ],
            })
    return runs


def save_runs(runs: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    """Write runs as JSONL and as a flat CSV table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "runs.jsonl"
    csv_path = output_dir / "runs.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "method", "experiment", "rank", "passage_id", "score"])
        for run in runs:
            for item in run["results"]:
                writer.writerow([
                    run["question_id"], run["method"], run["experiment"],
                    item["rank"], item["passage_id"], f"{item['score']:.6f}",
                ])

    return jsonl_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BM25/Dense/Hybrid retrieval runs.")
    parser.add_argument("--questions", required=True, help="JSON or JSONL question file")
    parser.add_argument("--experiment", default="size_1000_overlap_0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--output-dir", default=None,
                        help="Default: results/retrieval/<experiment>")
    args = parser.parse_args()

    from .hybrid import HybridRetriever

    base_dir = Path(__file__).resolve().parents[1]
    questions = load_questions(args.questions)
    print(f"{len(questions)} soru yüklendi.")

    retriever = HybridRetriever.from_experiment(args.experiment, base_dir)
    runs = run_questions(
        retriever, questions,
        experiment=args.experiment,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
    )

    output_dir = args.output_dir or base_dir / "results" / "retrieval" / args.experiment
    jsonl_path, csv_path = save_runs(runs, output_dir)
    print(f"Kaydedildi:\n  {jsonl_path}\n  {csv_path}")


if __name__ == "__main__":
    main()

