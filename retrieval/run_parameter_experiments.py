"""Parameter sensitivity experiments and result file generator for SLAI Retrieval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from .bm25 import BM25Index, RetrievalResult, extract_legal_references
from .export_results import save_runs
from .hybrid import HybridRetriever
from .rrf import reciprocal_rank_fusion

EXPERIMENTS = [
    "size_500_overlap_0",
    "size_1000_overlap_0",
    "size_1000_overlap_100",
    "size_1000_overlap_200",
    "size_1500_overlap_0",
]

BM25_PARAM_GRID = [
    (1.2, 0.50),
    (1.2, 0.75),
    (1.5, 0.50),
    (1.5, 0.75),
    (1.5, 0.90),
    (2.0, 0.75),
]

RRF_PARAM_GRID = [
    (10, (1.0, 1.0)),
    (30, (1.0, 1.0)),
    (60, (1.0, 1.0)),
    (60, (1.0, 2.0)),
    (60, (2.0, 1.0)),
]

BENCHMARK_CASES = [
    {
        "question_id": "q001",
        "name": "Soru 1 (İSG Kurulu Çalışan Sayısı)",
        "query": "İş sağlığı ve güvenliği kurulu kaç çalışan olan işyerlerinde kurulmalıdır?",
        "expected_doc": "1.5.6331",
        "expected_marker": "MADDE 22",
    },
    {
        "question_id": "q002",
        "name": "Soru 2 (İSG Eğitim Yenileme Sıklığı)",
        "query": "İşveren çalışanların iş sağlığı ve güvenliği eğitimlerini hangi sıklıkla yenilemek zorundadır?",
        "expected_doc": "1.5.6331",
        "expected_marker": "MADDE 17",
    },
    {
        "question_id": "q003",
        "name": "Soru 3 (Çok Tehlikeli Sınıfta İG Uzmanı Çalışma Süresi)",
        "query": "Çok tehlikeli sınıfta yer alan işyerlerinde iş güvenliği uzmanının çalışan başına aylık çalışma süresi nedir?",
        "expected_doc": "7.5.16923",
        "expected_marker": "MADDE 12",
    },
    {
        "question_id": "q004",
        "name": "Soru 4 (Madde Numaralı Sorgu - 6331 Madde 22)",
        "query": "6331 sayılı Kanun Madde 22 hükümlerine göre kurul yükümlülüğü nedir?",
        "expected_doc": "1.5.6331",
        "expected_marker": "MADDE 22",
    },
]


def load_experiment_chunks(base_dir: Path, experiment: str = "size_1000_overlap_0") -> dict[str, dict[str, Any]]:
    chunks_dir = base_dir / "data" / "chunks" / experiment
    passages: dict[str, dict[str, Any]] = {}
    for file_path in sorted(chunks_dir.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                passages[rec["passage_id"]] = rec
    return passages


def load_reference_vector(
    base_dir: Path,
    experiment: str,
    target_doc: str,
    target_marker: str,
) -> list[float]:
    emb_dir = base_dir / "data" / "embeddings" / experiment
    for file_path in sorted(emb_dir.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                if rec.get("title") == target_doc and rec.get("section_marker") == target_marker:
                    return rec["embedding"]
    raise ValueError(f"Reference vector not found for {target_doc} {target_marker}")


def write_csv_and_json(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_bm25_parameter_study(passages: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    corpus = {pid: rec["text"] for pid, rec in passages.items()}
    rows: list[dict[str, Any]] = []

    for k1, b in BM25_PARAM_GRID:
        idx = BM25Index(corpus, k1=k1, b=b)
        for case in BENCHMARK_CASES:
            hits = idx.search(case["query"], top_k=5)
            top1 = passages[hits[0].passage_id] if hits else {}
            rows.append({
                "k1": k1,
                "b": b,
                "question_id": case["question_id"],
                "case": case["name"],
                "top1_passage_id": hits[0].passage_id if hits else "",
                "top1_source": f"{top1.get('title', '')} | {top1.get('section_marker', '')}",
                "top1_score": round(hits[0].score, 4) if hits else 0.0,
            })
    return rows


def run_rrf_parameter_study(
    bm25_ranking: list[RetrievalResult],
    dense_ranking: list[RetrievalResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rrf_k, weights in RRF_PARAM_GRID:
        fused = reciprocal_rank_fusion(
            [bm25_ranking, dense_ranking],
            k=rrf_k,
            weights=list(weights),
            top_k=5,
        )
        rows.append({
            "rrf_k": rrf_k,
            "weights_bm25_dense": f"{weights[0]:.0f}:{weights[1]:.0f}",
            "top1_passage_id": fused[0].passage_id if fused else "",
            "top3_ids": ",".join(r.passage_id for r in fused[:3]),
            "top1_score": round(fused[0].score, 6) if fused else 0.0,
        })
    return rows


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    results_root = base_dir / "results" / "retrieval"
    results_root.mkdir(parents=True, exist_ok=True)

    passages_1000 = load_experiment_chunks(base_dir, "size_1000_overlap_0")
    print(f"Yüklenen pasaj sayısı (size_1000_overlap_0): {len(passages_1000)}")

    # 1. BM25 (k1, b) Parametre Deneyi
    bm25_rows = run_bm25_parameter_study(passages_1000)
    write_csv_and_json(
        bm25_rows,
        results_root / "bm25_parameter_experiments.csv",
        results_root / "bm25_parameter_experiments.json",
    )

    # 2. 5 Chunking Deneyi Karşılaştırması
    chunk_exp_rows: list[dict[str, Any]] = []
    for exp_name in EXPERIMENTS:
        exp_passages = load_experiment_chunks(base_dir, exp_name)
        exp_ret = HybridRetriever(exp_passages, qdrant_client=None, collection_name=f"isg_{exp_name}")
        for case in BENCHMARK_CASES:
            hits = exp_ret.search_bm25(case["query"], top_k=5, article_boost=10.0)
            top1 = exp_passages[hits[0].passage_id]
            chunk_exp_rows.append({
                "experiment": exp_name,
                "total_passages": len(exp_passages),
                "question_id": case["question_id"],
                "case": case["name"],
                "top1_passage_id": hits[0].passage_id,
                "top1_source": f"{top1.get('title', '')} | {top1.get('section_marker', '')}",
                "top1_score": round(hits[0].score, 4),
            })
    write_csv_and_json(
        chunk_exp_rows,
        results_root / "chunking_5_experiments_comparison.csv",
        results_root / "chunking_5_experiments_comparison.json",
    )

    # 3. 5 Deney İçin Qdrant Dense + BM25 + Hybrid (RRF) Çıktıları (Yusuf ve Zeynep için)
    client = QdrantClient(path=str(base_dir / "data" / "qdrant"))
    try:
        all_rrf_rows: list[dict[str, Any]] = []
        for exp_name in EXPERIMENTS:
            exp_passages = load_experiment_chunks(base_dir, exp_name)
            exp_ret = HybridRetriever(exp_passages, qdrant_client=client, collection_name=f"isg_{exp_name}")
            exp_runs: list[dict[str, Any]] = []
            zeynep_contexts: list[dict[str, Any]] = []

            for case in BENCHMARK_CASES:
                bm25_hits = exp_ret.search_bm25(case["query"], top_k=20, article_boost=10.0)
                q_vec = load_reference_vector(base_dir, exp_name, case["expected_doc"], case["expected_marker"])
                q_points = client.query_points(
                    collection_name=f"isg_{exp_name}",
                    query=q_vec,
                    limit=20,
                ).points
                dense_hits = [
                    RetrievalResult(p.payload["passage_id"], float(p.score))
                    for p in q_points
                ]
                hybrid_hits = HybridRetriever.fuse(bm25_hits, dense_hits, top_k=5)

                if exp_name == "size_1000_overlap_0":
                    for rrf_row in run_rrf_parameter_study(bm25_hits, dense_hits):
                        all_rrf_rows.append({"question_id": case["question_id"], "case": case["name"], **rrf_row})

                per_method = {
                    "bm25": bm25_hits[:5],
                    "dense": dense_hits[:5],
                    "hybrid": hybrid_hits[:5],
                }
                for method_name, method_hits in per_method.items():
                    exp_runs.append({
                        "question_id": case["question_id"],
                        "question": case["query"],
                        "method": method_name,
                        "experiment": exp_name,
                        "results": [
                            {"rank": idx, "passage_id": r.passage_id, "score": round(r.score, 6)}
                            for idx, r in enumerate(method_hits, start=1)
                        ],
                    })

                zeynep_contexts.append({
                    "question_id": case["question_id"],
                    "question": case["query"],
                    "experiment": exp_name,
                    "methods": {
                        m_name: [
                            {
                                "rank": idx,
                                "passage_id": r.passage_id,
                                "score": round(r.score, 6),
                                "title": exp_passages[r.passage_id].get("title"),
                                "section_marker": exp_passages[r.passage_id].get("section_marker"),
                                "chapter_title": exp_passages[r.passage_id].get("chapter_title"),
                                "text": exp_passages[r.passage_id].get("text"),
                            }
                            for idx, r in enumerate(m_hits, start=1)
                        ]
                        for m_name, m_hits in per_method.items()
                    },
                })

            exp_out_dir = results_root / exp_name
            save_runs(exp_runs, exp_out_dir)
            with (exp_out_dir / "llm_contexts.json").open("w", encoding="utf-8") as f:
                json.dump(zeynep_contexts, f, ensure_ascii=False, indent=2)

        write_csv_and_json(
            all_rrf_rows,
            results_root / "rrf_parameter_experiments.csv",
            results_root / "rrf_parameter_experiments.json",
        )
    finally:
        client.close()

    print("\nTüm çıktı dosyaları başarıyla oluşturuldu:")
    for p in sorted(results_root.rglob("*")):
        if p.is_file():
            print(f"  - {p.relative_to(base_dir)}")


if __name__ == "__main__":
    main()
