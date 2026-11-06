"""CLI demo script to compare BM25, Dense, and Hybrid Retrieval on OHS queries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from retrieval.hybrid import HybridRetriever


DEFAULT_TEST_QUERIES = [
    "İş sağlığı ve güvenliği kurulu kaç çalışan olan işyerlerinde kurulmalıdır?",
    "İşveren çalışanların iş sağlığı ve güvenliği eğitimlerini hangi sıklıkla yenilemek zorundadır?",
    "Çok tehlikeli sınıfta yer alan işyerlerinde iş güvenliği uzmanının çalışan başına aylık çalışma süresi nedir?",
]


def display_results(method_name: str, results: list, retriever: HybridRetriever, max_chars: int = 140) -> None:
    print(f"\n--- [{method_name}] En İyi {len(results)} Pasaj ---")
    if not results:
        print("  (Sonuç bulunamadı)")
        return

    for idx, res in enumerate(results, start=1):
        info = retriever.get_passage_data(res.passage_id) or {}
        title = info.get("title", "Bilinmeyen Belge")
        marker = info.get("section_marker") or info.get("section_type", "")
        text_preview = (info.get("text") or "").replace("\n", " ").strip()[:max_chars]
        score_str = f"{res.score:.4f}"

        print(f" {idx}. [{res.passage_id}] Skor: {score_str}")
        print(f"    Kaynak: {title} | {marker}")
        print(f"    Özet  : {text_preview}...")


def run_demo(experiment: str = "size_1000_overlap_0", query: str | None = None, top_k: int = 3) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    print("=" * 70)
    print("SLAI ISG - Hybrid Retrieval Entegrasyon Testi")
    print(f"Deney Konfigürasyonu: {experiment}")
    print("=" * 70)

    try:
        retriever = HybridRetriever.from_experiment(
            experiment_name=experiment,
            base_dir=base_dir,
        )
    except FileNotFoundError as err:
        print(f"\nHATA: {err}")
        print("Lütfen önce belge işleme, chunking ve embedding adımlarını tamamlayın.")
        sys.exit(1)

    queries = [query] if query else DEFAULT_TEST_QUERIES

    for q_idx, q in enumerate(queries, start=1):
        print("\n" + "#" * 70)
        print(f"SORU {q_idx}: {q}")
        print("#" * 70)

        comparisons = retriever.compare(q, top_k=top_k, candidate_top_k=20)
        display_results("1. BM25 (Lexical / Kelime Bazlı)", comparisons["bm25"], retriever)
        display_results("2. Dense (BGE-M3 Anlamsal / Qdrant)", comparisons["dense"], retriever)
        display_results("3. Hybrid (BM25 + Dense RRF Birleşimi)", comparisons["hybrid"], retriever)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval comparison demo.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="size_1000_overlap_0",
        help="Experiment name (default: size_1000_overlap_0)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query to test. If omitted, standard OHS benchmark questions are run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to display per method (default: 3)",
    )
    args = parser.parse_args()
    run_demo(experiment=args.experiment, query=args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()

