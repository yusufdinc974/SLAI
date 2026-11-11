"""Hybrid Retrieval pipeline combining BM25 and Dense Qdrant search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bm25 import BM25Index, RetrievalResult, extract_legal_references
from .rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Combines BM25 lexical search and BGE-M3 Dense search using RRF."""

    def __init__(
        self,
        passages: Mapping[str, str | dict[str, Any]],
        qdrant_client: Any,
        collection_name: str,
        embedding_model: Any = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize retriever with passage corpus, Qdrant client, and model.

        Args:
            passages: Dict mapping passage_id to either plain text or chunk dict.
            qdrant_client: An initialized QdrantClient instance.
            collection_name: Target Qdrant collection name (e.g. 'isg_size_1000_overlap_0').
            embedding_model: Loaded SentenceTransformer model or model name string.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.passage_data: dict[str, dict[str, Any]] = {}
        bm25_corpus: dict[str, str] = {}

        for pid, val in passages.items():
            if isinstance(val, dict):
                self.passage_data[pid] = val
                bm25_corpus[pid] = val.get("text", "")
            else:
                self.passage_data[pid] = {"passage_id": pid, "text": str(val)}
                bm25_corpus[pid] = str(val)

        self.bm25_index = BM25Index(bm25_corpus, k1=k1, b=b)
        self.client = qdrant_client
        self.collection_name = collection_name

        if isinstance(embedding_model, str):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
        else:
            self.model = embedding_model

    @classmethod
    def from_experiment(
        cls,
        experiment_name: str,
        base_dir: str | Path,
        qdrant_client: Any = None,
        model: Any = "BAAI/bge-m3",
    ) -> HybridRetriever:
        """Factory method to load retriever directly from data/chunks and data/qdrant.

        Args:
            experiment_name: Experiment folder name, e.g. 'size_1000_overlap_0'.
            base_dir: Project root directory (containing data/ folder).
            qdrant_client: Optional QdrantClient instance. If None, connects to data/qdrant.
            model: SentenceTransformer model instance or model name string.
        """
        base_path = Path(base_dir)
        chunks_dir = base_path / "data" / "chunks" / experiment_name
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

        passages: dict[str, dict[str, Any]] = {}
        for file_path in sorted(chunks_dir.glob("*.json")):
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list) or not all(
                isinstance(rec, dict) and "passage_id" in rec and "text" in rec
                for rec in records
            ):
                raise ValueError(
                    f"{file_path} is not a chunk file (expected a list of "
                    "objects with 'passage_id' and 'text'). It may be a "
                    "data/processed document copied by mistake; regenerate "
                    "chunks with: python -m document_processing.chunking.chunker"
                )
            for rec in records:
                passages[rec["passage_id"]] = rec

        if not passages:
            raise FileNotFoundError(f"No chunks found in {chunks_dir}")

        if qdrant_client is None:
            from qdrant_client import QdrantClient
            qdrant_path = base_path / "data" / "qdrant"
            qdrant_client = QdrantClient(path=str(qdrant_path))

        collection_name = f"isg_{experiment_name}"
        return cls(
            passages=passages,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedding_model=model,
        )

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
        *,
        article_boost: float = 10.0,
    ) -> list[RetrievalResult]:
        """Perform lexical BM25 search with optional metadata boost for article/law numbers."""
        refs = extract_legal_references(query)
        if article_boost <= 0 or (not refs["articles"] and not refs["laws"]):
            return self.bm25_index.search(query, top_k=top_k)

        total = max(top_k, self.bm25_index.passage_count)
        base_results = self.bm25_index.search(query, top_k=total)
        boosted: list[RetrievalResult] = []

        for res in base_results:
            meta = self.passage_data.get(res.passage_id, {})
            bonus = 0.0
            sec_num = str(meta.get("section_number") or "").strip().upper()
            sec_type = str(meta.get("section_type") or "").strip()
            title = str(meta.get("title") or "").strip()

            law_match = bool(
                title and any(title.endswith(law) for law in refs["laws"])
            )
            art_match = bool(
                sec_num
                and sec_num in refs["articles"]
                and (
                    not refs["section_types"]
                    or not sec_type
                    or sec_type in refs["section_types"]
                )
            )

            if art_match and law_match:
                bonus += article_boost * 2.5
            elif art_match and not refs["laws"]:
                bonus += article_boost
            elif art_match and refs["laws"]:
                bonus += article_boost * 0.25
            elif law_match:
                bonus += article_boost * 0.5

            boosted.append(RetrievalResult(res.passage_id, res.score + bonus))

        boosted.sort(key=lambda r: (-r.score, r.passage_id))
        return boosted[:top_k]

    def search_dense(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Perform semantic Dense search using BGE-M3 and Qdrant."""
        if self.model is None:
            raise ValueError("Embedding model is required for dense search.")

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            RetrievalResult(
                passage_id=hit.payload["passage_id"],
                score=float(hit.score),
            )
            for hit in hits
        ]

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_top_k: int = 20,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Perform hybrid search fusing BM25 and Dense with Reciprocal Rank Fusion.

        Args:
            query: User search query.
            top_k: Final number of ranked passages to return.
            candidate_top_k: Number of candidate passages retrieved from each method.
            weights: Ranking weights for [BM25, Dense]. Defaults to [1.0, 1.0].
            rrf_k: RRF smoothing constant (standard default is 60).
        """
        bm25_candidates = self.search_bm25(query, top_k=candidate_top_k)
        dense_candidates = self.search_dense(query, top_k=candidate_top_k)

        return self.fuse(
            bm25_candidates,
            dense_candidates,
            top_k=top_k,
            weights=weights,
            rrf_k=rrf_k,
        )

    @staticmethod
    def fuse(
        bm25_results: Sequence[RetrievalResult],
        dense_results: Sequence[RetrievalResult],
        *,
        top_k: int = 5,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Fuse already-retrieved BM25 and Dense rankings with RRF.

        Useful when the individual rankings are also needed (e.g. for
        exporting all three methods) so each search runs only once.
        """
        return reciprocal_rank_fusion(
            [list(bm25_results), list(dense_results)],
            k=rrf_k,
            weights=weights,
            top_k=top_k,
        )

    def compare(
        self,
        query: str,
        top_k: int = 5,
        candidate_top_k: int = 20,
    ) -> dict[str, list[RetrievalResult]]:
        """Retrieve top results across BM25, Dense, and Hybrid for comparison analysis."""
        bm25_results = self.search_bm25(query, top_k=candidate_top_k)
        dense_results = self.search_dense(query, top_k=candidate_top_k)
        hybrid_results = self.fuse(bm25_results, dense_results, top_k=top_k)

        return {
            "bm25": bm25_results[:top_k],
            "dense": dense_results[:top_k],
            "hybrid": hybrid_results,
        }

    def get_passage_data(self, passage_id: str) -> dict[str, Any] | None:
        """Return the metadata and text for a given passage_id."""
        return self.passage_data.get(passage_id)

