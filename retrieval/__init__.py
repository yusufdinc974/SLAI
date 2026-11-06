"""Retrieval prototypes and hybrid pipeline for the SLAI project."""

from .bm25 import BM25Index, RetrievalResult, extract_legal_references, tokenize
from .rrf import reciprocal_rank_fusion
from .hybrid import HybridRetriever

__all__ = [
    "BM25Index",
    "RetrievalResult",
    "extract_legal_references",
    "reciprocal_rank_fusion",
    "tokenize",
    "HybridRetriever",
]
