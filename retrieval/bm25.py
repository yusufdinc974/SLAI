"""A small, dependency-free BM25 implementation for Turkish OHS passages."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

_ARTICLE_AFTER_PATTERN = re.compile(
    r"(?:(geçici|ek)\s+)?madde\s+(\d+(?:\s*/\s*[a-zA-Z])?)",
    re.IGNORECASE | re.UNICODE,
)
_ARTICLE_BEFORE_PATTERN = re.compile(
    r"(\d+)\s*(?:\.|'|’)?\s*(?:nc[ıiuü]|inc[ıiuü]|unc[uü]|ünc[uü])?\s*madde",
    re.IGNORECASE | re.UNICODE,
)
_LAW_NUMBER_PATTERN = re.compile(
    r"\b(\d{4})\s*sayılı",
    re.IGNORECASE | re.UNICODE,
)


def extract_legal_references(query: str) -> dict[str, set[str]]:
    """Extract explicit article numbers and 4-digit law numbers from a Turkish query.

    Recognizes forms such as ``"Madde 22"``, ``"22 nci madde"``, ``"17. madde"``,
    ``"Geçici Madde 9"``, and ``"6331 sayılı Kanun"``.
    """
    normalized = query.replace("İ", "i").casefold()
    articles: set[str] = set()
    section_types: set[str] = set()

    for prefix, number in _ARTICLE_AFTER_PATTERN.findall(normalized):
        clean_num = re.sub(r"\s+", "", number).upper()
        articles.add(clean_num)
        if prefix == "geçici":
            section_types.add("provisional_article")
        elif prefix == "ek":
            section_types.add("additional_article")
        else:
            section_types.add("article")

    for number in _ARTICLE_BEFORE_PATTERN.findall(normalized):
        articles.add(number.strip())
        if not section_types:
            section_types.add("article")

    laws = {match.strip() for match in _LAW_NUMBER_PATTERN.findall(normalized)}
    return {
        "articles": articles,
        "laws": laws,
        "section_types": section_types,
    }


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase Unicode word tokens.

    Keeping Unicode letters intact is important for Turkish terms such as
    ``iş`` and ``sağlığı``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # ``İ`` casefolding produces ``i`` plus a combining dot; normalize it
    # explicitly so Turkish words remain single tokens.
    return _TOKEN_PATTERN.findall(text.replace("İ", "i").casefold())


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked passage returned by a retrieval method."""

    passage_id: str
    score: float


class BM25Index:
    """Okapi BM25 index over passage IDs and their text."""

    def __init__(
        self,
        passages: Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        items = passages.items() if isinstance(passages, Mapping) else passages
        self._passages = dict(items)
        if any(not isinstance(pid, str) or not isinstance(text, str)
               for pid, text in self._passages.items()):
            raise TypeError("passages must contain string IDs and text")

        self.k1 = k1
        self.b = b
        self._tokens = {
            passage_id: tokenize(text)
            for passage_id, text in self._passages.items()
        }
        self._term_frequencies = {
            passage_id: Counter(tokens)
            for passage_id, tokens in self._tokens.items()
        }
        self._document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies.values():
            self._document_frequency.update(frequencies.keys())
        self._average_document_length = (
            sum(map(len, self._tokens.values())) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    @property
    def passage_count(self) -> int:
        return len(self._passages)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return the highest-scoring passages for ``query``."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = tokenize(query)
        if not query_terms or not self._passages:
            return []

        query_term_counts = Counter(query_terms)
        document_count = len(self._passages)
        scores: list[RetrievalResult] = []
        for passage_id, frequencies in self._term_frequencies.items():
            document_length = len(self._tokens[passage_id])
            score = 0.0
            for term, query_frequency in query_term_counts.items():
                term_frequency = frequencies.get(term, 0)
                document_frequency = self._document_frequency.get(term, 0)
                if not term_frequency or not document_frequency:
                    continue
                idf = math.log(
                    1 + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = term_frequency + self.k1 * (
                    1 - self.b
                    + self.b * document_length / self._average_document_length
                )
                score += (
                    idf
                    * (term_frequency * (self.k1 + 1) / denominator)
                    * query_frequency
                )
            scores.append(RetrievalResult(passage_id, score))

        return sorted(
            scores,
            key=lambda result: (-result.score, result.passage_id),
        )[:top_k]
