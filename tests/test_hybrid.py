import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrieval import HybridRetriever, RetrievalResult


class FakeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class FakeModel:
    """Stands in for SentenceTransformer; records the queries it encodes."""

    def __init__(self):
        self.queries = []

    def encode(self, query, **kwargs):
        self.queries.append(query)
        return FakeVector([0.1, 0.2, 0.3])


class FakeQdrant:
    """Stands in for QdrantClient.query_points with a fixed dense ranking."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def query_points(self, collection_name, query, limit):
        self.calls.append((collection_name, limit))
        points = [
            SimpleNamespace(payload={"passage_id": pid}, score=score)
            for pid, score in self.ranking[:limit]
        ]
        return SimpleNamespace(points=points)


PASSAGES = {
    "p1": {"passage_id": "p1", "text": "İş sağlığı ve güvenliği kurulu elli çalışan", "title": "6331"},
    "p2": {"passage_id": "p2", "text": "Çalışanların eğitim yükümlülüğü", "title": "Eğitim"},
    "p3": {"passage_id": "p3", "text": "Risk değerlendirmesi yapılır", "title": "Risk"},
}


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.client = FakeQdrant([("p2", 0.9), ("p1", 0.8), ("p3", 0.1)])
        self.retriever = HybridRetriever(
            PASSAGES, self.client, "isg_test", embedding_model=self.model
        )

    def test_dense_search_uses_collection_and_returns_results(self):
        results = self.retriever.search_dense("kurul", top_k=2)
        self.assertEqual([r.passage_id for r in results], ["p2", "p1"])
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(self.client.calls, [("isg_test", 2)])

    def test_hybrid_fuses_bm25_and_dense(self):
        # BM25 ranks p1 first (exact terms); dense ranks p1 second.
        # p1 appears high in both lists, so it should win after RRF.
        results = self.retriever.search_hybrid("iş sağlığı kurulu", top_k=3)
        self.assertEqual(results[0].passage_id, "p1")
        self.assertLessEqual(len(results), 3)

    def test_compare_returns_all_three_methods(self):
        out = self.retriever.compare("kurulu", top_k=2)
        self.assertEqual(set(out), {"bm25", "dense", "hybrid"})
        self.assertTrue(all(len(v) <= 2 for v in out.values()))

    def test_plain_text_passages_are_accepted(self):
        retriever = HybridRetriever(
            {"a": "iş güvenliği"}, self.client, "isg_test", embedding_model=self.model
        )
        self.assertEqual(retriever.get_passage_data("a")["text"], "iş güvenliği")

    def test_from_experiment_loads_chunks_and_names_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunk_dir = Path(tmp) / "data" / "chunks" / "size_1000_overlap_0"
            chunk_dir.mkdir(parents=True)
            with open(chunk_dir / "doc.json", "w", encoding="utf-8") as f:
                json.dump(list(PASSAGES.values()), f, ensure_ascii=False)

            retriever = HybridRetriever.from_experiment(
                "size_1000_overlap_0", tmp, qdrant_client=self.client, model=self.model
            )
        self.assertEqual(retriever.collection_name, "isg_size_1000_overlap_0")
        self.assertEqual(retriever.bm25_index.passage_count, 3)

    def test_missing_experiment_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                HybridRetriever.from_experiment(
                    "yok", tmp, qdrant_client=self.client, model=self.model
                )


if __name__ == "__main__":
    unittest.main()
