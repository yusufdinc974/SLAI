import unittest

from retrieval import BM25Index, RetrievalResult, reciprocal_rank_fusion, tokenize


class BM25IndexTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index({
            "p1": "İş sağlığı ve güvenliği uzmanı",
            "p2": "Çalışanların eğitim ve denetim yükümlülüğü",
            "p3": "İşverenin iş sağlığı yükümlülüğü",
        })

    def test_turkish_tokenization_is_case_insensitive(self):
        self.assertEqual(tokenize("İŞ Sağlığı"), ["iş", "sağlığı"])

    def test_search_returns_ranked_standard_results(self):
        results = self.index.search("iş sağlığı", top_k=2)
        self.assertEqual([result.passage_id for result in results], ["p3", "p1"])
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertGreater(results[0].score, results[1].score)

    def test_empty_query_returns_no_results(self):
        self.assertEqual(self.index.search(""), [])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            BM25Index({}, b=2)
        with self.assertRaises(ValueError):
            self.index.search("iş", top_k=0)


class RRFTests(unittest.TestCase):
    def test_fuses_dense_and_bm25_rankings(self):
        dense = [
            RetrievalResult("p1", 0.9),
            RetrievalResult("p2", 0.8),
        ]
        bm25 = ["p2", "p3"]
        results = reciprocal_rank_fusion([dense, bm25], k=1)
        self.assertEqual([result.passage_id for result in results], ["p2", "p1", "p3"])

    def test_weights_and_top_k_are_applied(self):
        results = reciprocal_rank_fusion(
            [["p1", "p2"], ["p2", "p3"]],
            k=1,
            weights=[1, 3],
            top_k=2,
        )
        self.assertEqual([result.passage_id for result in results], ["p2", "p3"])

    def test_weight_count_must_match_rankings(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["p1"]], weights=[1, 2])


if __name__ == "__main__":
    unittest.main()
