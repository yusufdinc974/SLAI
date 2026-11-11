import csv
import json
import tempfile
import unittest
from pathlib import Path

from retrieval import HybridRetriever
from retrieval.export_results import load_questions, run_questions, save_runs
from test_hybrid import PASSAGES, FakeModel, FakeQdrant


class LoadQuestionsTests(unittest.TestCase):
    def _write(self, tmp, name, content):
        path = Path(tmp) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_json_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "q.json", json.dumps([
                {"question_id": "q1", "question": "İSG kurulu", "gold": ["p1"]},
            ], ensure_ascii=False))
            self.assertEqual(load_questions(path), [{"question_id": "q1", "question": "İSG kurulu"}])

    def test_loads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "q.jsonl",
                               '{"question_id": 1, "question": "a"}\n\n{"question_id": 2, "question": "b"}\n')
            self.assertEqual([q["question_id"] for q in load_questions(path)], ["1", "2"])

    def test_loads_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "q.csv",
                               "question_id,category,question,gold_refs\nq1,Mevzuat,İSG kurulu,fad54922e43a:article:22\n")
            self.assertEqual(load_questions(path), [{"question_id": "q1", "question": "İSG kurulu"}])

    def test_rejects_duplicates_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            dup = self._write(tmp, "d.jsonl",
                              '{"question_id": "q1", "question": "a"}\n{"question_id": "q1", "question": "b"}\n')
            with self.assertRaises(ValueError):
                load_questions(dup)
            bad = self._write(tmp, "b.jsonl", '{"question": "a"}\n')
            with self.assertRaises(ValueError):
                load_questions(bad)


class RunAndSaveTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(
            PASSAGES,
            FakeQdrant([("p2", 0.9), ("p1", 0.8), ("p3", 0.1)]),
            "isg_test",
            embedding_model=FakeModel(),
        )
        self.questions = [{"question_id": "q1", "question": "iş sağlığı kurulu"}]

    def test_runs_contain_all_methods_with_ranks(self):
        runs = run_questions(self.retriever, self.questions, experiment="exp", top_k=2)
        self.assertEqual([r["method"] for r in runs], ["bm25", "dense", "hybrid"])
        for run in runs:
            self.assertEqual(run["question_id"], "q1")
            self.assertLessEqual(len(run["results"]), 2)
            self.assertEqual([i["rank"] for i in run["results"]],
                             list(range(1, len(run["results"]) + 1)))
        self.assertEqual(runs[2]["results"][0]["passage_id"], "p1")

    def test_save_writes_jsonl_and_csv(self):
        runs = run_questions(self.retriever, self.questions, experiment="exp", top_k=2)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path, csv_path = save_runs(runs, tmp)
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["method"], "bm25")
            with csv_path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), sum(len(r["results"]) for r in runs))
            self.assertEqual(set(rows[0]), {"question_id", "method", "experiment",
                                            "rank", "passage_id", "score"})


if __name__ == "__main__":
    unittest.main()
