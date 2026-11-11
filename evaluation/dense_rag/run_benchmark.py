"""Run ISG-RAG benchmark through the project's actual HybridRetriever.

Runs BM25, Dense and Hybrid retrieval for every benchmark question, saves
ranked passage metadata, and optionally calls a project generation adapter.

Example retrieval-only:
  python -m evaluation.run_benchmark --split dev --experiment size_1000_overlap_0

With a generation adapter:
  python -m evaluation.run_benchmark --split dev --experiment size_1000_overlap_0 ^
    --method hybrid --generator-module your_module --generator-function generate_answer

The generator must accept (question, contexts) and return either a string or:
  {"answer": "...", "citations": [{"document_id":"...", "article":"..."}]}
"""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
from typing import Any

from retrieval.hybrid import HybridRetriever

def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data=json.load(f)
    if not isinstance(data,list):
        raise ValueError("Benchmark JSON must contain a list.")
    ids=[x.get("question_id") for x in data]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate question_id in benchmark.")
    return data

def serialize_results(results, retriever):
    out=[]
    for rank,res in enumerate(results,1):
        p=retriever.get_passage_data(res.passage_id) or {}
        out.append({
            "rank":rank,
            "passage_id":res.passage_id,
            "score":float(res.score),
            "document_id":p.get("document_id"),
            "title":p.get("title"),
            "section_type":p.get("section_type"),
            "section_number":str(p.get("section_number","")),
            "section_marker":p.get("section_marker"),
            "paragraph_numbers":p.get("paragraph_numbers",[]),
            "text":p.get("text","")
        })
    return out

def load_generator(module_name, function_name):
    module=importlib.import_module(module_name)
    fn=getattr(module,function_name)
    if not callable(fn):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return fn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark",default="evaluation/benchmark_150_frozen.json")
    ap.add_argument("--split",choices=["dev","frozen_test","all"],default="dev")
    ap.add_argument("--experiment",default="size_1000_overlap_0")
    ap.add_argument("--top-k",type=int,default=10)
    ap.add_argument("--candidate-top-k",type=int,default=20)
    ap.add_argument("--method",choices=["bm25","dense","hybrid"],default="hybrid")
    ap.add_argument("--generator-module")
    ap.add_argument("--generator-function",default="generate_answer")
    ap.add_argument("--output",default=None)
    args=ap.parse_args()

    base=Path(__file__).resolve().parents[1]
    bench=Path(args.benchmark)
    if not bench.is_absolute(): bench=base/bench
    questions=load_records(bench)
    if args.split!="all":
        questions=[q for q in questions if q.get("split")==args.split]
    if not questions:
        raise RuntimeError(f"No questions for split={args.split}")

    # Actual project pipeline: loads chunk metadata, local Qdrant and BGE-M3.
    retriever=HybridRetriever.from_experiment(
        experiment_name=args.experiment,
        base_dir=base
    )
    generator=None
    if args.generator_module:
        generator=load_generator(args.generator_module,args.generator_function)

    output=Path(args.output) if args.output else (
        base/"results"/"benchmark"/args.experiment/
        f"{args.split}_{args.method}_results.jsonl"
    )
    if not output.is_absolute(): output=base/output
    output.parent.mkdir(parents=True,exist_ok=True)

    try:
        with output.open("w",encoding="utf-8") as f:
            for i,q in enumerate(questions,1):
                question=q["question"]
                # Retrieve each branch once; Hybrid uses BM25 + Dense RRF.
                bm25=retriever.search_bm25(question,top_k=args.candidate_top_k)
                dense=retriever.search_dense(question,top_k=args.candidate_top_k)
                hybrid=retriever.fuse(
                    bm25,dense,top_k=args.top_k
                )
                rankings={
                    "bm25":serialize_results(bm25[:args.top_k],retriever),
                    "dense":serialize_results(dense[:args.top_k],retriever),
                    "hybrid":serialize_results(hybrid,retriever)
                }
                contexts=rankings[args.method]
                answer=""
                citations=[]
                if generator is not None:
                    generated=generator(question,contexts)
                    if isinstance(generated,str):
                        answer=generated
                    elif isinstance(generated,dict):
                        answer=str(generated.get("answer",generated.get("generated_answer","")) or "")
                        citations=generated.get("citations",generated.get("predicted_citations",[])) or []
                    else:
                        raise TypeError("Generator must return str or dict.")

                row={
                    "question_id":q["question_id"],
                    "split":q.get("split"),
                    "method":args.method,
                    "question":question,
                    "retrieved_passages":contexts,
                    "all_retrieval_methods":rankings,
                    "generated_answer":answer,
                    "predicted_citations":citations,
                    "gold_document_id":q.get("gold_document_id"),
                    "gold_article":q.get("gold_article"),
                    "gold_citation":q.get("gold_citation")
                }
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
                f.flush()
                print(f"[{i}/{len(questions)}] {q['question_id']} retrieved={len(contexts)}")
    finally:
        retriever.client.close()
    print(f"Saved: {output}")
    if generator is None:
        print("Retrieval-only run. No LLM answer was generated because no generator adapter was supplied.")

if __name__=="__main__":
    main()
