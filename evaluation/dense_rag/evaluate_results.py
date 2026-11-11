#!/usr/bin/env python3
"""Evaluate retrieval and answer outputs for the ISG-RAG benchmark.

Input JSONL schema, one row per question:
{
  "question_id": "ISG-QA-001",
  "retrieved_passages": [
    {
      "passage_id": "...",
      "document_id": "...",
      "section_number": "7",
      "paragraph_numbers": ["1"],
      "text": "..."
    }
  ],
  "generated_answer": "...",
  "predicted_citations": [
    {"document_id": "...", "article": "7", "paragraph": "1"}
  ],
  "human_answer_correct": 1
}

`human_answer_correct` is optional. If supplied (0/1), it is used for answer
correctness. Otherwise lexical Exact Match and token F1 are reported as
diagnostic metrics; they are NOT a substitute for semantic correctness.
"""
import argparse, json, re, unicodedata
from pathlib import Path
from collections import Counter

def load_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: {e}")
    return rows

def norm(s):
    s = unicodedata.normalize("NFKC", str(s or "")).casefold()
    s = s.replace("ı","i")
    s = re.sub(r"[^a-z0-9çğıöşü]+"," ",s)
    return " ".join(s.split())

def token_f1(pred, gold):
    p=norm(pred).split(); g=norm(gold).split()
    if not p or not g: return 1.0 if p==g else 0.0
    common=Counter(p) & Counter(g)
    n=sum(common.values())
    if n==0: return 0.0
    precision=n/len(p); recall=n/len(g)
    return 2*precision*recall/(precision+recall)

def same_article(hit, gold):
    return (
        str(hit.get("document_id","")) == str(gold.get("gold_document_id",""))
        and str(hit.get("section_number","")) == str(gold.get("gold_article",""))
    )

def retrieval_metrics(gold, result):
    hits=result.get("retrieved_passages") or []
    recall5=1.0 if any(same_article(x,gold) for x in hits[:5]) else 0.0
    rr=0.0
    for rank,x in enumerate(hits[:10],1):
        if same_article(x,gold):
            rr=1.0/rank
            break
    return recall5,rr

def citation_matches(cite,gold):
    # Support both structured citations and exact gold_citation strings.
    if isinstance(cite,str):
        return norm(cite)==norm(gold.get("gold_citation",""))
    doc=cite.get("document_id")
    art=cite.get("article",cite.get("section_number"))
    para=cite.get("paragraph")
    ok=(str(doc or "")==str(gold.get("gold_document_id",""))
        and str(art or "")==str(gold.get("gold_article","")))
    if para is not None and gold.get("gold_paragraph") is not None:
        ok = ok and str(para)==str(gold.get("gold_paragraph"))
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark",required=True)
    ap.add_argument("--results",required=True)
    ap.add_argument("--split",choices=["dev","frozen_test","all"],default="all")
    ap.add_argument("--out",default="evaluation_report.json")
    args=ap.parse_args()
    gold_rows=load_jsonl(args.benchmark) if args.benchmark.endswith(".jsonl") else json.load(open(args.benchmark,encoding="utf-8"))
    result_rows=load_jsonl(args.results)
    gold_by_id={x["question_id"]:x for x in gold_rows}
    result_by_id={x["question_id"]:x for x in result_rows}
    selected=[x for x in gold_rows if args.split=="all" or x.get("split")==args.split]
    missing=[x["question_id"] for x in selected if x["question_id"] not in result_by_id]
    extra=[x for x in result_by_id if x not in gold_by_id]
    r5=[]; mrr=[]; em=[]; f1=[]; cite=[]; human=[]
    details=[]
    for g in selected:
        r=result_by_id.get(g["question_id"],{})
        a,b=retrieval_metrics(g,r); r5.append(a); mrr.append(b)
        pred=r.get("generated_answer","")
        gold=g.get("reference_answer","")
        em.append(float(norm(pred)==norm(gold)))
        f1.append(token_f1(pred,gold))
        citations=r.get("predicted_citations") or []
        cite.append(float(any(citation_matches(c,g) for c in citations)))
        if r.get("human_answer_correct") in (0,1,False,True):
            human.append(float(r["human_answer_correct"]))
        details.append({
            "question_id":g["question_id"],"split":g.get("split"),
            "recall_at_5":a,"reciprocal_rank_at_10":b,
            "exact_match":em[-1],"token_f1":f1[-1],"citation_accuracy":cite[-1],
            "human_answer_correct":r.get("human_answer_correct")
        })
    mean=lambda xs: round(sum(xs)/len(xs),4) if xs else None
    report={
      "split":args.split,"evaluated_questions":len(selected),
      "missing_result_ids":missing,"extra_result_ids":extra,
      "metrics":{
        "Recall@5":mean(r5),"MRR@10":mean(mrr),
        "Answer Exact Match (diagnostic)":mean(em),
        "Answer Token F1 (diagnostic)":mean(f1),
        "Human Answer Correctness":mean(human),
        "Citation Accuracy":mean(cite)
      },
      "notes":[
        "Retrieval relevance is judged by exact gold document_id + article number.",
        "Recall@5 and MRR@10 require retrieved_passages in rank order.",
        "Exact Match and Token F1 are lexical diagnostics, not semantic correctness.",
        "Use human_answer_correct=0/1 for reviewed semantic answer correctness.",
        "Citation accuracy checks whether at least one predicted citation matches the gold document and article; paragraph is checked when supplied."
      ],
      "per_question":details
    }
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"split":args.split,"n":len(selected),"metrics":report["metrics"],"missing":len(missing),"extra":len(extra)},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
