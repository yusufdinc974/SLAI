# Retrieval Çıktıları ve Ekip Entegrasyon Rehberi (Zeynep & Yusuf İçin)

Bu doküman, **Bölüm 3 (BM25, Dense Retrieval Entegrasyonu ve Hybrid RRF)** kapsamında üretilen çıktıların **Kişi 3 (Zeynep — LLM & UI)** ve **Kişi 4 (Yusuf — Değerlendirme & Metrikler)** tarafından nasıl kullanılacağını ve yeni soru seti eklendiğinde nelerin yeniden çalıştırılacağını açıklar.

---

## 1. Zeynep (Kişi 3 — LLM Cevap Üretimi & Arayüz) İçin Rehber

### A. Hazır Pasaj Metinlerini Almak (`llm_contexts.json`)
LLM prompt şablonunu test ederken doğrudan hazır çekilmiş ilk 5 pasaj metnini ve kaynak bilgilerini kullanabilirsin:
- **Önerilen (Varsayılan) Dosya:** `results/retrieval/size_1000_overlap_100/llm_contexts.json`
- **Diğer Deney Dosyaları:**
  - `results/retrieval/size_500_overlap_0/llm_contexts.json`
  - `results/retrieval/size_1000_overlap_0/llm_contexts.json`
  - `results/retrieval/size_1000_overlap_200/llm_contexts.json`
  - `results/retrieval/size_1500_overlap_0/llm_contexts.json`

**Dosya Yapısı (`llm_contexts.json`):**
```json
[
  {
    "question_id": "q1",
    "question": "İş sağlığı ve güvenliği kurulu hangi işyerlerinde kurulur ve işverenin yükümlülüğü nedir?",
    "methods": {
      "hybrid": [
        {
          "rank": 1,
          "passage_id": "36224b53687618e0f302",
          "score": 0.03252,
          "title": "1.5.6331",
          "section_type": "MADDE",
          "section_number": "22",
          "text": "MADDE 22 – (1) Elli ve daha fazla çalışanın bulunduğu ve altı aydan fazla süren..."
        }
      ],
      "bm25": [...],
      "dense": [...]
    }
  }
]
```

### B. Canlı Sistemde (UI / LLM Modülünde) Python İle Çağırmak
Arayüzden kullanıcı canlı bir soru sorduğunda ilk 5 pasajı ve metinlerini almak için:
```python
from retrieval import HybridRetriever

# Varsayılan deney ile retriever başlatılır
retriever = HybridRetriever.from_experiment("size_1000_overlap_100")

# Hybrid (RRF), sadece BM25 veya sadece Dense arama yapılabilir:
results = retriever.search_hybrid(question, top_k=5)
# bm25_results = retriever.search_bm25(question, top_k=5)
# dense_results = retriever.search_dense(question, top_k=5)

# LLM promptuna verilecek pasaj metinleri ve madde bilgileri:
contexts = []
for rank, res in enumerate(results, start=1):
    passage = retriever.bm25_index.passage_map[res.passage_id]
    contexts.append({
        "rank": rank,
        "passage_id": res.passage_id,
        "score": res.score,
        "title": passage.get("title"),
        "section_type": passage.get("section_type"),
        "section_number": passage.get("section_number"),
        "text": passage.get("text"),
    })
```

---

## 2. Yusuf (Kişi 4 — Soru Seti & Değerlendirme Metrikleri) İçin Rehber

### A. Metrik Hesaplama İçin Alacağın Sıralama Dosyaları (`runs.jsonl` & `runs.csv`)
`Precision@5`, `Recall@5`, `MRR@5` ve `nDCG@5` hesaplamaları için her 5 deneyin altında hem JSONL hem CSV formatında sıralama sonuçları hazırdır:

- `results/retrieval/size_500_overlap_0/runs.jsonl` ve `runs.csv`
- `results/retrieval/size_1000_overlap_0/runs.jsonl` ve `runs.csv`
- `results/retrieval/size_1000_overlap_100/runs.jsonl` ve `runs.csv` *(Varsayılan)*
- `results/retrieval/size_1000_overlap_200/runs.jsonl` ve `runs.csv`
- `results/retrieval/size_1500_overlap_0/runs.jsonl` ve `runs.csv`

**`runs.jsonl` Satır Formatı:**
```json
{
  "experiment": "size_1000_overlap_100",
  "question_id": "q1",
  "question": "İş sağlığı ve güvenliği kurulu hangi işyerlerinde kurulur ve işverenin yükümlülüğü nedir?",
  "bm25": [{"passage_id": "36224b53687618e0f302", "score": 21.8492}, ...],
  "dense": [{"passage_id": "bfeb1f54896a3cf0fd0b", "score": 0.8926}, ...],
  "hybrid": [{"passage_id": "36224b53687618e0f302", "score": 0.0325}, ...]
}
```

**`runs.csv` Sütun Formatı:**
`experiment,question_id,method,rank,passage_id,score`

> **Önemli Not (`passage_id` uyumu):** Tüm `passage_id` değerleri Dilara'nın `embeddings.zip` içindeki `passage_id` değerleriyle %100 aynıdır. Ground-truth (`relevant_passage_ids`) etiketlemesini yaparken bu `passage_id` değerlerini kullanabilirsin.

### B. Rapor İçin Hazır Parametre ve Deney Karşılaştırma Tabloları
- `results/retrieval/bm25_parameter_experiments.csv` (ve `.json`)
- `results/retrieval/rrf_parameter_experiments.csv` (ve `.json`)
- `results/retrieval/chunking_5_experiments_comparison.csv` (ve `.json`)

---

## 3. Yeni Soru Seti (`questions.json`) Eklendiğinde Ne Yeniden Çalıştırılacak?

40–50 soruluk resmi soru veri seti (`data/questions/questions.json`) hazırlandığında:

### A. Değişmeyecek Olanlar (Yeniden Denenmesine Gerek Yok)
- `retrieval/` altındaki Python modüllerinde (`bm25.py`, `rrf.py`, `hybrid.py`, `export_results.py`) hiçbir kod değişikliği yapılmaz.
- `bm25_parameter_experiments.csv`, `rrf_parameter_experiments.csv` ve `chunking_5_experiments_comparison.csv` dosyalarındaki parametre duyarlılık analizleri sabit korpus davranışı olduğu için değiştirmeye gerek yoktur.

### B. Sadece Yeni Sorular İçin `runs.jsonl` ve `runs.csv` Üretmek
1. Eğer yerel bilgisayarda `data/qdrant/` ve `data/chunks/` henüz çıkarılmadıysa bir kez `embeddings.zip`veya 'data.zip' içerisinden çıkarılır ve `python -m vectorstore.main` çalıştırılır.
2. `data/questions/questions.json` dosyası şu formatta kaydedilir:
   ```json
   [
     {"question_id": "q1", "question": "Soru metni 1..."},
     {"question_id": "q2", "question": "Soru metni 2..."}
   ]
   ```
3. Ardından şu komut çalıştırılarak 40–50 sorunun tamamı için `runs.jsonl` ve `runs.csv` otomatik güncellenir:
   ```bash
   python -m retrieval.export_results --questions data/questions/questions.json --experiment size_1000_overlap_100 --output results/retrieval/size_1000_overlap_100/runs.jsonl --csv results/retrieval/size_1000_overlap_100/runs.csv
   ```
