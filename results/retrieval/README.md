# `results/retrieval/` — Çıktı Dosyaları Rehberi

Bu klasör, **Bölüm 3 (BM25, Dense Retrieval Entegrasyonu ve Hybrid RRF)** kapsamında üretilen hazır çıktı dosyalarını ve deney karşılaştırma tablolarını içerir. Detaylı entegrasyon rehberi için [`docs/retrieval-entegrasyon-README.md`](../../docs/retrieval-entegrasyon-README.md) ve [`docs/evrim-calisma.md`](../../docs/evrim-calisma.md) dosyalarına bakabilirsiniz.

## Klasör İçeriği

| Dosya / Klasör | Hedef Kişi / Modül | Açıklama |
|---|---|---|
| `<deney_adi>/runs.jsonl` | **Kişi 4 (Yusuf — Evaluation)** | Her soru için `bm25`, `dense`, `hybrid` ilk 5 `passage_id` ve `score` çıktısı (JSONL) |
| `<deney_adi>/runs.csv` | **Kişi 4 (Yusuf — Evaluation)** | Metrik hesaplama (`Precision@5`, `Recall@5`, `MRR@5`, `nDCG@5`) için tablo formatı (CSV) |
| `<deney_adi>/llm_contexts.json` | **Kişi 3 (Zeynep — LLM & UI)** | Her soru ve yöntem için ilk 5 pasajın tam metni (`text`), kanun adı (`title`) ve madde numarası (`section_number`) |
| `bm25_parameter_experiments.csv` / `.json` | **Ortak Rapor (Bölüm 3.1)** | BM25 `k1`, `b` ve `article_boost` parametre duyarlılık deney sonuçları |
| `rrf_parameter_experiments.csv` / `.json` | **Ortak Rapor (Bölüm 3.3)** | RRF `rrf_k`, ağırlıklar ve BM25 vs. Dense vs. Hybrid karşılaştırma sonuçları |
| `chunking_5_experiments_comparison.csv` / `.json` | **Ortak Rapor (Bölüm 3.2 & 3.3)** | 5 farklı chunking deneyinin (`500/0`, `1000/0`, `1000/100`, `1000/200`, `1500/0`) karşılaştırması |

## Yeni Soru Seti (`questions.json`) Eklendiğinde Güncelleme Komutu

40–50 soruluk resmi soru veri seti (`data/questions/questions.json`) eklendiğinde sıralama dosyalarını güncellemek için:

```bash
python -m retrieval.export_results --questions data/questions/questions.json --experiment size_1000_overlap_100 --output results/retrieval/size_1000_overlap_100/runs.jsonl --csv results/retrieval/size_1000_overlap_100/runs.csv
```
