# Evaluation entegrasyonu

Bu klasör, benchmark'ı doğrudan repodaki `retrieval.HybridRetriever` sınıfına bağlar.

## 1. Gereken yerel veri

GitHub ZIP'inde büyük/yerel veri klasörleri bulunmuyor. Runner'ın çalışması için proje kökünde şunlar mevcut olmalı:

- `data/chunks/size_1000_overlap_0/*.json`
- `data/qdrant/` (ilgili `isg_size_1000_overlap_0` collection'ı)
- BGE-M3 modelini indirebilmek için internet erişimi veya önceden cache'lenmiş model

Bunlar senin bilgisayarındaki mevcut proje klasöründe zaten varsa, ZIP'ten çıkardığın yeni klasörde yeniden üretme. Entegrasyon dosyalarını mevcut `SLAI-main` klasörüne kopyalayabilir veya mevcut verileri yeni klasöre aktarabilirsin.

## 2. Retrieval-only çalıştırma

Proje kök klasöründe:

```powershell
python -m evaluation.run_benchmark --split dev --experiment size_1000_overlap_0 --method hybrid
```

Bu komut her soru için BM25, Dense ve Hybrid sonuçlarını hesaplar; seçilen yöntemin ilk 10 chunk'ını metadata ve metinleriyle JSONL'e yazar. Varsayılan çıktı:

`results/benchmark/size_1000_overlap_0/dev_hybrid_results.jsonl`

Dense veya BM25 için `--method dense` / `--method bm25` kullan.

## 3. Qwen generation bağlantısı

Yüklenen repoda Qwen inference fonksiyonu bulunmuyor. `evaluation/generator_adapter_template.py` dosyasını kopyalayıp gerçek model kodunuza göre `generate_answer(question, contexts)` fonksiyonunu doldurun. Ardından:

```powershell
python -m evaluation.run_benchmark --split dev --experiment size_1000_overlap_0 --method hybrid --generator-module evaluation.generator_adapter_template --generator-function generate_answer
```

Adapter doldurulmadan `--generator-module` vermeyin.

## 4. Metrik hesaplama

Runner çıktısı `retrieved_passages` listesinde `document_id` ve `section_number` alanlarını içerir. Değerlendirme:

```powershell
python evaluation/evaluate_results.py --benchmark evaluation/benchmark_150_frozen.json --results results/benchmark/size_1000_overlap_0/dev_hybrid_results.jsonl --split dev --out results/benchmark/dev_report.json
```

`Recall@5` ve `MRR@10` retrieval metrikleridir. `Answer Exact Match` ve `Token F1` yalnızca leksik tanılayıcıdır; semantik doğruluk için insan tarafından `human_answer_correct` alanı eklenmelidir.

## 5. Test protokolü

Önce `dev` (30 soru) üzerinde retrieval/prompt ayarlarını yap. Ayarları sabitledikten sonra `frozen_test` (120 soru) üzerinde değerlendir. Test sonuçlarına bakarak prompt veya parametre değiştirme.
