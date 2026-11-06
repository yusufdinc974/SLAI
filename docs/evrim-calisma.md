# BM25, Dense Entegrasyonu ve Hybrid Retrieval (RRF)

## 1. BM25 (Kelime Tabanlı Arama)

İSG mevzuat metinlerinde geçen spesifik hukuki terimlerin, madde numaralarının ve anahtar kelimelerin tam eşleşmeyle yakalanabilmesi amacıyla `retrieval/bm25.py` altında Okapi BM25 algoritması geliştirilmiştir.

BM25 modülünün temel özellikleri:

- **Türkçe Uyumlu Tokenizer:** Büyük `İ` / küçük `i`, `ı`, `ş`, `ğ`, `ü`, `ö`, `ç` gibi Türkçe karakterleri koruyarak küçük harfe dönüştüren (`casefold`) Unicode tabanlı kelime ayrıştırıcı kullanılmıştır.
- **Okapi BM25 Parametreleri:** Varsayılan olarak `k1 = 1.5` (terim frekansı doygunluğu) ve `b = 0.75` (belge uzunluğu normalizasyonu) kullanılmaktadır.
- **Ortak Çıktı Arayüzü:** Tüm arama sonuçları `passage_id` ve `score` alanlarını taşıyan standart `RetrievalResult` veri sınıfı formatında döndürülür.

## 2. Dense Retrieval Entegrasyonu

Dilara tarafından oluşturulan Qdrant vektör veritabanı koleksiyonları (`isg_size_1000_overlap_0` vb.) ve **BAAI/bge-m3** (1024 boyutlu) dense embedding modeli, `retrieval/hybrid.py` içerisindeki `HybridRetriever` sınıfı aracılığıyla ortak retrieval arayüzüne bağlanmıştır.

Dense arama sürecinde:

- Kullanıcı sorusu BGE-M3 modeli ile normalize edilmiş 1024 boyutlu sorgu vektörüne dönüştürülür.
- Qdrant `query_points` API'si kullanılarak Cosine benzerliğine göre en yüksek skorlu `candidate_top_k` (varsayılan 20) pasaj getirilir.
- Sonuçlar BM25 ile aynı `RetrievalResult(passage_id, score)` formatına dönüştürülür.

## 3. Reciprocal Rank Fusion (RRF) ile Hybrid Retrieval

BM25 skorları (sınırsız pozitif reel sayılar) ile Dense Cosine benzerlik skorları (`[-1, 1]` aralığı) farklı ölçeklerde olduğundan, ham skorları doğrudan toplamak yerine sıralama konumlarını (rank) birleştiren **Reciprocal Rank Fusion (RRF)** algoritması (`retrieval/rrf.py`) uygulanmıştır:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

- Sabit yumuşatma parametresi $k = 60$ olarak ayarlanmıştır.
- Her iki yöntemden gelen ilk 20 aday pasaj (`candidate_top_k = 20`) RRF ile birleştirilerek nihai en iyi $N$ (`top_k = 5`) pasaj seçilir.

## 4. Deneysel Karşılaştırma Bulguları (`size_1000_overlap_0`)

20 mevzuat belgesinden elde edilen **2.193 pasaj** üzerinde (`isg_size_1000_overlap_0` koleksiyonu) BM25, Dense ve Hybrid yöntemleri karşılaştırılmıştır (`python -m retrieval.run_hybrid_demo`):

| Soru | BM25 (1. Sıra) | Dense (1. Sıra) | Hybrid RRF (1. ve 2. Sıra) | Gözlem ve Analiz |
|---|---|---|---|---|
| **Soru 1:** İSG kurulu kaç çalışan olan işyerlerinde kurulmalıdır? | `1.5.6331` MADDE 22 *(Doğru)* — ancak 3. sırada mülga 4857 Madde 80 yer aldı | `1.5.6331` MADDE 22 *(Doğru)* | **1.** `1.5.6331` MADDE 22<br>**2.** `7.5.16924` MADDE 10 | Hybrid, mülga maddeyi ilk 3'ten eleyerek yürürlükteki kanun ve yönetmeliği korudu. |
| **Soru 2:** İşveren çalışanların İSG eğitimlerini hangi sıklıkla yenilemek zorundadır? | `7.5.45087` MADDE 5 *(Yönetmelik)* — 6331 Madde 17 ilk 3'e giremedi | `1.5.6331` MADDE 17 *(Kanun)* — Yönetmelik 3. sırada kaldı | **1.** `7.5.45087` MADDE 5<br>**2.** `1.5.6331` MADDE 17 | Hybrid arama, hem ilgili yönetmelik maddesini hem de ana kanun maddesini ilk 2 sıraya taşıdı. |
| **Soru 3:** Çok tehlikeli sınıfta iş güvenliği uzmanının çalışan başına aylık çalışma süresi nedir? | `7.5.18615` MADDE 19 *(Yanlış: Diğer sağlık personeli)* | `7.5.16923` MADDE 12 *(Doğru: İş güvenliği uzmanı)* | **1.** `7.5.16923` MADDE 12 *(Doğru)* | BM25 kelime benzerliği nedeniyle "diğer sağlık personeli" maddesini 1. sıraya koyarken, Hybrid arama doğru yönetmelik maddesini 1. sıraya çıkardı. |

## 5. Değerlendirme ve Çıktı Formatı

Yusuf'un değerlendirme modülü (Recall@k, MRR) ve Zeynep'in LLM cevap üretim modülü ile entegrasyon için `retrieval/export_results.py` aracı geliştirilmiştir. Bu modül soru veri setini okuyarak her üç yöntemin sıralamalarını standart `runs.jsonl` ve `runs.csv` formatlarında dışa aktarır.

## 6. Sorumluluk Kapsamı

Bu çalışma kapsamında Türkçe uyumlu BM25 indeksi, RRF birleştirme algoritması, Qdrant + BGE-M3 Dense aramasının HybridRetriever çatısı altında entegrasyonu, birim testleri (`tests/`) ve yöntemler arası karşılaştırma demosu tamamlanmıştır.

