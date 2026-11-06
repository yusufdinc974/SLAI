# BM25, Dense Entegrasyonu ve Hybrid Retrieval (RRF)

## 1. BM25 (Kelime ve Madde Numarası Tabanlı Arama)

İSG mevzuat metinlerinde geçen spesifik hukuki terimlerin, madde numaralarının ve anahtar kelimelerin tam eşleşmeyle yakalanabilmesi amacıyla `retrieval/bm25.py` altında Okapi BM25 algoritması geliştirilmiştir.

BM25 modülünün temel özellikleri:

- **Türkçe Uyumlu Tokenizer:** Büyük `İ` / küçük `i`, `ı`, `ş`, `ğ`, `ü`, `ö`, `ç` gibi Türkçe karakterleri koruyarak küçük harfe dönüştüren (`casefold`) Unicode tabanlı kelime ayrıştırıcı kullanılmıştır.
- **Madde ve Kanun Numarası Farkındalığı (`extract_legal_references`):** Sorgularda geçen *"Madde 22"*, *"22 nci madde"*, *"17. madde"*, *"Geçici Madde 9"* ve *"6331 sayılı Kanun"* gibi hukuki referanslar otomatik olarak ayrıştırılır ve pasaj metadata'sındaki `section_number`, `section_type` ve `title` alanlarıyla eşleşen pasajlara `article_boost` uygulanır.
- **Okapi BM25 Parametreleri:** Varsayılan olarak `k1 = 1.5` (terim frekansı doygunluğu) ve `b = 0.75` (belge uzunluğu normalizasyonu) kullanılmaktadır.
- **Ortak Çıktı Arayüzü:** Tüm arama sonuçları `passage_id` ve `score` alanlarını taşıyan standart `RetrievalResult` veri sınıfı formatında döndürülür.

## 2. Dense Retrieval Entegrasyonu ve 5 Deney Koleksiyonu

Dilara tarafından oluşturulan 5 farklı chunking deneyine ait Qdrant koleksiyonları ve **BAAI/bge-m3** (1024 boyutlu) dense embedding modeli, `retrieval/hybrid.py` içerisindeki `HybridRetriever` sınıfı aracılığıyla ortak retrieval arayüzüne bağlanmıştır.

Hem BM25 hem de Qdrant Dense Retrieval modüllerinde aynı `passage_id` korpusunu kullanan 5 deney yapılandırması doğrulanmıştır:

| Deney Adı | Qdrant Koleksiyonu | Chunk Boyutu | Overlap | Toplam Pasaj (Point) Sayısı |
|---|---|---:|---:|---:|
| `size_500_overlap_0` | `isg_size_500_overlap_0` | 500 | 0 | 4.089 |
| `size_1000_overlap_0` | `isg_size_1000_overlap_0` | 1000 | 0 | 2.193 |
| `size_1000_overlap_100` | `isg_size_1000_overlap_100` | 1000 | 100 | 2.267 |
| `size_1000_overlap_200` | `isg_size_1000_overlap_200` | 1000 | 200 | 2.266 |
| `size_1500_overlap_0` | `isg_size_1500_overlap_0` | 1500 | 0 | 1.610 |

## 3. Reciprocal Rank Fusion (RRF) ile Hybrid Retrieval

BM25 skorları (sınırsız pozitif reel sayılar) ile Dense Cosine benzerlik skorları (`[-1, 1]` aralığı) farklı ölçeklerde olduğundan, ham skorları doğrudan toplamak yerine sıralama konumlarını (rank) birleştiren **Reciprocal Rank Fusion (RRF)** algoritması (`retrieval/rrf.py`) uygulanmıştır:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

- Sabit yumuşatma parametresi $k = 60$ olarak ayarlanmıştır.
- Her iki yöntemden gelen ilk 20 aday pasaj (`candidate_top_k = 20`) RRF ile birleştirilerek nihai en iyi $N$ (`top_k = 5`) pasaj seçilir.

## 4. Yöntem Karşılaştırması: BM25 vs. Dense vs. Hybrid RRF

Temel deney koleksiyonu (`isg_size_1000_overlap_0`, 2.193 pasaj) üzerinde BM25-only, Dense-only ve Hybrid (RRF) yöntemleri karşılaştırılmıştır (`python -m retrieval.run_hybrid_demo`):

| Soru | BM25-Only (1. Sıra) | Dense-Only (1. Sıra) | Hybrid RRF (1. ve 2. Sıra) | Analiz |
|---|---|---|---|---|
| **Soru 1:** İSG kurulu kaç çalışan olan işyerlerinde kurulmalıdır? | `1.5.6331` MADDE 22 *(Doğru)* — 3. sırada mülga 4857 Madde 80 yer aldı | `1.5.6331` MADDE 22 *(Doğru)* | **1.** `1.5.6331` MADDE 22<br>**2.** `7.5.16924` MADDE 10 | Hybrid, mülga maddeyi ilk 3'ten eleyerek yürürlükteki kanun ve yönetmeliği korudu. |
| **Soru 2:** İşveren çalışanların İSG eğitimlerini hangi sıklıkla yenilemek zorundadır? | `7.5.45087` MADDE 5 *(Yönetmelik)* — 6331 Madde 17 ilk 3'e giremedi | `1.5.6331` MADDE 17 *(Kanun)* — Yönetmelik 3. sırada kaldı | **1.** `7.5.45087` MADDE 5<br>**2.** `1.5.6331` MADDE 17 | Hybrid arama, hem ilgili yönetmelik maddesini hem de ana kanun maddesini ilk 2 sıraya taşıdı. |
| **Soru 3:** Çok tehlikeli sınıfta iş güvenliği uzmanının çalışan başına aylık çalışma süresi nedir? | `7.5.18615` MADDE 19 *(Yanlış: Diğer sağlık personeli)* | `7.5.16923` MADDE 12 *(Doğru: İş güvenliği uzmanı)* | **1.** `7.5.16923` MADDE 12 *(Doğru)* | BM25 kelime benzerliği nedeniyle "diğer sağlık personeli" maddesini 1. sıraya koyarken, Hybrid arama doğru yönetmelik maddesini 1. sıraya çıkardı. |

## 5. Beş Chunking Deneyinde ve Parametrelerde Duyarlılık Analizi

`python -m retrieval.run_parameter_experiments` çalıştırılarak 5 farklı chunking deneyinin, madde numarası ağırlıklandırmasının (`article_boost`) ve BM25/RRF parametrelerinin etkileri incelenmiştir:

### 5.1. 5 Farklı Chunking Deneyinin Karşılaştırması

| Deney Konfigürasyonu | Pasaj Sayısı | Soru 1 (İSG Kurulu) 1. Sıra | Soru 2 (Eğitim Sıklığı) 1. Sıra | Soru 3 (6331 Madde 22) 1. Sıra |
|---|---:|---|---|---|
| `size_500_overlap_0` | 4.089 | `7.5.16924` MADDE 10 *(18.89)* | `1.5.6331` MADDE 17 *(18.47)* | `1.5.6331` MADDE 22 *(27.30)* |
| `size_1000_overlap_0` | 2.193 | **`1.5.6331` MADDE 22** *(18.00)* | **`7.5.45087` MADDE 5** *(24.08)* | **`1.5.6331` MADDE 22** *(27.81)* |
| `size_1000_overlap_100` | 2.267 | **`1.5.6331` MADDE 22** *(17.83)* | **`7.5.45087` MADDE 5** *(24.10)* | **`1.5.6331` MADDE 22** *(27.94)* |
| `size_1000_overlap_200` | 2.266 | **`1.5.6331` MADDE 22** *(17.77)* | **`7.5.45087` MADDE 5** *(23.58)* | **`1.5.6331` MADDE 22** *(27.89)* |
| `size_1500_overlap_0` | 1.610 | `7.5.16924` MADDE 11 *(14.86)* | **`7.5.45087` MADDE 5** *(23.82)* | **`1.5.6331` MADDE 22** *(26.60)* |

**Gözlem:** Çok küçük (`500`) veya çok büyük (`1500`) chunk boyutlarında BM25 uzunluk normalizasyonu nedeniyle Soru 1'de yan yönetmelik maddeleri öne geçerken, **`size_1000` (`overlap 0, 100, 200`)** deneylerinin üçünde de en doğru kanun ve yönetmelik maddeleri 1. sırada yakalanmıştır.

### 5.2. Madde Numarası Farkındalıklı Arama Etkisi

- **Sorgu:** *"6331 sayılı Kanun Madde 22 hükümlerine göre kurul yükümlülüğü nedir?"*
- **Standart BM25 (`article_boost = 0`):** 1. sıraya `7.5.17031 | MADDE 3` (Yönetmelik dayanak maddesi, Skor: `20.8120`) gelmektedir; çünkü kanun numarası (`6331`) yalnızca yönetmeliklerin dayanak cümlesinde açıkça geçmektedir.
- **Madde-Boost BM25 (`article_boost = 10.0`):** `extract_legal_references` ile `Kanun={'6331'}` ve `Madde={'22'}` yakalanarak pasaj metadata'sı üzerinden doğrudan **`1.5.6331 | MADDE 22`** (Skor: `27.8148`) 1. sıraya taşınmıştır.

### 5.3. BM25 Parametre (`k1`, `b`) Analizi

| `k1` | `b` | Soru 1 (6331 Madde 22) Skoru | Soru 2 (45087 Madde 5) Skoru |
|---:|---:|---:|---:|
| 1.2 | 0.50 | 16.0428 | 24.2970 |
| 1.2 | 0.75 | 17.0161 | 23.7399 |
| 1.5 | 0.50 | 16.8316 | 24.7076 |
| **1.5** | **0.75** | **17.9964** | **24.0835** |
| 1.5 | 0.90 | 18.7876 | 23.7248 |
| 2.0 | 0.75 | 19.4047 | 24.5705 |

## 6. Değerlendirme ve Çıktı Formatı

Yusuf'un değerlendirme modülü (Recall@k, MRR) ve Zeynep'in LLM cevap üretim modülü ile entegrasyon için `retrieval/export_results.py` aracı geliştirilmiştir. Bu modül soru veri setini okuyarak her üç yöntemin sıralamalarını 5 deneyin tamamı için standart `runs.jsonl` ve `runs.csv` formatlarında dışa aktarır.
