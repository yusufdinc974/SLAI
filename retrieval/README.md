# Retrieval prototipi

Bu klasör, SLAI projesinin Evrim tarafından yürütülen BM25 ve Hybrid Retrieval
çalışmalarının ilk prototipini içerir. Kod şu anda Python standart
kütüphanesiyle çalışır; harici paket kurulumu gerektirmez.

## Dosyalar

- `bm25.py`: Türkçe karakterleri koruyan tokenizer, `BM25Index` ve ortak
  `RetrievalResult` tipi.
- `rrf.py`: BM25 ve Dense Retrieval sıralamalarını Reciprocal Rank Fusion
  (RRF) ile birleştiren `reciprocal_rank_fusion` fonksiyonu.
- `__init__.py`: Dışarıya açılan sınıf ve fonksiyonlar.

## Kullanım

```python
from retrieval import BM25Index, reciprocal_rank_fusion

passages = {
    "madde-1": "İşveren iş sağlığı ve güvenliği tedbirlerini almak zorundadır.",
    "madde-2": "Çalışanlara iş sağlığı ve güvenliği eğitimi verilir.",
}

bm25 = BM25Index(passages)
bm25_results = bm25.search("iş sağlığı güvenliği", top_k=5)

# Dilara'nın Dense Retrieval çıktısı şimdilik passage_id sıralaması olarak
# verilebilir. Entegrasyonda aynı formatta RetrievalResult kullanılmalıdır.
dense_results = ["madde-2", "madde-1"]
hybrid_results = reciprocal_rank_fusion(
    [bm25_results, dense_results],
    top_k=5,
)

for result in hybrid_results:
    print(result.passage_id, result.score)
```

## Ortak arayüz

Her retrieval yöntemi sıralı sonuç üretmelidir:

- `passage_id`: Korpustaki benzersiz pasaj kimliği.
- `score`: Yöntemin ham veya birleşik skoru.

RRF ham skorları doğrudan karşılaştırmaz; sıralamadaki konumları birleştirir.
Bu nedenle BM25 ve BGE-M3 skorları farklı ölçeklerde olsa da birlikte
kullanılabilir.

## Sonraki aşama

1. 6331 sayılı Kanun ve 24 yönetmelikten oluşan temizlenmiş korpusun ortak
   `passage_id` formatına aktarılması.
2. Gerçek mevzuat pasajlarıyla BM25 indeksinin Colab'da oluşturulması.
3. Dense Retrieval modülünün BGE-M3/Qdrant çıktısının bu arayüze bağlanması.
4. BM25-only, Dense-only ve Hybrid sonuçlarının aynı sorular üzerinde
   kaydedilmesi.
5. Yusuf'un değerlendirme modülü için sonuçların JSON/CSV formatında
   dışa aktarılması.
6. RRF `k` değeri ve yöntem ağırlıkları için deney tablosu oluşturulması.

Bu prototip henüz mevzuat dosyası indirme, BGE-M3 embedding veya Qdrant
kurulumu yapmaz; bunlar entegrasyon aşamasının sorumluluğudur.
