# Belge İşleme, Chunking, Dense Embedding ve Vectorstore

## 1. Belge İşleme (Document Processing)

Bu aşamanın amacı, ISG mevzuatına ait PDF belgelerini RAG sisteminde kullanılabilecek yapılandırılmış metinlere dönüştürmektir.

Belge işleme süreci dört temel adımdan oluşur:

- **Document Loading:** `data/raw/` klasöründeki desteklenen PDF ve DOCX dosyalarının bulunması.
- **Text Extraction:** PDF ve DOCX dosyalarındaki metinlerin sayfa bilgileri korunarak çıkarılması.
- **Text Cleaning:** Gereksiz boşlukların, satır sonlarının ve null karakterlerinin temizlenmesi.
- **Document Processing:** Belgelerin metadata bilgileriyle birlikte JSON formatında `data/processed/` klasörüne kaydedilmesi.

Her belge için belge kimliği, başlık, kategori, kaynak dosya yolu, sayfa sayısı ve karakter sayısı gibi metadata bilgileri oluşturulur.

Ayrıca metin çıkarılamayan sayfalar tespit edilerek OCR gereksinimi işaretlenir.

## 2. Chunking

Belge işleme aşamasından elde edilen JSON dosyaları, retrieval sisteminin kullanabileceği daha küçük metin parçalarına ayrılır.

Chunking işlemi mevzuatın yapısı dikkate alınarak gerçekleştirilir. Özellikle MADDE, GEÇİCİ MADDE, EK MADDE, EK, fıkra ve bölüm başlıkları korunmaya çalışılır.

Farklı chunk boyutlarının retrieval performansına etkisini incelemek amacıyla beş deney yapılandırması oluşturulmuştur:

| Chunk Size | Overlap |
|---|---|
| 500 | 0 |
| 1000 | 0 |
| 1000 | 100 |
| 1000 | 200 |
| 1500 | 0 |

Her chunk için aşağıdaki metadata bilgileri saklanır:

- `passage_id`
- `document_id`
- `title`
- `category`
- `section_type`
- `section_number`
- `section_marker`
- `chapter_title`
- `paragraph_numbers`
- `passage_index`
- `chunk_size`
- `overlap`
- `character_count`

Bu metadata yapısı, retrieval sonuçlarının ilgili mevzuat maddesi ve paragrafıyla ilişkilendirilmesini sağlar.

Çıktılar `data/chunks/` altında deney bazında saklanır.

## 3. Dense Embedding Generation

Chunk'ların anlamsal olarak aranabilir hale getirilmesi için **BAAI/bge-m3** embedding modeli kullanılmıştır.

Embedding üretiminde:

- Model CPU üzerinde çalıştırılır.
- Maksimum token uzunluğu 512 olarak belirlenir.
- Batch size 16 kullanılır.
- Embedding vektörleri normalize edilir.
- Her chunk'ın metadata bilgileri embedding kaydında korunur.

BGE-M3 modeli 1024 boyutlu vektörler üretir.

Embedding çıktıları `data/embeddings/` klasöründe deney bazında JSON dosyalarına kaydedilir.

Bu yapı sayesinde farklı chunking deneylerinin embedding sonuçları birbirinden bağımsız olarak saklanabilir ve karşılaştırılabilir.

## 4. Vectorstore ve Qdrant

Üretilen embedding vektörleri, dense retrieval işlemlerinde kullanılmak üzere Qdrant vektör veritabanına aktarılır.

Qdrant yapılandırmasında:

- Vector size: 1024
- Distance metric: Cosine
- Upload batch size: 128
- Point ID: UUID5 ile deterministik olarak oluşturulur.

Her chunk'ın `passage_id` ve diğer metadata bilgileri Qdrant payload alanında saklanır.

Beş farklı chunking deneyine karşılık beş ayrı collection oluşturulmuştur:

- `isg_size_500_overlap_0`
- `isg_size_1000_overlap_0`
- `isg_size_1000_overlap_100`
- `isg_size_1000_overlap_200`
- `isg_size_1500_overlap_0`

Veri aktarımından sonra Qdrant üzerindeki point sayısı, beklenen embedding sayısıyla karşılaştırılarak doğrulanır.

Mevcut vektör verileri yerel olarak saklanır; büyük veri dosyaları GitHub deposuna dahil edilmez.

## 5. Kullanılan Teknolojiler

| Teknoloji | Kullanım amacı |
|---|---|
| Python | Geliştirme dili |
| PyMuPDF | PDF metin çıkarma |
| python-docx | DOCX metin çıkarma |
| BGE-M3 | Dense embedding üretimi |
| Sentence Transformers | Embedding modelini çalıştırma |
| PyTorch | Model altyapısı |
| Qdrant | Vektör veritabanı |
| JSON | Ara veri ve embedding kayıtları |

## 6. Sorumluluk Kapsamı

Bu çalışma kapsamında belge işleme, mevzuat yapısına uygun chunking, dense embedding üretimi ve Qdrant indeksleme modülleri geliştirilmiştir.

Ayrıca farklı chunk size ve overlap değerleri için deney altyapısı hazırlanmış, embedding ve Qdrant kayıtlarının sayısal tutarlılığını kontrol eden doğrulama adımları uygulanmıştır.