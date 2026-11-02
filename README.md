# ISG RAG Projesi

İş Sağlığı ve Güvenliği (İSG) mevzuatı üzerinde çalışan bir
Retrieval-Augmented Generation (RAG) sistemi geliştirilmesi ve Dense Retrieval
ile Hybrid Retrieval yaklaşımlarının karşılaştırılması hedeflenmektedir.

## Ekip ve görev dağılımı

### Dilara (Kişi 1) — Document Processing ve Dense Retrieval

- 6331 sayılı Kanun ve ilgili yönetmelikleri toplamak, kaynak bilgilerini
  kaydetmek.
- Mevzuat metinlerini temizlemek, madde/fıkra yapısını koruyan chunking
  yaklaşımı geliştirmek.
- Chunk boyutu ve overlap parametrelerini incelemek.
- Her pasaj için `passage_id` ve kaynak metadata'sı oluşturmak.
- BGE-M3 embedding modelini ve vektör indeksini uygulamak.
- Sorular için en alakalı ilk 5 pasajı üretmek.
- Chunking stratejilerinin Dense Retrieval performansına etkisini analiz etmek.

**Teslimatlar:** Temizlenmiş mevzuat korpusu, chunking ve metadata kodu, Dense
Retrieval modülü ve chunking deney sonuçları.

**Araştırma sorusu:** Chunking stratejisi Dense Retrieval performansını nasıl
etkiler?

### Evrim (Kişi 2) — BM25 ve Hybrid Retrieval

- Mevzuat pasajları üzerinde BM25 indeksi ve anahtar kelime/madde numarası
  araması geliştirmek.
- Dense Retrieval sonuçlarını ortak veri formatına dönüştürmek.
- Dense ve BM25 sonuçlarını Reciprocal Rank Fusion (RRF) ile birleştirmek.
- Dense-only, BM25-only ve Hybrid Retrieval sonuçlarını karşılaştırmak.
- Hukuki terim ve madde numarası içeren sorularda performans analizi yapmak.

**Teslimatlar:** BM25 arama modülü, RRF tabanlı Hybrid Retrieval modülü,
sıralanmış retrieval sonuçları ve karşılaştırma raporu.

**Araştırma sorusu:** Dense ve BM25 sonuçlarının birleştirilmesi, tekil
retrieval yöntemlerine kıyasla arama başarısını artırır mı?

### Zeynep (Kişi 3) — LLM Generation ve Grounded Answering

- Donanım koşullarına uygun yerel bir LLM seçmek ve inference sürecini
  yapılandırmak.
- Yalnızca sağlanan mevzuat bağlamına dayalı cevap üretmek için prompt
  tasarlamak.
- Dense ve Hybrid Retrieval sonuçlarını aynı generation modülüne bağlamak.
- Cevaplarla birlikte mevzuat maddelerini ve kaynak bilgilerini sunmak.
- Üretilen cevapları referans cevaplarla karşılaştırmak.
- Desteklenmeyen iddiaları, yanlış madde atıflarını ve eksik cevapları analiz
  etmek.

**Teslimatlar:** LLM generation modülü, prompt/inference yapılandırmaları,
  kaynak gösteren cevap üretim pipeline'ı ve hata analizi raporu.

**Araştırma sorusu:** Retrieval yönteminin değişmesi, cevapların doğruluğunu ve
kaynaklarla desteklenme düzeyini nasıl etkiler?

### Yusuf (Kişi 4) — Retrieval Evaluation ve Statistical Analysis

- Türkçe İSG soruları, referans cevapları ve ilgili pasaj kimliklerinden oluşan
  değerlendirme veri setine katkı sağlamak.
- 45 geliştirme ve 255 test sorusundan oluşan veri bölünmesini yönetmek.
- Recall@1, Recall@3, Recall@5 ve MRR metriklerini uygulamak.
- Dense, BM25 ve Hybrid sonuçlarını aynı test soruları üzerinde karşılaştırmak.
- Soruları madde numarası, hukuki terim, anlamsal/parafraz ve diğer kategorilere
  ayırmak.
- Eşleştirilmiş bootstrap ile güven aralıkları oluşturmak.
- Sonuçları tablo, grafik ve raporlarla sunmak.

**Teslimatlar:** Değerlendirme kodu, retrieval metrikleri, istatistiksel analiz,
kategori bazlı karşılaştırma ve deney sonuçları raporu.

**Araştırma sorusu:** Hybrid Retrieval başarısı tüm soru türlerinde tutarlı mı,
yoksa belirli kategorilerde mi yoğunlaşır?

## Ortak soru veri seti

300 soruluk veri seti aşağıdaki şekilde paylaşılır:

| Ekip üyesi | Soru katkısı | Ek sorumluluk |
|---|---:|---|
| Dilara | 75 soru | Mevzuat kapsamı ve kaynak kontrolü |
| Evrim | 75 soru | Soruların ilgili pasajlarla eşleşme kontrolü |
| Zeynep | 75 soru | Referans cevapların mevzuata uygunluk kontrolü |
| Yusuf | 75 soru | Veri formatı, etiket tutarlılığı ve veri bölünmesi |
| **Toplam** | **300 soru** | |

Her üyenin hazırladığı sorular başka bir ekip üyesi tarafından örneklem
üzerinden çapraz kontrol edilecektir. Geliştirme ve test verileri ayrılacak;
test etiketleri sistem sonuçlarına bakılarak değiştirilmeyecektir.

## Haftalık çalışma planı

| Hafta | Dilara | Evrim | Zeynep | Yusuf |
|---:|---|---|---|---|
| 1 | Mevzuat, chunking ve Dense prototipi | BM25 ve RRF prototipi | LLM ve prompt denemeleri | Metrikler ve deney tasarımı |
| 2 | Dense Retrieval ve chunking deneyleri | Hybrid Retrieval entegrasyonu | Uçtan uca generation pipeline'ı | Değerlendirme kodu ve veri şeması |
| 3 | 75 soru, korpus kontrolü ve Dense testleri | 75 soru ve retrieval testleri | 75 soru ve cevap kalitesi testleri | 75 soru, etiket kontrolü ve analiz |
| 4 | Son deneyler ve dokümantasyon | Retrieval karşılaştırması | Generation hata analizi | İstatistiksel sonuçlar ve rapor |

Modüller ortak arayüzler tamamlandıkça birleştirilecek ve deneyler aynı veri
seti ile tekrarlanabilir biçimde yürütülecektir.

## Ortak teknik standartlar

- Her mevzuat pasajı benzersiz bir `passage_id` taşımalıdır.
- Her soru benzersiz bir `question_id` taşımalıdır.
- Retrieval modülleri sıralanmış pasaj kimliklerini ve skorları standart formatta
  üretmelidir.
- Dense ve Hybrid sistemleri aynı korpusu, test sorularını ve karşılaştırılabilir
  retrieval ayarlarını kullanmalıdır.
- Cevap üretimi karşılaştırmalarında LLM, prompt ve üretim parametreleri sabit
  tutulmalıdır.
- Kod, deney ayarları ve sonuç dosyaları ortak Git deposunda tutulmalıdır.

## Başarı ölçütü

Her ekip üyesi kendi modülünü geliştirmenin yanında deney yürütmekten,
sonuçları yorumlamaktan ve ortak veri setine katkı sağlamaktan sorumludur.
Projenin sonunda Dense, BM25 ve Hybrid Retrieval yöntemlerinin hem retrieval
başarısı hem de kaynaklı cevap kalitesi açısından karşılaştırmalı sonuçları
sunulacaktır.

*Son güncelleme: 2 Kasım 2026*
