# ISG RAG Projesi

## Proje açıklaması

Bu proje, Türkiye'deki İş Sağlığı ve Güvenliği (İSG) mevzuatı üzerinde çalışan
yapı-farkındalıklı bir Hybrid Retrieval-Augmented Generation (RAG) sistemi
geliştirmeyi amaçlar. Sistem; 6331 sayılı İş Sağlığı ve Güvenliği Kanunu ile
24 ikincil mevzuat metnini Madde, Fıkra ve Bent hiyerarşisini koruyarak işler.
Hedef kullanıcılar İSG uzmanları, hukuki uyum denetçileri, işverenler ve
Türkiye'deki çalışanlardır.

Temel araştırma sorusu şudur:

> Yapı-farkındalıklı Hybrid RAG pipeline'ı, aynı Qwen3-8B modelinin retrieval
> kullanılmayan temel (baseline) sürümüne göre Türkçe İSG mevzuatı sorularında
> cevap doğruluğunu ve kaynaklara bağlılığını artırır mı?

## Kullanılan yöntem

1. **Mevzuatın yapısal işlenmesi:** Mevzuat, Madde/Fıkra/Bent hiyerarşisi ve
   başlık bilgileri korunarak parçalara ayrılır. Ek tabloların metin
   çıkarımında hizası bozulursa Markdown veya anahtar-değer JSON biçiminde
   serileştirilir.
2. **İndeksleme:** Parçalar Qdrant üzerinde BGE-M3 dense ve sparse vektörleri
   ile indekslenir.
3. **Hybrid retrieval:** Her soru için en iyi 20 hibrit sonuç alınır; reranker
   uygulanarak en iyi 5 pasaj seçilir.
4. **Cevap üretimi:** Seçilen pasajlar yerel Qwen3-8B modeline bağlam olarak
   verilir. Donanım elverirse Gemma 3 12B ile çapraz kontrol yapılır.
5. **Baseline karşılaştırması:** Aynı Qwen3-8B modeli retrieval olmadan,
   zero-shot biçimde çalıştırılır. Böylece kazanım yalnızca model farkından
   değil, retrieval yaklaşımından ölçülür.

## Veri seti ve deney tasarımı

- Korpus: 6331 sayılı Kanun ve 24 yönetmelik olmak üzere yaklaşık 1.450 madde.
- Değerlendirme seti: 250 Türkçe soru-cevap çifti.
- Soru türleri: 150 çoktan seçmeli ÖSYM tarzı soru ve 100 açık uçlu soru.
- Bölünme: 50 geliştirme ve 200 test sorusu; eğitim seti kullanılmaz.
- Tüm yöntemler aynı soru seti, korpus ve model ayarlarıyla karşılaştırılır.
- Pilot çalışmada retrieval kullanılmayan Qwen3-8B, 20 soruda %55 doğruluk
  göstermiş ve güncel olmayan para cezası tutarları gibi uydurma bilgiler
  üretmiştir. Bu sonuç, RAG yaklaşımının kaynaklı ve güncel cevap üretme
  potansiyelini test etmek için başlangıç noktasıdır.

## Başarı ölçütleri

- **Retrieval:** Doğru madde veya pasajın bulunup bulunmadığı.
- **Çoktan seçmeli sorular:** Doğru seçeneğin işaretlenmesi.
- **Açık uçlu sorular:** Doğruluk, doğru mevzuat atfı ve uydurma bilgi
  üretmeme.
- **Kaynaklı cevap kalitesi:** Citation F1, RAGAS ve hata sınıflandırması.
- **Başarı hedefi:** RAG sisteminin baseline modele göre en az 5 puan daha iyi
  sonuç vermesi ve farkın istatistiksel testlerle rastlantısal olmadığının
  gösterilmesi.

## Proje çıktıları ve yol haritası

- Yapısal olarak ayrıştırılmış mevzuat korpusu.
- 250 soruluk değerlendirme veri seti.
- Qdrant hibrit indeksi ve reranking pipeline'ı.
- Baseline ve RAG ölçüm sonuçları.
- Hata analizi, demo ve yeniden üretilebilir deney yapılandırmaları.
- Uzun vadede IEEE SIU veya ACL/EMNLP Legal NLP gibi bir çalıştaya
  gönderilebilecek araştırma çıktısı.

## Riskler ve önlemler

| Risk | Önlem |
|---|---|
| Ek tabloların metin çıkarımında hizasını kaybetmesi | Tabloları Markdown veya anahtar-değer JSON olarak serileştirmek |
| Gemma 3 12B, BGE-M3 ve reranker'ın T4 GPU belleğini aşması | 4-bit Qwen3-8B kullanmak; embedding ve reranking işlemlerini sırayla çalıştırmak |

---

## Ekip ve görev dağılımı

## Evrim'in 1. hafta prototipi

BM25 ve RRF prototipi, `retrieval/` altında Python standart kütüphanesiyle
uygulanmıştır:

```python
from retrieval import BM25Index, reciprocal_rank_fusion

bm25 = BM25Index({
    "madde-1": "İşveren iş sağlığı ve güvenliği tedbirlerini alır.",
    "madde-2": "Çalışanların eğitim ve denetim yükümlülüğü.",
})

bm25_results = bm25.search("iş sağlığı", top_k=5)
hybrid_results = reciprocal_rank_fusion(
    [bm25_results, ["madde-2", "madde-1"]],  # ikinci liste Dense sonuçlarıdır
    top_k=5,
)
```

Her iki yöntem de `passage_id` ve `score` alanlarına sahip aynı sonuç formatını
kullanır. RRF ham skorları değil, sıralama konumlarını birleştirir; bu nedenle
BM25 ve Dense skorlarının farklı ölçeklerde olması sorun oluşturmaz.

Testleri çalıştırmak için:

```bash
python -m unittest discover -s tests -v
```

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
- 50 geliştirme ve 200 test sorusundan oluşan veri bölünmesini yönetmek.
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

250 soruluk veri seti aşağıdaki şekilde paylaşılır:

| Ekip üyesi | Soru katkısı | Ek sorumluluk |
|---|---:|---|
| Dilara | 62 soru | Mevzuat kapsamı ve kaynak kontrolü |
| Evrim | 63 soru | Soruların ilgili pasajlarla eşleşme kontrolü |
| Zeynep | 62 soru | Referans cevapların mevzuata uygunluk kontrolü |
| Yusuf | 63 soru | Veri formatı, etiket tutarlılığı ve veri bölünmesi |
| **Toplam** | **250 soru** | |

Her üyenin hazırladığı sorular başka bir ekip üyesi tarafından örneklem
üzerinden çapraz kontrol edilecektir. Geliştirme ve test verileri ayrılacak;
test etiketleri sistem sonuçlarına bakılarak değiştirilmeyecektir.

## Haftalık çalışma planı

| Hafta | Dilara | Evrim | Zeynep | Yusuf |
|---:|---|---|---|---|
| 1 | Mevzuat, chunking ve Dense prototipi | BM25 ve RRF prototipi | LLM ve prompt denemeleri | Metrikler ve deney tasarımı |
| 2 | Dense Retrieval ve chunking deneyleri | Hybrid Retrieval entegrasyonu | Uçtan uca generation pipeline'ı | Değerlendirme kodu ve veri şeması |
| 3 | 62 soru, korpus kontrolü ve Dense testleri | 63 soru ve retrieval testleri | 62 soru ve cevap kalitesi testleri | 63 soru, etiket kontrolü ve analiz |
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

## Teknik Dokümantasyon
- [Belge İşleme, Chunking, Dense Embedding ve Vectorstore](docs/dilara-calisma.md)
- [BM25, Dense Entegrasyonu ve Hybrid Retrieval (RRF)](docs/evrim-calisma.md)

*Son güncelleme: 6 Kasım 2026*