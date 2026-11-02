# Retrieval testleri

Bu klasör, `retrieval/` altındaki BM25 ve RRF prototiplerinin beklenen
davranışlarını doğrular. Testler Python'un `unittest` modülünü kullanır ve
harici bağımlılık gerektirmez.

## Testleri çalıştırma

Proje kök dizininde:

```bash
python -m unittest discover -s tests -v
```

Google Colab'da proje GitHub'dan klonlandıktan sonra:

```python
%cd SLAI
!python -m unittest discover -s tests -v
```

Başarılı çalışmada tüm testlerin `OK` ile tamamlanması beklenir.

## Mevcut kontroller

- Türkçe `İ`, `ş`, `ğ` gibi karakterlerin doğru tokenize edilmesi.
- BM25 aramasının skor sırasına göre sonuç döndürmesi.
- Boş sorgunun güvenli biçimde boş sonuç döndürmesi.
- Geçersiz BM25 ve RRF parametrelerinin reddedilmesi.
- Dense ve BM25 sıralamalarının RRF ile birleştirilmesi.
- RRF ağırlıklarının ve `top_k` sınırının uygulanması.
- Ağırlık sayısının ranking sayısıyla eşleşmesinin zorunlu olması.

## Sonraki test aşaması

Gerçek mevzuat korpusu ve Dense Retrieval bağlandığında aşağıdaki testler
eklenecektir:

1. Aynı pasaj korpusunun BM25 ve Dense yöntemlerinde kullanıldığını doğrulama.
2. `passage_id` değerlerinin benzersiz ve eksiksiz olduğunu kontrol etme.
3. Gerçek Türkçe İSG sorgularında beklenen ilgili maddelerin ilk 5 içinde
   bulunmasını doğrulama.
4. BM25-only, Dense-only ve Hybrid sonuçlarını aynı test sorularında
   karşılaştırma.
5. Recall@1, Recall@3, Recall@5 ve MRR hesaplarını Yusuf'un değerlendirme
   modülüyle doğrulama.
6. RRF sonuçlarının JSON/CSV çıktı şemasına uygunluğunu kontrol etme.

Testler, retrieval skorlarının belirli sayısal değerlerine değil; sıralama,
kimlik ve çıktı sözleşmelerine odaklanır. Böylece model veya korpus değişse
bile temel davranış kontrolleri korunur.
