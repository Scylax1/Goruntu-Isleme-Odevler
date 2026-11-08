# Vektöre Genliğe ve Orana Bağlı Rastgele Gürültü Ekleme

`GurultuEkle.py`

```python
gurultu_ekle(vektor, oran, tohum=None)  ->  gürültülü yeni vektör
```

Fonksiyon şu adımları izliyor:

- **Genlik:** `max |x − ortalama|`, sinyalin tepe genliği.
- **Gürültü kuvveti:** `sigma = oran · genlik`, oran ∈ [0, 1]. Aralık dışındaki oran `ValueError` veriyor.
- **Ekleme:** Her elemana bağımsız N(0, sigma²) Gauss gürültüsü ekleniyor. Gauss sayıları Box-Muller dönüşümüyle iki düzgün dağılımlı sayıdan elle üretiliyor: `z = sqrt(−2 ln u1) · cos(2π u2)`.
- Girdi vektörü değiştirilmiyor; kopyası döndürülüyor. Aynı `tohum` aynı gürültüyü üretiyor.

Gürültü genliğe göre ölçeklendiği için aynı oran her sinyalde aynı sinyal/gürültü oranını (SNR) veriyor. A = 1 ve A = 12 olan sinüslerde oran 0.3 ikisinde de **8.2 dB** verdi.

## Sonuç (A = 5, f = 4 Hz, N = 256)

| Oran | Sigma | Ölçülen std | SNR (dB) | Kuramsal SNR = 10 log₁₀(1 / 2·oran²) |
|---|---|---|---|---|
| 0.0 | 0 | 0 | ∞ | ∞ |
| 0.1 | 0.5 | 0.52 | 16.7 | 17.0 |
| 0.3 | 1.5 | 1.55 | 7.2 | 7.4 |
| 0.6 | 3.0 | 3.10 | 1.2 | 1.4 |
| 1.0 | 5.0 | 5.16 | −3.3 | −3.0 |

`ciktilar/gurultu_oranlari.png`: Her oran için gürültülü sinyal (kırmızı) ve temiz sinyal (siyah), sağda eklenen gürültünün histogramı ve kuramsal Gauss eğrisi.
- Oran 0.3'te sinüs hâlâ açıkça seçiliyor.
- Oran 1.0'da (SNR < 0 dB) gürültünün gücü sinyalinkini geçiyor ve gözle sinüs zor seçiliyor.

Bütün satırlarda aynı tohum kullanıldı. Bu yüzden gürültünün biçimi aynı, yalnızca ölçeği değişiyor; oranların etkisi doğrudan karşılaştırılabiliyor.
