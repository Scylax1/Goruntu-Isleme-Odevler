# Resme Gauss ve Tuz-Biber Gürültüsü Ekleme, Harmonik Matrisleri

`ResimGurultu.py`

```python
gauss_gurultu_ekle(resim, sigma, tohum=None)
tuz_biber_ekle(resim, oran, tuz_payi=0.5, tohum=None)
```

- **`gauss_gurultu_ekle`:** Her piksele bağımsız N(0, σ²) gürültü ekliyor ve sonucu 0..255'e kırpıyor. σ gürültünün **kuvveti**.
- **`tuz_biber_ekle`:** Piksellerin `oran` kadarını bozuyor. Bozulanların `tuz_payi` kadarı 255 (tuz), gerisi 0 (biber) oluyor. `oran` gürültünün **oranı**.

Harmonik matrisleri 7. görevdeki `dft2` ile hesaplandı.

## Sonuç (cameraman 512x512)

| Resim | Ort. parlaklık | Eklenen std | Düşük frekans ort. \|F\| (r = 5-20) | Yüksek frekans ort. \|F\| (r = 100-200) | Beyaz gürültüden beklenen |
|---|---|---|---|---|---|
| Orijinal | 129.06 | 0 | 185 027 | 5 776 | — |
| Gauss σ = 15 | 129.23 | 14.7 | 184 067 | 8 925 | 6 651 |
| Gauss σ = 40 | 130.22 | 36.4 | 176 244 | 17 450 | 16 531 |
| Tuz-biber %5 | 128.94 | 33.1 | 176 859 | 16 028 | 14 998 |
| Tuz-biber %20 | 128.68 | 66.0 | 151 556 | 29 574 | 29 969 |

Beklenen gürültü genliği ≈ std · √(MN) · √π/2.

## Yorum

- **Gürültü yüksek frekanslara düz bir taban ekliyor** (`radyal_spektrum.png`). Gürültü pikselden piksele bağımsız (beyaz) olduğu için enerjisi bütün frekanslara eşit dağılıyor.
  - Orijinal resmin genliği frekansla hızla düşüyor.
  - Düşük frekanslarda resim tabandan 10-30 kat güçlü olduğu için gürültü fark edilmiyor.
  - Yaklaşık 30-50. harmonikten sonra taban resmin üstüne çıkıyor ve oradaki ince ayrıntıyı tamamen örtüyor.
- **Ölçülen taban kuramla uyuşuyor.** σ = 15'te yüksek frekanstaki 8 925, resmin kendi 5 776'sı ile gürültünün 6 651'inin güç toplamı (√(5776² + 6651²) ≈ 8 800). Daha güçlü gürültülerde taban doğrudan beklenen değere oturuyor (16 531 → 17 450, 29 969 → 29 574).
- **Harmonik matrisi görüntülerinde** (`gurultu_harmonik_matrisleri.png`):
  - Orijinalin köşeleri koyu; gürültülülerde köşeler parlak ve düz renkli.
  - Merkezdeki yıldız deseni (tripod, kenarlar) gürültü arttıkça tabanın içinde kayboluyor ama tamamen silinmiyor.
  - Fark haritası (log|F| − log|F_orijinal|) merkezde ~0, dışa doğru giderek kırmızı. Gürültü düşük frekansları neredeyse hiç değiştirmiyor, yüksek frekansları ise yüzlerce kat güçlendiriyor.
- **Tuz-biber düşük frekansları da etkiliyor.** %20'de düşük frekans genliği 185 bin'den 152 bin'e düşüyor (%18). Tuz-biber piksele bir şey **eklemiyor**, pikselin yerine 0 ya da 255 **koyuyor**; resmin %20'si yok oluyor ve resim 0.8 katına iniyor. Gauss σ = 40'ta da benzer küçük bir düşüş var; 0 ve 255'te kırpma resmin uç değerlerini değiştiriyor.
- **Ortalama parlaklık (DC) neredeyse aynı kalıyor.** İki gürültünün de ortalaması ~0: Gauss sıfır ortalamalı, tuz ve biber yarı yarıya. Gauss σ = 40'ta parlaklık 129 → 130: koyu bölgelerde negatife giden değerler 0'a kırpılıyor ve ortalama hafifçe yukarı kayıyor.
