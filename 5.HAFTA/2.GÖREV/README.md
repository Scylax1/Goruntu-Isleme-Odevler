# calcHist Fonksiyonunu Elle Yazma

`KendiCalcHist.py`, `cv2.calcHist`'in tek kanallı karşılığı olan `kendi_calc_hist` fonksiyonunu baştan yazar ve renkli resmin (1. görevdeki baboon.jpg) her kanalı için ayrı ayrı çalıştırır.

## Fonksiyon

İmza OpenCV ile aynı. Parametreleri aynı sırayla alıyor ve sonucu aynı biçimde (`histSize x 1`, `float32`) döndürüyor:

```python
kendi_calc_hist(images, channels, mask, histSize, ranges)
cv2.calcHist   (images, channels, mask, histSize, ranges)
```

- **Kutu hesabı** OpenCV'nin düzgün aralıklı histogramıyla aynı: `kutu = floor((değer - alt) * kutu_sayısı / (üst - alt))`. `[alt, üst)` aralığı dışındaki değerler sayılmıyor.
- **Kutu tablosu.** Her olası değerin (0..255) kutusu döngüden önce bir kez hesaplanıyor. Piksel döngüsünde çarpma ya da bölme yok, yalnızca tablodan okuma var.
- **Maske.** Maskenin 0 olduğu pikseller atlanıyor.
- Pikseller açık iki iç içe Python döngüsüyle dolaşılıyor.

## OpenCV ile karşılaştırma

5 ayar × 3 kanal = 15 deneme yapıldı:

| Ayar | Sayılan piksel (B / G / R) | En büyük fark |
|---|---|---|
| 256 kutu, [0, 256) | 262 144 / 262 144 / 262 144 | 0 |
| 32 kutu, [0, 256) | 262 144 / 262 144 / 262 144 | 0 |
| 10 kutu, [0, 256) (eşit bölünmüyor, 25.6) | 262 144 / 262 144 / 262 144 | 0 |
| 50 kutu, [100, 200) (aralık dışı atılıyor) | 97 631 / 166 637 / 147 314 | 0 |
| 256 kutu, ortada daire maske | 90 785 / 90 785 / 90 785 | 0 |

Bütün denemelerde `kendi_calc_hist` ile `cv2.calcHist` **birebir aynı** sonucu verdi.

Süre: Python döngüsü kanal başına ~12 ms, `cv2.calcHist` ~0.1 ms (~100 kat hızlı).

`ciktilar/kendi_calchist.png`: Üst satırda 256 kutulu histogramlar, alt satırda 32 kutulu maskeli histogramlar var. Çubuklar `kendi_calc_hist`'in sonucu, üstteki kesik çizgi ve noktalar `cv2.calcHist`'in sonucu; ikisi tam üst üste düşüyor.
