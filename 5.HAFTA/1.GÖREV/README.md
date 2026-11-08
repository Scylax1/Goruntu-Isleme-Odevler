# Renkli Görüntünün Kanal Histogramları

`KanalHistogramlari.py`, renkli bir resmi `cv2.imread` ile okuyup her kanal için ayrı histogramı `cv2.calcHist` ile çıkarır.

**Girdi:** `ciktilar/baboon.jpg`. OpenCV'nin örnek resimlerinden biri, 512x512 boyutunda, 3 kanallı. Başka bir resim için: `python KanalHistogramlari.py resim.jpg`.

## Dikkat edilen noktalar

- `cv2.imread` kanalları **B, G, R** sırasıyla okur: kanal 0 mavi, kanal 1 yeşil, kanal 2 kırmızıdır. Matplotlib RGB beklediği için resim gösterilirken `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` ile çevriliyor. Çevrilmezse babunun kırmızı burnu mavi görünür.
- `cv2.calcHist([resim], [kanal], None, [256], [0, 256])` çağrısındaki parametreler sırasıyla: resim listesi, kanal indeksi, maske yok, 256 kutu, [0, 256) aralığı.
- Her histogramın toplamının piksel sayısına (262 144) eşit olduğu program içinde kontrol ediliyor.

## Sonuç

| Kanal | Ortalama | Std | Min | Max | En sık değer |
|---|---|---|---|---|---|
| Mavi (B) | 112.9 | 60.6 | 0 | 255 | 75 |
| Yeşil (G) | 129.1 | 47.5 | 0 | 238 | 193 |
| Kırmızı (R) | 137.1 | 55.5 | 0 | 255 | 121 |

`ciktilar/kanal_histogramlari.png` üst satırda orijinal resmi ve her kanalı gri resim olarak, alt satırda üç histogramı ayrı ayrı ve üst üste gösteriyor:

- **Mavi** histogramın 220-230 civarında ayrı bir tepesi var. Bu tepe, yanaklardaki açık mavi bölgelerden geliyor; mavi kanal resminde yanaklar en parlak yer.
- **Kırmızı** histogramın 240-250 civarındaki tepesi kırmızı burundan geliyor; kırmızı kanalda burun neredeyse beyaz.
- **Yeşil** histogram 190 civarında en sık değerine ulaşıp 238'de bitiyor. Yeşil kanalda hiç doymuş (255) piksel yok. Tüylerin sarı-yeşil tonu bu kanalı orta-yüksek değerlerde topluyor.
