# RGB Görüntüyü Gri Tonlu Görüntüye Dönüştürme

`GriDonusum.py`, renkli resmi (1. görevdeki baboon.jpg) iki yolla tek kanallı gri resme dönüştürür: `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` ve kendi yazdığım `kendi_bgr2gray`. Ardından iki sonucun eşleşen piksellerinin **mutlak farklarının toplamını (SAD)** hata olarak raporlar.

## "Doğru şekilde" dönüştürmek

1. **Kanal sırası.** `cv2.imread` resmi BGR sırasıyla okuyor. Kırmızının ağırlığı mavi kanala uygulanırsa sonuç yanlış olur.
2. **Ağırlıklar.** Göz yeşile en duyarlı, maviye en az duyarlı. Bu yüzden düz ortalama değil, ITU-R BT.601 luma ağırlıkları kullanılıyor: **Y = 0.299 R + 0.587 G + 0.114 B**.

   | Saf renk | Luma | Düz ortalama |
   |---|---|---|
   | Kırmızı | 76 | 85 |
   | Yeşil | 150 | 85 |
   | Mavi | 29 | 85 |

   Düz ortalamada yeşil ile mavi aynı griye düşüyor, oysa yeşil gözümüze çok daha parlak görünüyor.

## OpenCV'nin tam formülü

OpenCV 8 bitlik resimde kayan nokta değil, **15 bitlik sabit noktalı** tamsayı aritmetiği kullanıyor:

```
Y = (B*3735 + G*19235 + R*9798 + 16384) >> 15
```

- 9798 = round(0.299·2¹⁵) ve 19235 = round(0.587·2¹⁵).
- Mavinin katsayısı round(0.114·2¹⁵) = 3736 **değil, 3735**. Toplam tam 32768 olsun, beyaz 255'e gitsin diye bir eksik alınmış.

Bu formül, bütün 2²⁴ = 16 777 216 BGR rengi üzerinde denendi ve `cv2.cvtColor` ile her renkte aynı sonucu verdi. İlk denemede eski OpenCV sürümlerinin 14 bitlik katsayılarını (1868, 9617, 4899) kullanmıştım; bunlar 43 864 renkte 1 farklı sonuç veriyor.

## Sonuç (512x512 = 262 144 piksel)

| Yöntem | SAD (hata) | En büyük fark | Farklı piksel |
|---|---|---|---|
| **kendi_bgr2gray (15 bit sabit nokta)** | **0** | 0 | 0 |
| Kayan noktalı formül, `round(0.299R + 0.587G + 0.114B)` | 3 | 1 | 3 |
| Eski 14 bitlik katsayılar | 2 | 1 | 2 |
| HATALI: kanal sırası ters | 2 763 848 | 47 | 249 629 |
| HATALI: düz ortalama | 1 622 955 | 34 | 248 185 |

- Kendi fonksiyonumla OpenCV arasındaki hata **0**.
- Kayan noktalı formül de doğru bir dönüşüm. 3 pikseldeki 1'lik fark, tam .5'e düşen değerlerin farklı yuvarlanmasından geliyor.
- Asıl büyük hatalar kanal sırasını karıştırmaktan (piksel başına ortalama 10.5 gri seviye) ve düz ortalama almaktan (6.2) geliyor. Fark haritalarında hatanın en çok burun (kırmızı) ve yanaklarda (mavi), yani doygun renkli yerlerde olduğu görülüyor (`ciktilar/gri_karsilastirma.png`).

Süre: `cv2.cvtColor` ~2.3 ms, piksel piksel Python döngüsü ~60 ms.
