# Renk Uzayı Dönüşümlerini Elle Yazma

`RenkUzayiDonusumleri.py`, bir renk uzayındaki görüntüyü başka bir renk uzayına çeviren 7 fonksiyonu baştan yazar ve sonuçları `cv2.cvtColor` ile sayısal olarak karşılaştırır.

## Dönüşümler

| Fonksiyon | OpenCV kodu | OpenCV'nin 8 bitte kullandığı aritmetik |
|---|---|---|
| `bgr2gray` | `COLOR_BGR2GRAY` | 15 bit: R 9798, G 19235, B 3735 |
| `bgr2ycrcb` | `COLOR_BGR2YCrCb` | 14 bit: Y için 4899 / 9617 / 1868; Cr = (R−Y)·11682, Cb = (B−Y)·9241, +128 |
| `bgr2yuv` | `COLOR_BGR2YUV` | 14 bit: aynı Y; U = (B−Y)·8061, V = (R−Y)·14369, +128 |
| `bgr2xyz` | `COLOR_BGR2XYZ` | 12 bit: sRGB (D65) matrisi, katsayılar round(m·4096) |
| `bgr2hsv` | `COLOR_BGR2HSV` | 12 bitlik bölme tabloları, H 0..179 |
| `bgr2hsv_full` | `COLOR_BGR2HSV_FULL` | Aynısı, H 0..255 |
| `ycrcb2bgr` | `COLOR_YCrCb2BGR` | Ters dönüşüm, 14 bit: 22987, −11698, −5636, 29049 |

**Hata neden tam sıfır?** OpenCV 8 bitlik resimleri kayan noktayla değil, sabit noktalı tamsayı aritmetiğiyle dönüştürüyor:
- Katsayılar 2ⁿ ile çarpılıp tamsayıya yuvarlanıyor.
- Sonuç `(x + 2ⁿ⁻¹) >> n` ile geri ölçekleniyor.
- 0..255 dışına taşan değerler kırpılıyor (saturate).

Ders kitabındaki kayan noktalı formül bazı piksellerde 1 farklı sonuç verir. Fonksiyonlar bu yüzden OpenCV'nin katsayıları ve kaydırmalarıyla yazıldı. HSV'de bölme de döngüden önce hesaplanan tablolarla yapılıyor: S için `round(255·2¹²/V)`, H için `round(H_aralığı·2¹²/(6·fark))`.

## Fonksiyon pointer karşılığı

C++'ta dönüşümler aynı imzalı fonksiyonların adreslerini tutan bir diziyle seçilirdi:

```cpp
typedef void (*DonusumFn)(const cv::Mat&, cv::Mat&);
DonusumFn tablo[] = { bgr2gray, bgr2ycrcb, bgr2yuv, ... };
tablo[i](girdi, cikti);
```

Python'da fonksiyonlar birinci sınıf nesne; parantezsiz fonksiyon adı, o fonksiyona bir referanstır. Kodda bunun karşılığı şöyle:

```python
DONUSUMLER = [("BGR -> GRAY", bgr2gray, cv2.COLOR_BGR2GRAY), ...]   # fonksiyon referansı tablosu

def donustur(resim, donusum_fonksiyonu):    # hangi dönüşüm olduğunu bilmeden
    return donusum_fonksiyonu(resim)        # referans üzerinden çağırır
```

## Sonuç

İki test yapıldı:
1. **baboon.jpg** (262 144 piksel).
2. **Bütün 2²⁴ = 16 777 216 BGR rengini birer kez içeren 4096x4096 yapay resim.** Bu test "bu resimde tutuyor" yerine "her olası girdide tutuyor" demeyi sağlıyor. Ters dönüşüm için de her olası (Y, Cr, Cb) üçlüsü denenmiş oluyor.

| Dönüşüm | baboon SAD | 16.7M renk SAD | En büyük fark |
|---|---|---|---|
| BGR → GRAY | 0 | 0 | 0 |
| BGR → YCrCb | 0 | 0 | 0 |
| BGR → YUV | 0 | 0 | 0 |
| BGR → XYZ | 0 | 0 | 0 |
| BGR → HSV | 0 | 0 | 0 |
| BGR → HSV_FULL | 0 | 0 | 0 |
| YCrCb → BGR | 0 | 0 | 0 |
| **Toplam** | **0** | **0** | |

Beklenen hata değeri olan sıfıra ulaşıldı. Numpy ile yazılmış fonksiyonlar 16.7M pikselde 0.14–0.85 s, OpenCV ~3 ms.

**Not: HLS.** OpenCV 8 bitlik HLS'yi kayan noktayla hesaplıyor: önce 1/255 ile float32'ye çeviriyor, sonra H, L, S'yi hesaplıyor. Bunu float32 ile birebir taklit ettim (paydada `2 − (max+min)`, tonda kaynaşık çarp-topla). Sonuç 16.7M rengin 3'ünde 1 farklı çıktı. Bu 3 renkte ton tam olarak x.5'e düşüyor ve OpenCV'nin SIMD kodu son bitte farklı yuvarlıyor. Tamsayı eşleşmesi garanti edilemediği için HLS ana tabloya konmadı.

`ciktilar/renk_uzaylari.png`: Her dönüşümün üç kanalı ayrı gri resimler olarak gösteriliyor, son sütunda |kendi − OpenCV| haritası (hepsi 0).
