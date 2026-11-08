# Canny + Medyan + OpenCV HoughCircles ile Çember Bulma

`HoughCemberOpenCV.py`, bir resimde `cv2.HoughCircles` ile çemberleri bulur. Çemberler yeşil, merkezleri kırmızı çiziliyor.

Test resmi programın içinde çiziliyor (`ciktilar/test_resmi.png`). 400x300 boyutunda; içinde farklı parlaklıkta 6 dolu daire (r = 22 … 55) ve tuzak olarak bir kare var. Üzerine σ = 10 Gauss gürültüsü ile %2 tuz ve %2 biber gürültüsü eklendi. Gerçek merkezler ve yarıçaplar bilindiği için program her bulunan çemberin hatasını da yazdırıyor.

## cv2.HoughCircles nasıl çalışıyor?

`HOUGH_GRADIENT` yöntemi 3 boyutlu (a, b, r) matrisi kurmuyor, bunun yerine iki aşamada çalışıyor:

1. Kendi içinde Canny uyguluyor (`param1`) ve Sobel ile her kenar pikselinin gradyan yönünü buluyor.
2. Her kenar pikseli, gradyan doğrultusundaki noktalara **merkez** oyu veriyor (2 boyutlu oylama). Oyu `param2`'yi geçen ve birbirinden `minDist` uzaktaki merkezler için yarıçap, kenar uzaklıklarının histogramından seçiliyor.

Ayarlar: `dp = 1`, `minDist = 20`, `param1 = 100`, `param2 = 24`, `r = 10 … 80`.

## A) Ödevdeki sıra: Canny → medyan (kenar resmine) → HoughCircles

Medyan, 14 686 kenar pikselinin 9 967'sini siliyor. 1. görevdeki sebeple ince Canny kenarları medyanda kayboluyor, geriye tuz-biber kümeleri kalıyor. HoughCircles bu dağınık noktalardan **74 sahte çember** üretiyor (`A_hough_cember.png`).

## B) Sırası değiştirilmiş: medyan → Canny → HoughCircles

| Bulunan merkez | r | Gerçek | Merkez hatası | r hatası |
|---|---|---|---|---|
| (239.5, 214.5) | 55.0 | (240, 215), 55 | 0.7 | 0.0 |
| (89.5, 88.5) | 40.9 | (90, 90), 40 | 1.6 | +0.9 |
| (321.5, 108.5) | 51.3 | (320, 110), 50 | 2.1 | +1.3 |
| (99.5, 220.5) | 35.5 | (100, 220), 35 | 0.7 | +0.5 |
| (220.5, 81.5) | 31.1 | (220, 80), 30 | 1.6 | +1.1 |
| (355.5, 255.5) | 23.2 | (355, 255), 22 | 0.7 | +1.2 |

6 çemberin 6'sı bulundu, sahte çember yok, kare çember olarak alınmadı. Yarıçaplar yaklaşık 1 piksel büyük çıkıyor, çünkü Canny kenarı dairenin sınırının dış tarafındaki piksele düşüyor.

## Gözlemler

- `param2` (merkez oy eşiği) sonucu en çok etkileyen parametre. B kenar resminde: 30 → 3 çember (küçükler kaçıyor), 22-25 → 6 çember, 20 → 7 (karenin köşesinde sahte bir çember), 15 → 16 çember.
- HoughCircles'a ikili kenar resmi verildiğinde, içindeki Canny yalnızca kenar piksellerini bulabiliyor. Gradyan yönü de Sobel'in ince çizgi üzerindeki tepkisinden geliyor. Buna rağmen sonuç iyi, çünkü çizgiye dik Sobel tepkisi yine merkeze bakıyor.

## Çıktılar (`ciktilar/`)

- `hough_cember_karsilastirma.png`: Üst satır A yolu, alt satır B yolu.
- `A_canny.png`, `A_canny_medyan.png`, `A_hough_cember.png`
- `B_medyan_canny.png`, `B_hough_cember.png`
- `test_resmi.png`
