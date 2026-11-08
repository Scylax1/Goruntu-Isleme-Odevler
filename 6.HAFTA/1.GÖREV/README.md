# Canny + Medyan + OpenCV Hough ile Doğru Bulma

`HoughDogruOpenCV.py`, bir resimde OpenCV'nin iki farklı Hough fonksiyonuyla doğruları bulup resmin üzerine çizer.

| Fonksiyon | Ne döndürür | Renk |
|---|---|---|
| `cv2.HoughLines` (standart Hough) | Sonsuz doğrular, `(rho, theta)` | Kırmızı |
| `cv2.HoughLinesP` (olasılıksal Hough) | Uç noktalı doğru parçaları, `(x1, y1, x2, y2)` | Yeşil |

Test resmi programın içinde çiziliyor (`ciktilar/test_resmi.png`). 400x300 boyutunda; içinde dikdörtgen, üçgen, bir eğik ve bir dikey kalın çizgi var. Üzerine σ = 10 Gauss gürültüsü ile %2 tuz ve %2 biber gürültüsü eklendi. Doğruların yeri bilindiği için sonuç gözle denetlenebiliyor. Başka bir resim için: `python HoughDogruOpenCV.py resim.png`.

## A) Ödevdeki sıra: Canny → medyan (kenar resmine) → Hough

| | Değer |
|---|---|
| Canny kenar pikseli | 15 647 |
| Medyanın sildiği kenar pikseli | 10 882 |
| Medyandan sonra kalan / yeni oluşan | 7 616 piksel |
| `cv2.HoughLines` (eşik 70) | 1 doğru, gürültüden gelen sahte bir doğru |
| `cv2.HoughLinesP` | 0 parça |

**Medyan, kenar resminde doğruları siliyor.** Canny kenarları 1 piksel inceliğinde. İkili (0/255) bir resimde 3x3 medyan, 9 pikselin çoğunluk oyu gibi çalışır: bir pikselin kenar kalması için penceresindeki 9 pikselden en az 5'inin kenar olması gerekir. İnce bir doğru üzerindeki pikselin penceresinde yalnızca 3 kenar pikseli vardır (3/9). Bu yüzden medyan doğruların hepsini siliyor. Tuz-biber noktalarının Canny'de oluşturduğu küçük kapalı halkalar ise yoğun oldukları için kalıyor, hatta birbirine yakın olanlar birleşiyor (`ciktilar/A_canny_medyan.png`). Kenar resmine uygulanan medyan, istenenin tersini yapıyor.

## B) Sırası değiştirilmiş: medyan → Canny → Hough

Medyan, gürültüyü kenar bulunmadan önce gri resim üzerinde temizlediğinde tuz-biber gürültüsü tamamen gidiyor ve geriye yalnızca şekillerin kenarları kalıyor (2 089 piksel).

| Fonksiyon | Sonuç |
|---|---|
| `cv2.HoughLines` (eşik 70) | 11 doğru: çizgiler, üçgenin iki kenarı ve tabanı, dikdörtgenin üst ve alt kenarı, dikey çizgi. Kalın çizgiler 2 kez bulunuyor (iki kenarı var) ve aynı doğrunun 1-2 derece yanında tekrarlar çıkıyor. |
| `cv2.HoughLinesP` | 9 parça: eğik çizginin iki kenarı, üçgenin üç kenarı, dikdörtgenin üst ve alt kenarı, dikey çizginin iki kenarı. Parçalar şekillerin gerçek uzunluğunda. |

## Gözlemler

- **HoughLines ve HoughLinesP farkı.** `HoughLines` doğruyu sonsuz uzatıyor, nerede başlayıp bittiğini söylemiyor. Ayrıca yerel maksimum bastırması yapmıyor: eşiği geçen her `(rho, theta)` hücresini döndürüyor, bu yüzden güçlü bir doğrunun yanındaki hücreler tekrar olarak çıkıyor. `HoughLinesP` her doğruyu piksel piksel izleyip parçalara bölüyor. Kısa parçaları (`minLineLength = 60`) atıyor, küçük boşlukları (`maxLineGap = 10`) birleştiriyor. Bu yüzden şekil kenarlarını çok daha temiz veriyor.
- **Dikdörtgenin dikey kenarları (100 piksel) bulunamadı.** Gürültü yüzünden Canny kenarı 1-2 piksel sağa sola dalgalanıyor ve oylar birkaç `rho` hücresine bölünüyor. Eşik 60'a indirilince bunlar da bulunuyor, ama sahte tekrarlar da artıyor.

## Çıktılar (`ciktilar/`)

- `hough_dogru_karsilastirma.png`: Üst satır A yolu, alt satır B yolu. B'de HoughLines kırmızı, HoughLinesP yeşil, sağda ikisi birlikte.
- `A_canny.png`, `A_canny_medyan.png`, `A_hough_sonuc.png`
- `B_medyan_canny.png`, `B_hough_sonuc.png`
- `test_resmi.png`
