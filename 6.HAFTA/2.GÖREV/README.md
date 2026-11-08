# Hough Dönüşümü ile Doğru Bulma (elle yazılmış)

`HoughDogruElle.py`, `cv2.HoughLines` kullanmadan Hough doğru dönüşümünü baştan yazar. Girdi, 1. görevdeki test resmi (aynı tohumla yeniden üretiliyor). Kenarlar medyan → Gauss → Canny ile bulunuyor (2 089 kenar pikseli).

## Yöntem

Doğrunun normal gösterimi: **ρ = x·cos θ + y·sin θ**, θ ∈ [0°, 180°), ρ ∈ [−D, D]. D, resmin köşegeni (500).

1. **sin / cos tablosu.** 180 açının cos ve sin değerleri döngüden önce bir kez hesaplanıp listelerde tutuluyor. Oylamada her seferinde yeniden hesaplanmıyor; 2 089 × 180 = 376 020 kez tablodan okunuyor.
2. **Oylama matrisi.** `H`, 1001 × 180 boyutunda bir tamsayı matrisi; satırlar ρ + D, sütunlar θ. Oylama iç içe iki açık Python döngüsüyle yapılıyor:
   ```python
   for x, y in kenar_pikselleri:
       for t in range(180):
           rho = x * COS[t] + y * SIN[t]
           H[round(rho) + D, t] += 1
   ```
   Program bir pikselin oylarını adım adım da yazdırıyor:
   ```
   theta =  45: rho = 200*+0.707 + 100*+0.707 =   212.1  ->  H[712, 45] += 1
   ```
3. **Tepe bulma.** Bir hücre doğru sayılmak için iki şartı sağlamalı: oyu eşiği (70) geçmeli ve ±6 piksel ρ, ±6 derece θ penceresinde en büyük olmalı. θ = 180° sınırında (ρ, 180° + t) = (−ρ, t) eşitliği kullanılarak pencere başa sarılıyor.

## Sonuç

7 doğru bulundu:

| # | ρ | θ | Oy | Doğru |
|---|---|---|---|---|
| 1 | 274 | 78° | 294 | Eğik açık çizgi |
| 2 | −256 | 158° | 166 | Üçgenin sağ kenarı |
| 3 | 301 | 22° | 122 | Üçgenin sol kenarı |
| 4 | 260 | 90° | 121 | Üçgenin tabanı |
| 5 | 57 | 0° | 81 | Dikey koyu çizgi |
| 6 | 39 | 90° | 79 | Dikdörtgenin üst kenarı |
| 7 | 141 | 90° | 72 | Dikdörtgenin alt kenarı |

Kontrol için aynı eşikle `cv2.HoughLines` çalıştırıldı: bulunan 7 doğrunun 7'si OpenCV'de de var. OpenCV aynı eşikte 11 doğru döndürüyor, çünkü yerel maksimum bastırması yapmıyor. Fazladan 4 doğrusundan 3'ü tekrar: eğik çizginin, üçgenin sağ kenarının ve tabanının 1-3 derece yanındaki hücreler. Dördüncüsü (ρ = 279, θ = 78°) 3 piksel kalınlığındaki eğik çizginin öbür kenarı. Benim ±6 piksellik penceremde bu kenar, ρ = 274'teki güçlü komşusuna yenik düşüyor. Pencere küçültülürse (`KOMSULUK = 3`) o da bulunuyor, ama tekrarlar da geri geliyor. Oylama 0.09 s sürüyor.

## Oylama matrisini inceleme

- `oylama_matrisi_tepeler.png` (log ölçek, tepeler numaralı): Her kenar pikseli bir sinüs eğrisi çiziyor. Aynı doğru üzerindeki piksellerin eğrileri tek noktada kesişiyor ve o nokta parlak bir tepe oluyor. Uzun doğrular (1 numaralı eğik çizgi) en parlak tepeyi veriyor. Yatay kenarlar (4, 6, 7) θ = 90°'de üst üste diziliyor.
- Tepeler nokta değil, küçük **kelebek / X** şeklinde. Kesişim, θ'nın iki yanına yayılıyor. Kalın ve dalgalı kenarlar yakın hücrelere de oy veriyor. Yerel maksimum şartı bu yüzden gerekli.
- Şekillerin içi boş olsa da kenar pikselleri birçok doğruya oy verdiği için matrisin geniş bir bölgesi düşük oylarla dolu. Bu, Hough'un gürültüye dayanıklılığının bedeli: eşiğin bu "zeminin" üstünde seçilmesi gerekiyor.
- `oylama_ornegi.png`: Solda tek bir pikselin (200, 100) oylarının oluşturduğu sinüs. Sağda x + y = 300 doğrusu üzerindeki 5 noktanın sinüsleri; hepsi tam olarak ρ = 212, θ = 45°'de kesişiyor (oy = 5).

## Çıktılar (`ciktilar/`)

- `bulunan_dogrular.png`: Bulunan doğrular orijinal resim üzerinde, numaralı.
- `hough_elle_karsilastirma.png`: Orijinal, kenar, elle Hough ve cv2.HoughLines.
- `oylama_matrisi_ham.png`, `oylama_matrisi_log.png`, `oylama_matrisi_tepeler.png`
- `oylama_ornegi.png`, `kenar.png`
