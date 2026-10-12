# Prewitt ve Sobel Operatörlerinin Karşılaştırılması

## 1. Operatörler

Her iki operatör de 3x3 boyutunda, birinci türevi yaklaşık olarak hesaplayan kenar bulma filtresidir. Biri x, diğeri y yönündedir:

| | x yönü (dikey kenar) | y yönü (yatay kenar) |
|---|---|---|
| **Prewitt** | `[-1 0 1; -1 0 1; -1 0 1]` | `[-1 -1 -1; 0 0 0; 1 1 1]` |
| **Sobel** | `[-1 0 1; -2 0 2; -1 0 1]` | `[-1 -2 -1; 0 0 0; 1 2 1]` |

İki operatör de **yumuşatma × türev** şeklinde ayrıştırılabilir. Türev kısmı ortaktır: `[-1 0 1]`. Fark, türeve dik yöndeki yumuşatmadadır:

- **Prewitt**, üç satırı eşit ağırlıkla ortalar: `[1 1 1]`.
- **Sobel**, orta satıra iki kat ağırlık verir: `[1 2 1]`. Bu küçük bir Gauss yaklaşımıdır.

Kenar büyüklüğü her ikisinde de `|G| = sqrt(Gx² + Gy²)` ile hesaplanır.

## 2. Deney düzeni

`PrewittSobelKarsilastirma.py` scripti aynı girdiyi iki operatöre de veriyor. Girdi, 3x3 Gauss ile yumuşatılmış 512x512 cameraman resmidir. Konvolüsyon 3. görevdeki elle yazılmış fonksiyonla yapılıyor.

Adil bir kıyas için çekirdekler normalize edildi: Prewitt 1/6, Sobel 1/8 ile çarpıldı. Böylece birim eğimli bir rampada ikisi de aynı değeri veriyor ve geriye kalan farklar yalnızca çekirdeğin şeklinden geliyor.

## 3. Sonuçlar

| Ölçüm | Prewitt | Sobel |
|---|---|---|
| Ortalama \|G\| | 4.50 | 4.56 |
| En büyük \|G\| | 80.5 | 84.4 |
| Kenar pikseli (\|G\| > 12) | 22 936 | 23 301 |
| Gürültü (σ = 10) sonrası \|G\| değişimi (RMS) | **2.20** | 2.30 |
| Gürültüyle yeni çıkan sahte kenar pikseli | **2 506** | 2 753 |
| Disk kenarında ortalama yön hatası | 1.28° | **0.83°** |

- **Kenar haritaları neredeyse aynı.** İki büyüklük haritası arasındaki korelasyon 0.9996, eşiklenmiş kenar haritalarının örtüşmesi (ortak / birleşim) %97. Fark görüntüsü (`ciktilar/prewitt_sobel_karsilastirma.png`, sağ üst) 16 kat büyütülünce ancak seçilebiliyor. Piksel başına ortalama fark 0.14 gri seviye.
- **Sobel keskin kenarlarda biraz daha güçlü tepki veriyor** (en büyük değer 84.4'e karşı 80.5). Orta satırın ağırlığı yüksek olduğu için, kenarın tam üstündeki piksel sonuca daha çok katkı yapıyor. Kenar daha "ince ve parlak" çıkıyor.
- **Gürültüye Sobel biraz daha duyarlı.** Bu sonuç ilk bakışta beklenmedik görünebilir. Normalize edilmiş çekirdeklerin beyaz gürültüyü büyütme oranı Prewitt'te √6/6 ≈ 0.41, Sobel'de √12/8 ≈ 0.43'tür. Yani teorik oran ≈ 1.06, ölçülen oran 2.30 / 2.20 ≈ 1.05. Prewitt dokuz pikselin altısını eşit ağırlıkla kullandığı için gürültüyü daha geniş ortalıyor. Sobel'in "gürültüye dayanıklı" diye anılması, yumuşatma içermeyen basit türeve (`[-1 0 1]`, Roberts) göre geçerlidir; Prewitt'e göre değil. Fark küçük olduğu için resimde gözle seçilmiyor.
- **Asıl fark yön doğruluğunda.** Yapay bir diskin kenarında, operatörün bulduğu gradyan yönü gerçek kenar normaliyle kıyaslandı. Sobel'in ortalama açı hatası Prewitt'ten yaklaşık **%35 daha düşük** (0.83°'ye karşı 1.28°). Sebebi şu: `[1 2 1]` ağırlıklandırması çekirdeğin tepkisini yönden daha bağımsız (izotropik) yapıyor. Prewitt'in kare biçimli eşit ağırlıkları ise çapraz yönlerde gradyanı eksenlere doğru hafifçe saptırıyor.

## 4. Değerlendirme

Önce Gauss ile yumuşatılmış bir resimde iki operatörün kenar haritaları pratikte ayırt edilemiyor. Ön yumuşatma, Sobel'in kendi içindeki yumuşatmanın sağlayacağı avantajın büyük kısmını zaten veriyor.

- **Yalnızca kenar büyüklüğü** gerekiyorsa (eşikleyip kenar haritası çıkarmak gibi), iki operatör de iş görür. Prewitt gürültüye çok az daha dayanıklıdır ve tamsayı ağırlıkları biraz daha basittir.
- **Kenar yönü** de kullanılacaksa Sobel tercih edilmelidir. Canny'deki maksimum olmayanı bastırma, gradyan yönü histogramları (HOG) ve köşe bulma bunlara örnektir. Daha doğru açı vermesi, OpenCV gibi kütüphanelerde (`cv2.Sobel`, `cv2.Canny`) varsayılan operatör olmasının da başlıca nedenidir.

## Dosyalar

- `PrewittSobelKarsilastirma.py`: üç deneyi çalıştırır, sayıları yazdırır ve görüntüleri kaydeder.
- `ciktilar/prewitt_sobel_karsilastirma.png`: büyüklük haritaları, fark görüntüsü, gürültülü sonuçlar ve kenar örtüşmesi.
- `ciktilar/yon_testi.png`: yön testinde kullanılan disk ve iki operatörün sonucu.
