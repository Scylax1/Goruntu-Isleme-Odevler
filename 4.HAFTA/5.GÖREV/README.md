# Laplacian, Sobel ve Prewitt ile Kenar Bulma: Karşılaştırma

## 1. Yöntem

Üç operatör de 3x3 boyutundadır. Her biri, önce 3x3 Gauss filtresiyle (`1/16 · [1 2 1; 2 4 2; 1 2 1]`) hafifçe yumuşatılmış resme uygulandı. Konvolüsyonlar döngüyle elle yazıldı. Sonuçlar `cv2.GaussianBlur`, `cv2.Laplacian` ve `cv2.filter2D` ile kontrol edildi ve hepsi birebir aynı çıktı.

| Operatör | Çekirdek | Ne ölçer | Kenarı nasıl bulur |
|---|---|---|---|
| **Prewitt** | `[-1 0 1; -1 0 1; -1 0 1]` ve devriği | 1. türev (gradyan) | \|G\| = √(Gx² + Gy²) > eşik |
| **Sobel** | `[-1 0 1; -2 0 2; -1 0 1]` ve devriği | 1. türev (gradyan) | \|G\| = √(Gx² + Gy²) > eşik |
| **Laplacian** | `[0 1 0; 1 -4 1; 0 1 0]` | 2. türev, ∂²f/∂x² + ∂²f/∂y² | işaret değişimi (sıfır geçişi) |

Bir kenarın birinci türevinde **tepe**, ikinci türevinde ise **sıfır geçişi** oluşur. Bu yüzden Sobel ve Prewitt'te kenar, eşiği aşan bir *bant* olarak çıkar. Laplacian'da kenar, pozitif ve negatif bölgelerin arasındaki *çizgidir*.

Laplacian için `LaplacianFiltresi.py` bir pikseli şu durumda kenar sayar: Laplacian değeri sağ ya da alt komşusuyla zıt işaretliyse ve aradaki fark 24'ten büyükse. Bu eşik şarttır. Eşiksiz bırakılınca dokulu bölgelerdeki küçük dalgalanmalar da sıfır geçişi ürettiğinden resmin %37'si (97 334 piksel) kenar çıkıyor. Sobel ve Prewitt için 4. görevdeki normalize eşik (\|G\| > 12) kullanıldı.

## 2. Sonuçlar

Ölçümler `LaplacianSobelPrewittKarsilastirma.py` ile alındı. Deneylerde 512x512 cameraman resmi ve yarıçapı 100 piksel, kontrastı 150 gri seviye olan yapay bir disk kullanıldı.

| Ölçüm | Prewitt | Sobel | Laplacian |
|---|---|---|---|
| Cameraman kenar pikseli | 22 936 | 23 301 | **10 324** |
| Disk: kenar kalınlığı | 3.72 px | 3.64 px | **0.87 px** |
| Disk: kenar piksellerinin gerçek kenara ortalama uzaklığı | 0.93 px | 0.91 px | **0.23 px** |
| Disk: sinyal / gürültü (σ = 10) | **44.0** | 42.8 | 8.1 |
| Disk: düz iç bölgede sahte kenar (σ = 10) | **0** | **0** | 65 |
| Cameraman: gürültüyle yeni çıkan sahte kenar (σ = 10) | **935** | 1 052 | 4 230 |

**Konum ve incelik: Laplacian açık ara önde.** Laplacian'ın kenarı tek piksel kalınlığında (0.87 px) ve gerçek kenara ortalama 0.23 piksel uzaklıkta. Sobel ve Prewitt ise kenarı 3.6–3.7 piksel kalınlığında bir bant olarak veriyor (`ciktilar/disk_kenar_detay.png`). Cameraman'de Laplacian'ın daha az kenar pikseli bulmasının (10 324'e karşı ~23 000) nedeni de budur: daha az kenar bulmuyor, aynı kenarları daha ince çiziyor. Sobel kenar piksellerinin %89.6'sı bir Laplacian çizgisine en fazla 2 piksel uzaklıkta. `ciktilar/kenar_kalinligi_detay.png` dosyasında Laplacian'ın kırmızı çizgileri, Sobel'in gri bantlarının ortasından geçiyor.

**Gürültü: Laplacian açık ara geride.** Aynı gürültüde Laplacian'ın sinyal/gürültü oranı Sobel'inkinin yaklaşık **beşte biri** (8.1'e karşı 42.8). Cameraman'de ise yaklaşık **dört kat** fazla sahte kenar üretiyor. Diskin tamamen düz iç bölgesinde Sobel ve Prewitt hiç sahte kenar bulmazken Laplacian 65 tane buluyor. Bunun iki nedeni var:

1. İkinci türev, gürültünün yüksek frekanslı bileşenlerini birinci türevden daha çok büyütür.
2. Sobel ve Prewitt türeve dik yönde 3 pikseli ortalayarak kendi içlerinde de yumuşatma yapar. Laplacian'da böyle bir ortalama yoktur.

Yumuşatma yapılmadığında fark daha da büyüyor. `LaplacianFiltresi.py` çıktısına göre ham resimde sıfır geçişi kenar sayısı Gauss sonrasındakinin 6.5 katı (66 620'ye karşı 10 324). Çimen bölgesi tamamen beyaza dönüyor (`ciktilar/laplacian_karsilastirma.png`, sol alt).

**Sobel ve Prewitt arasında fark küçük.** 4. görevde de görüldüğü gibi iki operatörün kenar haritaları neredeyse aynı. Prewitt gürültüye çok az daha dayanıklı; Sobel ise kenar yönünü daha doğru veriyor.

## 3. Diğer farklar

- **Yön bilgisi:** Sobel ve Prewitt, Gx ve Gy'den kenarın yönünü (`atan2(Gy, Gx)`) de verir. Laplacian tek bir sayı ürettiği için kenarın yönünü vermez.
- **Kenar gücü:** Sıfır geçişi yalnızca "kenar var mı" sorusunu cevaplar. Kenarın ne kadar güçlü olduğu, iki taraftaki Laplacian farkından dolaylı olarak anlaşılır. \|G\| ise kenar gücünü doğrudan verir.
- **Süreklilik:** Laplacian çizgileri tek piksel inceliğinde olduğu için çapraz kenarlarda basamak geçişlerinde yer yer kopukluklar oluşuyor. Sobel ve Prewitt'in kalın bantları sürekli kalıyor.
- **İşlem maliyeti:** Laplacian tek bir 3x3 konvolüsyondur. Sobel ve Prewitt ise iki konvolüsyon ile bir karekök gerektirir.

## 4. Değerlendirme

- **Laplacian**, kenarın tam yerini ince bir çizgi olarak bulmakta en iyisi. Ancak gürültüye çok duyarlı. Ön yumuşatma ve sıfır geçişine eşik olmadan pratikte kullanılamıyor. Gauss ile birlikte kullanıldığında (LoG, Marr–Hildreth) ve daha büyük bir Gauss seçildiğinde bu sorun azalır.
- **Sobel ve Prewitt** gürültüye karşı çok daha sağlam. Ayrıca kenarın yönünü ve gücünü de verirler. Buna karşılık kenarı kalın bir bant olarak bulurlar. İnce kenar gerektiğinde Canny'deki gibi bir maksimum olmayanı bastırma adımı gerekir.
- Gürültülü gerçek resimlerde genel amaçlı kenar bulma için **Sobel** daha güvenli bir seçim. **Laplacian** ise az gürültülü resimlerde hassas konum gerektiğinde ya da keskinleştirme (resimden Laplacian'ı çıkarmak) gibi işlerde öne çıkıyor.

## Dosyalar

- `LaplacianFiltresi.py`: Gauss ile yumuşatma, 3x3 Laplacian ve sıfır geçişiyle kenar bulma; sonucu pencerede gösterir.
- `LaplacianSobelPrewittKarsilastirma.py`: üç operatörü aynı deneylerle kıyaslar ve tablodaki sayıları üretir.
- `ciktilar/laplacian_karsilastirma.png`: Laplacian'ın ara adımları, yumuşatmalı ve yumuşatmasız sonuçlar.
- `ciktilar/uc_operator_karsilastirma.png`: üç operatörün temiz ve gürültülü resimde bulduğu kenarlar.
- `ciktilar/kenar_kalinligi_detay.png`, `ciktilar/disk_kenar_detay.png`: kenar kalınlığı farkının yakın görüntüleri.
