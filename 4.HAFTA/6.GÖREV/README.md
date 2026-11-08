# 7x7 Laplacian of Gaussian (LoG) ile Kenar Bulma ve Diğer Filtrelerle Karşılaştırma

## 1. LoG filtresi

LoG, Gauss fonksiyonunun Laplacian'ıdır (Marr–Hildreth kenar bulucu):

```
LoG(x, y) = (x² + y² − 2σ²) / σ⁴ · exp(−(x² + y²) / 2σ²)
```

Konvolüsyon birleşmeli bir işlemdir: `∇²(G * f) = (∇²G) * f`. Bu yüzden "önce Gauss ile yumuşat, sonra Laplacian uygula" işlemi tek bir çekirdekle, tek geçişte yapılabilir. LoG'a ayrıca Gauss uygulanmaz, yumuşatma çekirdeğin içindedir.

**Çekirdek.** `LoGFiltresi.py` çekirdeği formülden üretiyor. Boyut 7x7, yani yarıçap 3. Gauss'un ~%99.7'si ±3σ içinde kaldığı için **σ = 1.0** seçildi. 7x7'ye kesilen çekirdeği iki adımda düzelttim:
1. Ortalamasını çıkardım. Böylece toplam tam 0 oldu ve düz bölgede yanıt 0.
2. `f = x²` resmine yanıtı 2 (gerçek Laplacian değeri) olacak şekilde ölçekledim. Böylece çıktı, 3x3 Laplacian ile aynı birimde.

```
 0.000  0.003  0.009  0.013  0.009  0.003  0.000
 0.003  0.018  0.040  0.044  0.040  0.018  0.003
 0.009  0.040  0.000 -0.098  0.000  0.040  0.009
 0.013  0.044 -0.098 -0.324 -0.098  0.044  0.013      ("Meksika şapkası":
 0.009  0.040  0.000 -0.098  0.000  0.040  0.009       negatif merkez,
 0.003  0.018  0.040  0.044  0.040  0.018  0.003       pozitif halka)
 0.000  0.003  0.009  0.013  0.009  0.003  0.000
```

**Kenar bulma.** Kenar, LoG'un işaret değiştirdiği yerdedir (sıfır geçişi). Bir pikselin sağ ya da alt komşusuyla işareti farklıysa ve aradaki sıçrama eşikten (11) büyükse, o piksel kenar sayılır. Eşiksiz bırakılınca resmin %41'i (106 970 piksel) kenar çıkıyor. Eşikle bu sayı 17 822'ye iniyor (`ciktilar/log_karsilastirma.png`).

Konvolüsyon döngüyle elle yazıldı. Sonuç `cv2.filter2D` ile kontrol edildi; en büyük fark 9·10⁻¹⁴, yani sayısal yuvarlama düzeyinde.

## 2. Deney düzeni

`LoGKarsilastirma.py` yedi kenar bulucuyu aynı deneylerle kıyaslıyor:

- **Prewitt** ve **Sobel**: birinci türev. \|G\| = √(Gx² + Gy²) eşiklenir.
- **Laplacian** (3x3, `[0 1 0; 1 -4 1; 0 1 0]`) ve **LoG**: ikinci türev. Sıfır geçişleri eşiklenir.

Prewitt, Sobel ve Laplacian hem doğrudan ham resme, hem de önce 3x3 Gauss ile yumuşatılmış resme uygulandı. LoG yalnızca ham resme uygulandı, çünkü yumuşatma zaten içinde.

**Adil eşik.** Operatörlerin çıktıları farklı birimlerde olduğu için hepsine aynı eşik sayısı verilemez. Bu yüzden her operatörün eşiği aynı kuralla belirlendi: 150 gri seviyelik temiz bir disk kenarındaki kendi tepe değerinin **%20'si**. Burada tepe değer, gradyan operatörlerinde \|G\|, sıfır geçişi operatörlerinde ise iki komşu arasındaki sıçramadır. Böylece hepsi aynı kontrastlı bir kenarı aynı oranda "zayıf" sayıyor. Bu kural önceki görevlerdeki sabit eşiklerden farklı olduğu için sayılar 4. ve 5. görevle birebir aynı değil.

**Test resimleri.** 512x512 cameraman resmi ve yarıçapı 100 piksel olan yapay bir disk. Gürültü testinde iki resme de σ = 10 Gauss gürültüsü eklendi.

## 3. Sonuçlar

| Operatör | Kenar kalınlığı | Gerçek kenara uzaklık | Sinyal / gürültü | Disk içinde sahte kenar | Cameraman'de yeni sahte kenar |
|---|---|---|---|---|---|
| Prewitt | 2.79 px | 0.70 px | 28.9 | 24 | 3 500 |
| Prewitt + Gauss | 3.78 px | 0.95 px | **44.0** | **0** | **1 175** |
| Sobel | 2.66 px | 0.66 px | 28.7 | 33 | 3 581 |
| Sobel + Gauss | 3.69 px | 0.92 px | 42.8 | **0** | 1 232 |
| Laplacian | **0.84 px** | 0.24 px | 5.0 | 9 488 | 80 144 |
| Laplacian + Gauss | 0.87 px | 0.23 px | 8.1 | 2 118 | 21 038 |
| **LoG 7x7** | 0.90 px | **0.22 px** | 8.6 | 592 | 9 324 |

*Kenar kalınlığı = disk kenarında bulunan kenar pikseli / çevre. Sinyal/gürültü = kenardaki tepe yanıt / düz bölgedeki gürültünün standart sapması. "Yeni sahte kenar" = gürültülü resimde bulunan ve temiz resimdeki hiçbir kenara 1 pikselden yakın olmayan kenar pikseli.*

**Gauss ön yumuşatmasının etkisi.** Ön yumuşatma her operatörü iyileştiriyor, ama ikinci türev operatörlerini çok daha fazla:

- **Sobel ve Prewitt:** Sinyal/gürültü 29'dan 43–44'e çıkıyor, sahte kenar 3 katı azalıyor. Bedeli, kenar bandının 2.7 pikselden 3.7 piksele kalınlaşması.
- **Laplacian:** Yumuşatma olmadan pratikte kullanılamıyor. Gürültülü cameraman'de 80 144 sahte kenar pikseli çıkıyor; bu, resmin yaklaşık üçte biri (`ciktilar/tum_operatorler_gurultulu.png`). Gauss bu sayıyı 4 kat azaltıyor.

**LoG ve "Laplacian + Gauss" karşılaştırması.** İkisi aynı fikrin iki farklı gerçeklemesi. Sinyal/gürültü oranları yakın (8.6'ya karşı 8.1), çünkü yumuşatma ölçekleri de yakın: 3x3 Gauss σ ≈ 0.85, LoG σ = 1.0. Yine de LoG belirgin şekilde daha az sahte kenar üretiyor: diskte **3.6 kat** (592'ye karşı 2 118), cameraman'de **2.3 kat** (9 324'e karşı 21 038). Bunun nedeni, sıfır geçişi kararının *komşu pikseller arasındaki sıçramaya* bakmasıdır:

- 3x3 Laplacian keskin bir çekirdek. Gürültüyü komşu pikseller arasında hızlı değişen bir desene dönüştürüyor ve bu desen büyük sıçramalar üretiyor.
- LoG'un 7x7'lik pürüzsüz çekirdeği ise gürültüyü daha geniş, yavaş değişen lekelere dönüştürüyor. Komşu pikseller arasındaki sıçramalar küçük kalıyor ve eşiği geçemiyor. Gerçek bir kenardaki sıçrama ise korunuyor.

Konum doğruluğu ve incelik ise korunuyor: LoG'un kenarı tek piksel kalınlığında ve gerçek kenara en yakın (0.22 px).

**LoG ve birinci türev operatörleri karşılaştırması.** LoG, ikinci türev operatörleri arasında en iyisi. Yine de gürültüye karşı Gauss + Sobel/Prewitt'in gerisinde kalıyor:
- Sinyal/gürültü oranı yaklaşık 5 kat düşük.
- Cameraman'de yaklaşık 8 kat fazla sahte kenar üretiyor.
- Diskin düz iç bölgesinde, Gauss'lu Sobel ve Prewitt hiç sahte kenar bulmazken LoG 592 tane buluyor.

Buna karşılık kenarı Sobel'in dörtte biri inceliğinde ve dört kat daha doğru konumda buluyor (`ciktilar/detay_karsilastirma.png`).

## 4. Değerlendirme

1. **Ön yumuşatma her kenar bulucu için faydalı.** İkinci türev operatörlerinde ise vazgeçilmez.
2. **LoG, Laplacian tabanlı kenar bulmanın doğru yolu.** Gauss ve Laplacian'ı tek çekirdekte birleştirdiği için iki ayrı 3x3 adımdan daha pürüzsüz bir yanıt veriyor. Aynı incelik ve doğrulukta çok daha az sahte kenar üretiyor.
3. **7x7 boyutu σ'yı sınırlıyor.** Çekirdek ±3σ'yı kapsamalı; bu yüzden 7x7 çekirdekte σ en fazla ~1 olabiliyor. Daha gürültülü resimlerde daha büyük bir σ (örneğin σ = 2 için 13x13 çekirdek) gürültüyü çok daha iyi bastırır, ama ince ayrıntıları da siler.
4. **Uygulama seçimi:**
   - *Gürültülü resimlerde ve yön bilgisi gerektiğinde* Gauss + Sobel daha güvenilir.
   - *Kenarın tam yeri ve tek piksel inceliği gerektiğinde* LoG daha uygun. Örnekler: ölçüm, nesne sınırının konumu, kapalı kontur çıkarma.
   - Sobel'in kalın bandını Canny gibi bir inceltme adımıyla birleştirmek, iki yaklaşımın avantajlarını bir araya getirir.

## Dosyalar

- `LoGFiltresi.py`: 7x7 LoG çekirdeğini üretir, elle konvolüsyon yapar, sıfır geçişleriyle kenar bulur ve sonucu pencerede gösterir.
- `LoGKarsilastirma.py`: yedi kenar bulucuyu aynı deneylerle kıyaslar ve tablodaki sayıları üretir.
- `ciktilar/log_karsilastirma.png`: çekirdek, LoG yanıtı, eşiksiz ve eşikli sıfır geçişleri.
- `ciktilar/tum_operatorler.png`, `ciktilar/tum_operatorler_gurultulu.png`: yedi operatörün temiz ve gürültülü resimde bulduğu kenarlar.
- `ciktilar/detay_karsilastirma.png`: tripod bölgesinde Sobel + Gauss, Laplacian, Laplacian + Gauss ve LoG yakın görüntüsü.
