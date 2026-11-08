# Ters DFT ile Orijinale Dönme ve Zero Mean Normalized Cross Correlation

`TersDFT_ZNCC.py`: 8. görevdeki gürültülü resimler (Gauss σ = 25, tuz-biber %10) üzerinde.

## Yöntem

**A) Doğrudan ters DFT:** `idft2(dft2(gürültülü))`. DFT kayıpsız ve tersinir bir dönüşüm. Harmonik matrisinde hiçbir şey değiştirilmezse ters DFT gürültülü resmin **aynısını** veriyor (fark 6·10⁻¹²) ve gürültü kendiliğinden gitmiyor. Orijinale dönmek için harmonik matrisinde gürültüyü ayırmak gerekiyor.

**B) Frekans uzayında Gauss alçak geçiren + ters DFT:**
- F·H hesaplanıyor; H = exp(−D²/2D0²).
- 8. görevde görüldüğü gibi resmin enerjisi düşük frekanslarda, gürültününki ise her yerde düz. Yüksek frekansları bastırmak, gürültünün çoğunu resmin küçük bir kısmı pahasına atıyor.
- D0 tarandı ve ZNCC'yi en büyük yapan değer seçildi.

**ZNCC:**

```
ZNCC = Σ(a − ā)(b − b̄) / √( Σ(a − ā)² · Σ(b − b̄)² )        −1 ≤ ZNCC ≤ 1
```

## Sonuç

**Gauss gürültüsü, σ = 25**

| | ZNCC | NCC | SSIM | PSNR (dB) | MSE | MAE |
|---|---|---|---|---|---|---|
| Gürültülü | 0.9500 | 0.9874 | 0.288 | 20.60 | 565.8 | 18.97 |
| A) idft2(dft2(gürültülü)) | 0.9500 | 0.9874 | 0.288 | 20.60 | 565.8 | 18.97 |
| **B) LP D0 = 80 + idft2** | **0.9883** | 0.9971 | 0.654 | 27.03 | 128.7 | 8.14 |
| (uzamsal medyan 3x3, kıyas) | 0.9844 | 0.9962 | 0.507 | 25.86 | 168.8 | 9.86 |

**Tuz-biber, %10**

| | ZNCC | NCC | SSIM | PSNR (dB) | MSE | MAE |
|---|---|---|---|---|---|---|
| Gürültülü | 0.8218 | 0.9523 | 0.185 | 14.77 | 2166.4 | 12.74 |
| A) idft2(dft2(gürültülü)) | 0.8218 | 0.9523 | 0.185 | 14.77 | 2166.4 | 12.74 |
| **B) LP D0 = 50 + idft2** | **0.9794** | 0.9941 | 0.570 | 23.76 | 273.5 | 12.10 |
| (uzamsal medyan 3x3, kıyas) | 0.9933 | 0.9984 | 0.850 | 29.56 | 71.9 | 3.72 |

**Yorum:**
- Frekans uzayında alçak geçiren, Gauss gürültüsünde ZNCC'yi 0.950'den **0.988**'e çıkarıyor ve uzamsal medyandan bile iyi.
- D0 küçüldükçe resim bulanıklaşıyor, büyüdükçe gürültü geri geliyor; aradaki en iyi değer 80.
- Tuz-biberde ZNCC 0.822'den 0.979'a çıkıyor, ama uzamsal medyan (0.993) açık ara daha iyi. Alçak geçiren her dürtüyü yok etmiyor, çevresine yayıyor. Bu yüzden MAE neredeyse hiç düşmüyor (12.7 → 12.1). Medyan ise dürtüleri doğrudan atıyor. 3. görevdeki 1D sonuçla aynı ders: dürtü gürültüsü doğrusal filtrelerle (frekans uzayındaki çarpma da doğrusal bir filtredir) iyi temizlenemez.

## İki piksel değerinin (iki resmin) yakınlığını ölçen metrikler

Orijinal resmin bilinen bozulmalarına metriklerin tepkisi:

| Bozulma | ZNCC | NCC | SSIM | PSNR (dB) | MSE | MAE |
|---|---|---|---|---|---|---|
| Aynı resim | 1.0000 | 1.0000 | 1.000 | ∞ | 0 | 0 |
| Parlaklık +30 | 0.9996 | 0.9962 | 0.904 | 18.62 | 892.8 | 29.81 |
| Kontrast ×0.5 (+64) | **1.0000** | 0.9715 | 0.792 | 16.81 | 1356.2 | 32.39 |
| 1 piksel kaydırma | 0.9768 | 0.9943 | 0.753 | 24.12 | 251.6 | 7.09 |
| Gauss bulanık σ = 2 | 0.9846 | 0.9962 | 0.750 | 25.91 | 166.9 | 6.72 |
| Negatif (255 − f) | **−1.0000** | 0.4996 | −0.090 | 4.77 | 21704 | 129.84 |
| Gauss gürültüsü σ = 25 | 0.9500 | 0.9874 | 0.288 | 20.60 | 565.8 | 18.97 |

| Metrik | Tanım | Avantajları | Dezavantajları |
|---|---|---|---|
| **SAD** (mutlak farklar toplamı) | Σ\|a − b\| | En basit ve hızlı; şablon eşlemede ve blok eşlemede (hareket kestirimi) yaygın | Resim boyutuna bağlı, karşılaştırılamaz; parlaklık ve kontrast değişimine çok duyarlı |
| **MAE** | SAD / piksel sayısı | Boyuttan bağımsız, gri seviye biriminde okunur; aykırı değerlere MSE'den az duyarlı | Parlaklık/kontrast değişimine duyarlı; algıyla zayıf ilişkili |
| **MSE / RMSE** | ortalama (a − b)² / karekökü | Türevlenebilir (optimizasyonda kayıp fonksiyonu); istatistiksel olarak anlamlı (Gauss gürültüsünde en uygun kestirici) | Büyük hataları kareyle büyütür; tuz-biberde MSE 2166, MAE yalnızca 12.7. Algıyla zayıf ilişkili: parlaklık +30 (gözle neredeyse aynı resim), MSE'ye göre σ = 25 gürültüden daha kötü (893'e karşı 566) |
| **PSNR** | 10 log₁₀(255² / MSE) | Desibel ölçeği; sıkıştırma ve gürültü giderme literatüründe standart | MSE'nin bütün sorunlarını taşır; farklı içerikli resimler arasında karşılaştırılamaz |
| **NCC** (ortalama çıkarılmadan) | Σab / √(Σa² Σb²) | Kontrast (çarpımsal) değişimine duyarsız; [0, 1]'de sınırlı | Parlaklık kaymasına duyarlı; negatif bile 0.50 çıkıyor. Değerler 1'e çok yığılıyor (gürültülü resim 0.987), ayırt etme gücü düşük |
| **ZNCC** | Ortalaması çıkarılmış NCC (= Pearson korelasyonu) | Parlaklık **ve** kontrast değişimine (a·f + b) tamamen duyarsız (kontrast ×0.5 + 64 → 1.0000); [−1, 1]'de sınırlı ve yorumlanabilir; ters ilişkiyi yakalar (negatif → −1). Farklı ışıkta çekilmiş resimlerde şablon eşleme ve stereo eşleme için en uygunu | Duyarsızlığı istenmediğinde dezavantaj: parlaklığı tamamen bozulmuş bir sonuç da "mükemmel" görünür. Düz (varyansı 0) bölgede tanımsız (0/0). SAD'dan pahalı. Yapısal bozulmayı (bulanıklık) zayıf cezalandırır (0.985) |
| **SSIM** | Yerel parlaklık × kontrast × yapı benzerliği, kayan pencerede | İnsan algısına en yakın; bulanıklığı ve gürültüyü güçlü cezalandırıyor (gürültüde 0.29, bulanıklıkta 0.75); yerel olduğu için hatanın **nerede** olduğunu bir haritada gösterebiliyor | Hesaplaması pahalı; pencere boyutu ve sabitler (C1, C2) gibi parametrelere bağlı; değer aralığı sezgisel değil; kaydırmaya çok duyarlı (1 px → 0.75) |

**Hangisi ne zaman?**
- İki resmin piksel piksel **aynı** olup olmadığına bakılıyorsa (sıkıştırma, yeniden üretim): MSE/PSNR.
- Işık koşulları farklıysa ve aranan **desen benzerliği** ise: ZNCC. Bu görevde de ZNCC kullanıldı; gürültü temizlemenin parlaklık ortalamasını (DC) değiştirmemesi zaten beklenen bir şey.
- **Algısal kalite** önemliyse: SSIM.
