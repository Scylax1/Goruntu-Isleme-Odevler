# Frekans Uzayında Low-pass, High-pass, Band-pass ve Band-reject Filtreleri

`FrekansFiltreleri.py`: **G = idft2(dft2(resim) · H)**, 7. görevdeki `dft2` ve `idft2` ile. D, harmoniğin merkeze (DC'ye) uzaklığı.

| Filtre | H(u, v) | Parametre |
|---|---|---|
| Low-pass | exp(−D² / 2D0²) | D0 = 40 |
| High-pass | 1 − exp(−D² / 2D0²) | D0 = 20 |
| Band-pass | exp(−((D² − C²) / (D·W))²) | C = 60, W = 20 |
| Band-reject | 1 − band-pass | C = 60, W = 20 |
| İdeal low-pass (kıyas) | 1 (D ≤ 40), 0 (D > 40) | D0 = 40 |

**Neden Gauss biçimi?** Keskin kesimli ideal filtre, uzay uzayında sinc fonksiyonuyla konvolüsyona karşılık geliyor ve kenarların çevresinde halkalanma (ringing) yaratıyor. Gauss'un ters dönüşümü yine Gauss olduğu için halkalanma olmuyor (`yakin_plan_halkalanma.png`).

**Resimler:**
- Orijinal cameraman.
- Gauss gürültüsü σ = 25 ve tuz-biber %10 (8. görevdeki fonksiyonlar).
- **Periyodik gürültü:** 40·sin(2π(40y + 45x)/512) eklendi. Bu gürültünün harmonikleri merkezden D = √(40² + 45²) = 60.2 uzakta iki noktada toplanıyor; band-reject (C = 60) tam onu hedefliyor.

## Sonuç

Tablodaki ölçüler:
- **ort.:** Çıktının ortalama parlaklığı.
- **Z/orij:** ZNCC(çıktı, orijinal resim).
- **Z/temiz-f:** ZNCC(çıktı, aynı filtrenin temiz resimdeki çıktısı). Filtre çıktısının gürültüden ne kadar etkilendiğini gösteriyor; 1 hiç etkilenmemiş demek.
- **kenar:** Ortalama Sobel büyüklüğünün orijinale oranı.

| Resim | Filtre | ort. | Z/orij | Z/temiz-f | kenar |
|---|---|---|---|---|---|
| **Orijinal** | — | 129.1 | 1.000 | | 1.00 |
| | Low-pass | 129.1 | 0.983 | | 0.43 |
| | High-pass | 0.0 | 0.346 | | 0.99 |
| | Band-pass | 0.0 | 0.128 | | 0.44 |
| | Band-reject | 129.1 | 0.996 | | 1.01 |
| | İdeal low-pass | 129.1 | 0.975 | | 0.56 |
| **Gauss σ = 25** | — | 129.6 | 0.950 | | 2.51 |
| | Low-pass | 129.6 | **0.982** | 0.999 | 0.52 |
| | High-pass | 0.0 | 0.212 | 0.613 | 2.47 |
| | Band-pass | 0.0 | 0.115 | 0.898 | 0.58 |
| | Band-reject | 129.6 | 0.947 | 0.951 | 2.42 |
| **Tuz-biber %10** | — | 128.8 | 0.822 | | 3.46 |
| | Low-pass | 128.8 | **0.978** | 0.995 | 0.65 |
| | High-pass | 0.0 | 0.119 | 0.344 | 3.45 |
| | Band-pass | 0.0 | 0.089 | 0.690 | 0.78 |
| | Band-reject | 128.8 | 0.821 | 0.825 | 3.50 |
| **Periyodik** | — | 130.0 | 0.935 | | 2.96 |
| | Low-pass | 130.0 | 0.975 | 0.992 | 1.04 |
| | High-pass | 0.0 | 0.193 | 0.560 | 2.91 |
| | Band-pass | 0.0 | 0.030 | 0.230 | 2.66 |
| | Band-reject | 130.0 | **0.995** | **0.999** | 1.06 |

## Yorum

### Genel parlaklık
- **Low-pass ve band-reject parlaklığı koruyor.** H(0, 0) = 1, yani DC (ortalama) olduğu gibi geçiyor; ortalama her resimde değişmiyor (129).
- **High-pass ve band-pass parlaklığı tamamen siliyor.** H(0, 0) = 0, çıktının ortalaması 0 ve yarısı negatif. Görseller +128 kaydırılarak gösterildi; geniş düz alanlar (gökyüzü, palto) aynı griye dönüşüyor. Resmin "ne kadar parlak olduğu" bilgisi kayboluyor, yalnızca "nerede değiştiği" kalıyor. Bu yüzden ZNCC/orijinal düşük (0.35 ve 0.13).

### Kenar geçişleri
- **Low-pass kenarları yumuşatıyor:** Kenar enerjisi orijinalin 0.43'üne düşüyor. Kenarlar yüksek frekanslı bileşenlerden oluşuyor ve atılıyorlar.
- **İdeal low-pass ayrıca halkalanma yaratıyor:** Kameramanın başı ve kamera çevresinde dalgalı gölgeler var (yakın plan). Kenar enerjisinin daha yüksek (0.56) çıkması daha keskin olduğundan değil, bu sahte dalgalardan.
- **High-pass yalnızca kenarları bırakıyor** (kenar oranı 0.99): Konturlar, tripodun ayakları, saç ve çimen dokusu kalıyor; düz alanlar kayboluyor. Kenar bulma ve keskinleştirmenin temeli bu (resim + k·high-pass = unsharp masking).
- **Band-pass orta ölçekteki yapıyı seçiyor:** Ne geniş alanlar (düşük frekans) ne de ince doku ve gürültü (yüksek frekans) kalıyor. Kenarların yalnızca belli bir kalınlıktaki bileşeni kalıyor ve sonuç dalgalı kontur çizgileri gibi görünüyor.
- **Band-reject orijinale en yakın sonuç** (0.996): 60 ± 10 harmonik bandı resmin enerjisinin küçük bir kısmını taşıyor.

### Gürültü
- **Gauss ve tuz-biber (beyaz gürültü): low-pass en iyisi.** ZNCC 0.950 → 0.982 ve 0.822 → 0.978. Gürültü bütün frekanslara eşit yayıldığı, resim ise düşük frekanslarda toplandığı için yüksek frekansları atmak gürültünün çoğunu atıyor. Bedeli bulanıklık.
- **High-pass gürültüyü büyütüyor.** Beyaz gürültünün enerjisinin büyük kısmı yüksek frekanslarda, H de orada 1. Çıktı temiz resmin high-pass çıktısına yalnızca 0.61 (Gauss) ve 0.34 (tuz-biber) benzeyebiliyor. Kenar bulmadan önce yumuşatma yapılmasının sebebi bu.
- **Band-pass yüksek frekansları attığı için gürültüyü high-pass'ten daha az geçiriyor** (0.90 / 0.69).
- **Band-reject beyaz gürültüye neredeyse hiç etki etmiyor** (0.950 → 0.947): Bandın dışındaki gürültü olduğu gibi geçiyor.
- **Periyodik gürültüde durum tersine dönüyor; band-reject en iyisi.** Bu gürültünün bütün enerjisi D = 60'taki iki harmonikte (`filtre_transfer_fonksiyonlari.png`, sağ üst). Band-reject o halkayı söndürüyor:
  - ZNCC 0.935'ten **0.995**'e çıkıyor, temiz resmin band-reject çıktısıyla 0.999 benzerlik var.
  - Kenar enerjisi neredeyse aynı kalıyor (1.06), parlaklık değişmiyor. Çizgiler tamamen kayboluyor, resim keskin kalıyor.
  - Low-pass (D0 = 40) aynı gürültüyü H(60) = 0.32 ile yalnızca zayıflatıyor; çizgiler soluk da olsa kalıyor ve resim bulanıklaşıyor.
- **Band-pass periyodik gürültüyü tam merkezinden geçiriyor:** Çıktı neredeyse yalnızca çizgilerden oluşuyor (0.23). Bu, gürültüyü ayıklamak (örneğin sonra çıkarmak) için kullanılabilir.

### Özet
Gürültü belli frekanslarda toplanmışsa (periyodik) o frekansları hedefleyen band-reject ya da notch filtre; bütün frekanslara yayılmışsa (beyaz) low-pass kullanılmalı. Beyaz gürültüde low-pass'in bedeli kenarların yumuşaması. Tuz-biberde 9. görevde görüldüğü gibi uzamsal medyan, frekans uzayındaki her doğrusal filtreden daha iyi.

## Çıktılar (`ciktilar/`)

- `frekans_filtreleri.png`: 4 resim × 5 filtre.
- `filtre_transfer_fonksiyonlari.png`: H(u, v) ve kesitleri (kırmızı çizgi periyodik gürültünün frekansı); periyodik gürültülü resmin harmonik matrisi band-reject öncesi ve sonrası.
- `yakin_plan_halkalanma.png`: Gauss ve ideal low-pass karşılaştırması, high-pass.
