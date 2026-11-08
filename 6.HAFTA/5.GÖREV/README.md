# Hough Dönüşümü ile Eş Merkezli Çember Bulma (elle yazılmış)

`HoughEsMerkezliCember.py`, aynı merkeze sahip çember gruplarını, oylamanın kendisini eş merkezliliğe göre kurarak bulur. Bunu yaparken çemberleri tek tek bulup merkezleri yakın olanları sonradan gruplamıyor.

## Fikir: radyal oy

4. görevde her kenar pikseli, r yarıçaplı çemberin **bütün 360 açısına** oy veriyordu. Bu durumda her halka kendi konisini oluşturuyor ve halkalar birbirine destek olmuyor. Eş merkezli grubu o yöntemle bulmak için çemberleri tek tek tespit edip merkezleri yakın olanları gruplamak gerekirdi. Bu bir son işlem olurdu.

Burada oylama, eş merkezliliğin geometrik tanımı üzerine kurulu:

> Bir çemberin her noktasındaki kenar normali (gradyan) merkezden geçer. Eş merkezli çemberlerin **hepsinin** normalleri **aynı** noktadan geçer.

Bu yüzden her kenar pikseli yalnızca **kendi gradyan doğrultusu üzerindeki** merkezlere oy veriyor (r başına 2 oy, gradyanın iki yanı). Hangi halkadan gelirse gelsin bütün oylar ortak merkezin `(a, b)` kolonunda toplanıyor, her halka da o kolonda kendi `r` katmanına düşüyor. Böylece:

| Yapı | Oylama matrisindeki izi |
|---|---|
| Eş merkezli grup | Aynı `(a, b)` kolonunda **birden fazla** r tepesi |
| Tek çember | Kolonda tek tepe |
| Merkezi kayık iç içe çemberler | Oylar **iki ayrı** kolona dağılıyor, her birinde tek tepe |

## Yöntem

1. **Kenar ve gradyan.** Kenarlar medyan → 3x3 Gauss → Canny ile bulunuyor. Gradyan yönü φ = atan2(Gy, Gx) ise σ = 2.5 Gauss ile yumuşatılmış resimden alınıyor ve 1 dereceye yuvarlanıyor. Radyal oyda açı hatası uzaklıkla büyüyor (3° hata r = 75'te oyu 4 piksel kaydırıyor). 3x3 Gauss'tan alınan gürültülü yönle en dış halkanın oranı 0.32'de kalıyordu, σ = 2.5 ile 0.65'e çıktı.
2. **sin / cos tablosu.** 0 … 359° için cos ve sin bir kez hesaplanıyor. Oylamada her piksel için açı indeksiyle tablodan okunuyor.
3. **3 boyutlu oylama matrisi** `H[b, a, r]`, boyutu 320 × 520 × 71 (r = 10 … 80). Üç döngü de açık Python döngüsü:
   ```python
   for x, y in kenar_pikselleri:
       k = aci[y, x]
       for ri, r in enumerate(yaricaplar):
           for yon in (1, -1):
               a = round(x - yon * r * COS[k])
               b = round(y - yon * r * SIN[k])
               H[b, a, ri] += 1
   ```
   3 592 kenar pikseli × 71 r × 2 yön ≈ 486 000 oy; 0.3 s sürüyor. Program bir pikselin oylarını adım adım yazdırıyor.
4. **Eşik.** Her r katmanında 3x3 komşuluk toplanıyor (açı yuvarlamasının 1-2 piksellik kaymasını tolere etmek için) ve 2πr'ye bölünüyor. Bir kolonda oranı **0.40**'ı geçen ve r ekseninde ±3 içinde yerel maksimum olan her r bir halka sayılıyor. Kolonunda **en az 2 halka** olan merkez, eş merkezli grup kabul ediliyor.

## Test resmi ve sonuç

520x320 boyutunda yapay resim. Gauss ve tuz-biber gürültüsü eklendi, içinde gerçek gruplar ve tuzaklar var:

| İçerik | Beklenen | Bulunan |
|---|---|---|
| (110, 140): r = 75, 55, 35, 15 | 4 halkalı grup | **4 halka**: r = 15 (1.20), 35 (1.13), 55 (0.93), 75 (0.65) ✔ |
| (300, 200): r = 60, 40, 20; dış halka dikdörtgenle kısmen örtülü | 3 halkalı grup | **3 halka**: r = 20 (1.20), 40 (1.02), 61 (0.55) ✔ |
| Tek daire (300, 60) r = 35 | Grup değil | 1 halka → reddedildi ✔ |
| Tek daire (450, 260) r = 28 | Grup değil | 1 halka → reddedildi ✔ |
| Eksantrik çift: (440, 100) r = 45 ve (452, 108) r = 20 | Grup değil | İki ayrı merkez, her birinde 1 halka → reddedildi ✔ |
| Kare | — | Hiçbir merkez oluşturmadı ✔ |

Oran 1'i biraz geçebiliyor, çünkü 3x3 toplamda komşu katmanlardan taşan oylar da sayılıyor. Dış halkaların oranı düşük çıkıyor. Bunun iki sebebi var: kalan açı hatası ve örtülen kısım (2. grubun r = 60 halkasının ~%25'i dikdörtgenin altında, 0.55).

**cv2.HoughCircles ile karşılaştırma.** Aynı kenar resminde 16 çember döndürüyor ama eş merkezli halkaların çoğunu bulamıyor. Her merkez için tek yarıçap seçiyor ve `minDist` ile yakın merkezleri bastırıyor. Bu yüzden 4 halkalı gruptan yalnızca r = 56'yı, 3 halkalı gruptan yalnızca r = 40'ı buluyor. Geri kalanlar, halkaların çevresine dağılmış sahte çemberler (`opencv_houghcircles.png`).

## Oylama matrisini inceleme (`es_merkezli_karsilastirma.png`)

- **Tek pikselin oyları:** 4. görevdeki koni yerine yalnızca gradyan doğrultusunda iki ışın.
- **Σr H(a, b, r):** Bütün ışınlar gerçek merkezlerde kesişiyor. Eş merkezli grupların merkezleri, birden fazla halkadan oy aldığı için en parlak noktalar.
- **H[:, :, r = 55] kesiti:** 1. grubun 3. halkası. Yalnızca o yarıçaptaki merkezde keskin bir tepe var.
- **Kolon başına halka sayısı:** Merkezlerdeki sayılar 4, 3, 1, 1, 1, 1. Eş merkezli olanlar ≥ 2.
- **Merkez kolonları, oran − r grafiği:** Gruplar renkli (yeşil 4 tepe, mavi 3 tepe), tek çemberler gri (tek tepe).

## Çıktılar (`ciktilar/`)

- `es_merkezli_sonuc.png`: Gruplar kendi renginde, tek çemberler koyu gri.
- `es_merkezli_karsilastirma.png`: Bütün görünümler tek resimde.
- `oylama_toplam.png`, `halka_sayisi_haritasi.png`, `opencv_houghcircles.png`, `kenar.png`, `test_resmi.png`
