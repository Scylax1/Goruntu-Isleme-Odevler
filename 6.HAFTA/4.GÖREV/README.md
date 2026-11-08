# Hough Dönüşümü ile Çember Bulma (elle yazılmış, 3 boyutlu oylama)

`HoughCemberElle.py`, `cv2.HoughCircles` kullanmadan çember Hough dönüşümünü baştan yazar. Girdi, 3. görevdeki test resmi: 6 daire ve tuzak olarak bir kare, gürültülü. Kenarlar medyan → Gauss → Canny ile bulunuyor (1 890 kenar pikseli).

## Yöntem

Merkezi (a, b), yarıçapı r olan çember: **a = x − r·cos θ, b = y − r·sin θ**.

1. **sin / cos tablosu.** θ = 0 … 359° için cos ve sin bir kez hesaplanıyor. Bunlardan her yarıçap için tamsayı ofsetler (dx, dy) = round(r·cos θ, r·sin θ) de bir kez hesaplanıyor. Küçük r'de birçok açı aynı piksele düşüyor; tekrarlar atılıyor (r = 15'te 360 açı → 104 nokta, r = 65'te → 355 nokta). Böylece bir piksel aynı hücreye iki kez oy veremiyor.
2. **3 boyutlu oylama matrisi** `H[b, a, r]`, boyutu 300 × 400 × 51 (r = 15 … 65).
   ```python
   for x, y in kenar_pikselleri:          # açık döngü
       for ri, r in enumerate(yaricaplar):  # açık döngü
           a = x - DX[r]                    # r yarıçaplı çemberin bütün noktaları
           b = y - DY[r]
           H[b, a, ri] += 1                 # oy
   ```
   En içteki açı döngüsü numpy dizi işlemiyle yapılıyor. Tamamen Python döngüsüyle 1 890 × 51 × ~250 ≈ 24 milyon işlem dakikalar sürerdi; böyle 1.4 s sürüyor. Program bir pikselin oylarını adım adım da yazdırıyor:
   ```
   theta =  90: a = 87 - 15*+0.00 =   87, b = 49 - 15*+1.00 =   34  ->  H[34, 87, r=15] += 1
   ```
3. **Eşik.** Oy, o yarıçaptaki çember nokta sayısına bölünüyor: **oran = H / nokta_sayısı[r]**, yani "çemberin ne kadarı görülüyor". Ham oyla eşik koyulsaydı büyük çemberler hep kayırılırdı. Oranı eşiği (0.45) geçen hücreler aday oluyor. Adaylar büyükten küçüğe sıralanıyor ve daha önce kabul edilmiş bir çembere hem merkezce (< 10 px) hem yarıçapça (< 10 px) yakın olanlar atılıyor.

## Sonuç

27 aday hücre, yakınlar elenince **6 çember**; 6'sı da doğru:

| Bulunan merkez | r | Oy | Oran | Gerçek |
|---|---|---|---|---|
| (240, 215) | 56 | 224 | 0.68 | (240, 215), 55 |
| (220, 80) | 31 | 133 | 0.67 | (220, 80), 30 |
| (90, 90) | 41 | 205 | 0.66 | (90, 90), 40 |
| (100, 220) | 36 | 158 | 0.66 | (100, 220), 35 |
| (320, 110) | 51 | 201 | 0.65 | (320, 110), 50 |
| (355, 255) | 22 | 95 | 0.59 | (355, 255), 22 |

Merkezler tam, yarıçaplar 3. görevdeki gibi yaklaşık 1 piksel büyük. Kare çember olarak alınmıyor.

**Eşiğin etkisi:** 0.25 – 0.45 arası 6 çember; 0.60 → 5 (en küçük daire 0.59 ile kaçıyor); 0.75 → 0. Gerçek çemberler 0.59 – 0.68 bandında, gürültü ve kare ise 0.25'in altında kalıyor. Arada geniş ve güvenli bir aralık var.

## Oylama matrisini inceleme (`hough_cember_elle.png`)

- **Orta satır:** Bulunan yarıçaplarda `H[:, :, r]` kesitleri. Her kenar pikseli r yarıçaplı bir çember çiziyor. Doğru yarıçapta bu çemberler gerçek merkezde tek noktada kesişiyor (yeşil daire içindeki parlak nokta). Yanlış yarıçaplarda kesişim, merkez etrafında halka şeklinde dağılıyor.
- **Sol alt:** r ekseni boyunca en büyük oran. 6 merkez parlak noktalar olarak görünüyor.
- **Orta alt:** Tek bir kenar pikselinin oyları. r = 15, 25, … 65 için iç içe çemberler, yani (a, b, r) uzayındaki koninin kesitleri.
- **Sağ alt:** Bulunan her merkezde oranın r'ye göre değişimi. Her merkezin kolonunda doğru yarıçapta tek ve keskin bir tepe var.

## Çıktılar (`ciktilar/`)

- `bulunan_cemberler.png`: Bulunan çemberler orijinal resim üzerinde, oranlarıyla.
- `hough_cember_elle.png`: Bütün görünümler tek resimde.
- `oylama_izdusum.png`, `oylama_koni_tek_piksel.png`, `kenar.png`
