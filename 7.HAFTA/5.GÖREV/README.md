# İki Gürültülü Sinyalin ve Toplamlarının DFT Analizi

`IkiSinyalDFT.py`: 1, 2 ve 4. görevlerdeki fonksiyonları (`sinus_doldur`, `gurultu_ekle`, `dft`) kullanıyor.

**Sinyaller:**
- **Sinyal 1:** f = 50 Hz, A = 12, gürültü oranı 0.3 (σ ≈ 3.6)
- **Sinyal 2:** f = 30 Hz, A = 5, gürültü oranı 0.3 (σ ≈ 1.5)
- **Toplam:** sinyal 1 + sinyal 2. İki bağımsız gürültü de toplanıyor: σ = √(3.6² + 1.5²) ≈ 3.9.

N = 512, fs = 512 Hz (1 saniye). İki frekans da tam sayı olduğu için harmonikler tam k = 30 ve k = 50'ye düşüyor, sızıntı yok.

## Sonuç

| Sinyal | En büyük harmonikler |
|---|---|
| Sinyal 1 | **k = 50: 12.44**, k = 121: 0.86, k = 235: 0.75 |
| Sinyal 2 | **k = 30: 4.92**, k = 203: 0.31, k = 254: 0.29 |
| Toplam | **k = 50: 12.35, k = 30: 4.65**, k = 153: 0.82 |

- Toplamın gürültü tabanı: ortalama 0.32, en büyük 0.82.
- Doğrusallık: |DFT(x1 + x2) − (DFT(x1) + DFT(x2))| = 3·10⁻¹².

## Yorum

- **Zaman uzayında toplam sinyali çözmek zor; frekans uzayında kolay.** Zaman grafiğinde 30 ve 50 Hz üst üste binip tepeleri değişen, gürültülü bir dalga oluşturuyor. Harmonik vektöründe ise iki ayrı ve net çizgi var: k = 30'da ~5, k = 50'de ~12. Genlikleri de doğrudan okunabiliyor.
- **DFT doğrusal:** Toplamın harmonik vektörü, iki sinyalin harmonik vektörlerinin toplamı. Bu yüzden her bileşen kendi harmoniğinde kalıyor ve birbirini etkilemiyor.
- **Toplamdaki tepeler tek tek ölçülenlerden biraz farklı:** k = 50'de 12.44 yerine 12.35, k = 30'da 4.92 yerine 4.65. Sebep, sinyal 2'nin gürültüsünün k = 50'ye, sinyal 1'in gürültüsünün de k = 30'a düşen payı. Bunlar karmaşık sayı olarak toplanıyor ve genliği biraz artırabiliyor ya da azaltabiliyor.
- **Zayıf bileşen de tabanın çok üstünde.** 5 genlikli 30 Hz bileşeni bile gürültü tabanının en büyük değerinin (0.82) 5 katından fazla. 6. görevdeki eşik filtresinin iki bileşeni de koruyabilmesi buna bağlı.

`ciktilar/iki_sinyal_dft.png`: Solda her sinyalin ilk 0.25 saniyesi (gürültülü ve gürültüsüz), sağda harmonik vektörleri.
