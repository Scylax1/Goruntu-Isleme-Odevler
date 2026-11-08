# Vektörü Sinüs Değerleriyle Dolduran Fonksiyon

`SinusSinyal.py`

```python
sinus_doldur(vektor, genlik, faz, frekans=None, periyot=None, ornekleme_frekansi=None)
```

Verilen vektörün her elemanını yerinde doldurur:

**vektor[n] = genlik · sin(2π · frekans · n / fs + faz)**

| Parametre | Anlamı |
|---|---|
| `genlik` | Tepe değeri |
| `faz` | Radyan cinsinden; n = 0'daki açı |
| `frekans` / `periyot` | İkisinden biri verilir; periyot = 1 / frekans (saniye) |
| `ornekleme_frekansi` | fs; verilmezse fs = N alınır, yani vektör tam 1 saniyelik sinyali tutar. Böylece frekans "vektör boyunca kaç tam devir" anlamına gelir ve DFT'de k. harmonik k Hz olur (4-6. görevler bunu kullanıyor). |

- **2'nin kuvveti kontrolü:** `n > 0 and n & (n - 1) == 0`. 2'nin kuvvetinin ikili gösteriminde tek bir 1 biti vardır; `n - 1` o bitin altını 1'lerle doldurur ve VE işleminin sonucu 0 olur. `N = 100` ile çağrı `ValueError` veriyor.
- **Örtüşme uyarısı:** Frekans fs/2'yi (Nyquist) geçerse fonksiyon uyarı yazıyor.
- **Doğrulama:** Sonuç numpy ile hesaplanan aynı formülle birebir aynı (fark 0).

## Görselleştirme (`ciktilar/sinus_ornekleri.png`)

Her panelde vektörün elemanları sap (stem) olarak, gerçek sürekli sinüs ince gri eğri olarak çizildi:

1. N = 64, A = 1, faz = 0, f = 2 Hz
2. N = 64, A = 3, faz = π/2, **periyot** = 0.25 s (f = 4 Hz); n = 0'da tepe değerinden başlıyor
3. N = 128, A = 2, faz = −π/4, f = 5 Hz; negatif değerden başlıyor
4. N = 512, A = 12, f = 50 Hz; 5. görevdeki sinyal, ilk 64 örnek
5. N = 64, f = 6 Hz, fs = 32 Hz; devir başına ~5 örnek, iyi örneklenmiş
6. **Örtüşme:** f = 30 Hz, fs = 32 Hz. Örnekler 30 Hz'lik eğriyi değil, kırmızı kesikli çizgiyle gösterilen −2 Hz'lik yavaş bir sinüsü izliyor (30 − 32 = −2). Örneklerden bu iki sinüs ayırt edilemez.
