# DFT ile Frekans Uzayına Dönüşüm

`DFT.py`: Elle yazılmış DFT ve ters DFT.

```python
dft(vektor)   ->  harmonik vektörü X (karmaşık, uzunluk N)
idft(X)       ->  zaman uzayındaki sinyal
```

**X[k] = Σₙ x[n] · (cos(2πkn/N) − j sin(2πkn/N))**

- **sin / cos tablosu:** Açı her zaman 2π·((k·n) mod N)/N olduğu için yalnızca N tane cos ve N tane sin hesaplanıp tabloda tutuluyor. Döngü tablodan okuyor; N² kez cos/sin hesaplanmıyor.
- **Uzunluk:** 2'nin kuvveti olmalı (1. görevdeki kontrol).
- **Doğrulama:**
  - `np.fft.fft` ile en büyük fark 7.6·10⁻¹³.
  - `idft(dft(x))` ile x arasındaki fark 3.2·10⁻¹⁴.
  - Parseval eşitliği tutuyor: Σx² = Σ|X|²/N = 46 825.5.

**Harmonik vektörünü okumak:** `genlik_spektrumu(X)` tek taraflı genliği veriyor: 2|X[k]|/N (DC ve Nyquist'te |X[k]|/N). A genlikli bir sinüsün harmoniği tam A çıkıyor.

## Sonuç (f = 50 Hz, A = 12, N = 512, fs = 512)

| Sinyal | Tepe | Tepe genliği | Faz | Diğer harmonikler (ort. / en büyük) |
|---|---|---|---|---|
| Gürültüsüz | k = 50 | **12.000** | −90.0° | 0.000 / 0.000 |
| Gürültülü, oran 0.3 (σ = 3.6) | k = 50 | 12.444 | −90.7° | 0.292 / 0.856 |
| Gürültülü, oran 1.0 (σ = 12) | k = 50 | 13.486 | −92.1° | 0.972 / 2.852 |
| Gürültüsüz, f = 50.5 Hz | k = 50 | 7.672 | | 0.165 / **7.607** |

## Yorum

- **Gürültüsüz sinyal tek bir harmonik veriyor:** k = 50'de tam 12, geri kalan bütün harmonikler 0. Sinüs vektör boyunca tam 50 devir yaptığı için bütün enerjisi tek bir harmoniğe düşüyor. Faz −90°, çünkü sin(θ) = cos(θ − 90°) ve DFT kosinüse göre ölçüyor.
- **Gürültü bütün harmoniklere düz bir taban ekliyor.** Beyaz gürültünün her harmoniğe düşen beklenen genliği σ·√(π/N):

  | σ | Kuramsal taban | Ölçülen ortalama |
  |---|---|---|
  | 3.6 | 0.282 | 0.292 |
  | 12 | 0.940 | 0.972 |

  Log ölçekli grafikte (`dft_log.png`) bu taban yatay bir çizgi olarak görülüyor. Gürültünün belli bir frekansı yok; enerjisini bütün harmoniklere eşit dağıtıyor.
- **Sinüs gürültüden yine de kolayca ayrılıyor.** Zaman uzayında σ = 12 gürültü sinüsü neredeyse gizliyor (SNR −3 dB). Frekans uzayında ise sinüsün 12'lik tepesi, gürültünün en büyük harmoniğinin (2.85) 4 katından büyük. Gürültünün enerjisi 256 harmoniğe bölünüyor, sinüsünki tek harmonikte toplanıyor. 6. görevdeki filtre bu farka dayanıyor.
- **Gürültü tepeyi ve fazı da biraz bozuyor** (12 → 12.44 / 13.49, −90° → −92°). k = 50'deki gürültü bileşeni sinüsün harmoniğiyle karmaşık olarak toplanıyor.
- **Sızıntı (leakage):** f = 50.5 Hz'de sinüs vektör boyunca tam sayı devir yapmıyor. Enerji k = 50 ve 51'e bölünüyor (7.67 + 7.61) ve yavaş azalan kuyruklarla çevredeki harmoniklere yayılıyor. Bu gürültü değil; DFT sinyali periyodik varsaydığı için uçlardaki süreksizlikten kaynaklanıyor.
