# 2 Boyutlu DFT

`DFT2D.py`

```python
dft2(resim)  ->  harmonik matrisi F (karmaşık, M x N)
idft2(F)     ->  resim
merkeze_al(F)   # DC'yi ortaya taşır (fftshift karşılığı)
```

**F[u, v] = Σₘ Σₙ f[m, n] · e^(−j2π(um/M + vn/N))**

Üstel ifade iki çarpana ayrılıyor: e^(−j2πum/M) · e^(−j2πvn/N). Bu yüzden 2D DFT, önce her satıra sonra her sütuna 1D DFT uygulamak demek. Matris biçiminde:

**F = W_M · f · W_Nᵀ**, burada W_N[k, n] = e^(−j2πkn/N).

- **DFT matrisi:** 4. görevdeki gibi yalnızca N tane cos/sin değerinden kuruluyor: W[k, n] = COS[(kn) mod N] − j·SIN[(kn) mod N].
- **Tanımın kendisi:** Dört iç içe döngüyle `dft2_dongu` fonksiyonunda da yazıldı.
- **Neden matris yolu?** 512x512 resimde satır satır Python döngüsüyle ~20 s sürerdi. Matris çarpımı aynı toplamları 17 ms'de hesaplıyor.

## Doğrulama

| Kontrol | Sonuç |
|---|---|
| 16x16 resimde dört döngülü tanım ile matris yolu | en büyük fark 5·10⁻¹¹ |
| 512x512'de `np.fft.fft2` ile | göreli fark 2.5·10⁻¹⁶ |
| `idft2(dft2(f)) − f` | en büyük 6·10⁻¹² |
| F[0, 0] | 33 832 495 = ortalama parlaklık 129.06 × 262 144 piksel |

## Harmonik matrisinin yorumu

**Yapay desenler (`ciktilar/yapay_desenler_dft.png`):**
- **Dikey çizgiler** (yatayda 8 devirlik sinüs): Merkezde DC ve (0, ±8)'de iki nokta. Tek bir sinüs, harmonik matrisinde tek bir nokta çifti.
- **Eğik çizgiler** (6, 12): (±12, ±6)'da iki nokta. Nokta çifti, çizgilere **dik** doğrultuda duruyor.
- **Kare:** Yatay ve dikey eksen boyunca uzanan bir haç ve ızgara deseni: x ve y'de sinc fonksiyonlarının çarpımı. Karenin dikey kenarları yatay eksende, yatay kenarları dikey eksende enerji veriyor.
- **Daire:** Merkezden dışa iç içe halkalar (Bessel / Airy deseni). Daire her yönde aynı olduğu için harmonik matrisi de dönel simetrik.

**Cameraman (`ciktilar/cameraman_dft.png`):**
- **Enerji merkezde toplanıyor:** Toplam enerjinin (DC dahil) %95.2'si merkeze 5 harmonikten yakın, %99'u 50 harmonikten yakın. Doğal resimlerin genliği frekansla hızla düşüyor; resmin büyük kısmı yavaş değişen bölgelerden oluşuyor.
- **Parlak eğik çizgiler:** Tripodun ayakları ve kameramanın kolu gibi uzun, düz kenarlardan geliyor; her biri kendine dik bir çizgi veriyor.
- **Parlak yatay ve dikey haç:** DFT resmi periyodik varsaydığı için sol-sağ ve üst-alt kenarlardaki parlaklık sıçramalarından geliyor (gökyüzü ↔ çimen).
- **Faz takası:** Bu deney fazın önemini gösteriyor:
  - Dairenin genliği + cameraman'in fazı: sonuç **cameraman'e** benziyor.
  - Cameraman'in genliği + dairenin fazı: sonuç **daireye** benziyor.

  Kenarların ve nesnelerin **nerede** olduğu bilgisi fazda. Genlik ise daha çok hangi frekansların ne kadar güçlü olduğunu söylüyor.
