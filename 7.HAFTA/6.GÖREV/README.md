# Eşik Tabanlı Frekans Filtresi

`FrekansFiltresi.py`

```python
frekans_filtresi(vektor, esik)  ->  (yeniden üretilmiş sinyal, filtrelenmiş X, maske)
```

1. `X = dft(vektor)` (4. görev)
2. Her harmoniğin sinüs genliği hesaplanıyor: 2|X[k]|/N (DC ve Nyquist'te |X[k]|/N).
3. Genliği **eşiğin altında kalan bütün harmonikler sıfırlanıyor.** k ve N−k aynı harmonik olduğu için genlikleri eşit; ikisi birlikte tutuluyor ya da birlikte atılıyor. Böylece sonuç gerçel kalıyor.
4. `idft(X_filtreli)` (4. görev) ile sinyal yeniden üretiliyor.

**Neden işe yarıyor?** Beyaz gürültü bütün harmoniklere küçük ve düz bir taban olarak dağılıyor (~0.3). Sinüsler ise tek harmonikte toplanıyor (12 ve 5). Eşik bu ikisinin arasına konunca gürültünün neredeyse tamamı atılıyor. Geriye yalnızca gürültünün o harmoniklere düşen küçük payı kalıyor.

## Sonuç (eşik = 2, 5. görevdeki sinyaller)

| Sinyal | Kalan harmonikler | RMSE önce | RMSE sonra | İyileşme |
|---|---|---|---|---|
| Sinyal 1 (50 Hz, A = 12) | [50] | 3.760 | 0.330 | 11.4× |
| Sinyal 2 (30 Hz, A = 5) | [30] | 1.459 | 0.075 | 19.6× |
| Toplam (50 + 30 Hz) | [30, 50] | 4.045 | 0.444 | 9.1× |

RMSE, gürültüsüz sinyale göre ölçüldü.

## Eşik taraması (toplam sinyal)

| Eşik | Kalan | RMSE |
|---|---|---|
| 0.25 | 160 harmonik | 3.87 |
| 0.5 | 36 harmonik | 2.56 |
| 0.75 | 5 harmonik | 1.08 |
| **1.0 – 4.6** | **[30, 50]** | **0.444** |
| 4.7 – 12 | [50] | 3.55 |
| 12.5 | — | 9.19 |

## Yorum

- **Geniş ve güvenli bir eşik aralığı var.** 1.0 ile 4.6 arasındaki her eşik aynı sonucu veriyor. Alt sınır, gürültü tabanının en büyük değeri (~0.86); üst sınır, en zayıf sinüsün genliği (4.65). Eşik bu aralığın ortasına (2) kondu.
- **Eşik düşükse gürültü harmonikleri geçiyor.** 0.25'te 160 harmonik kalıyor ve neredeyse hiç temizlik olmuyor.
- **Eşik 30 Hz'in genliğini geçince o bileşen siliniyor.** Bu bir kayıp gibi görünse de aynı filtrenin başka bir kullanımını gösteriyor: toplam sinyal eşik 6 ile filtrelenince yalnızca 50 Hz kalıyor ve sinyal 1'in gürültüsüz hâline RMSE 0.332 ile yaklaşılıyor. Yani filtre karışık sinyalden **bileşen ayırmak** için de kullanılabiliyor.
- **Kalan hata (0.33-0.44) tamamen gürültüden geliyor.** Kalan harmoniklere düşen gürültü payı silinemiyor (k = 50'de 12 yerine 12.44). Bu, aynı frekanstaki gürültüyü sinyalden ayırmanın yolu olmamasından kaynaklanıyor.
- **Sınırlama:** Bu filtre sinyalin birkaç güçlü harmonikten oluştuğu durumlar için uygun. Enerjisini birçok küçük harmoniğe yayan sinyaller (keskin kenarlı dalgalar, darbeler, frekansı tam sayı olmayan ve sızıntılı sinüsler) eşikle birlikte kırpılır ve bozulur.

`ciktilar/frekans_filtresi.png`: Her sinyal için gürültülü, gürültüsüz ve filtrelenmiş sinyal; harmonik vektöründe tutulan (kırmızı) ve atılan (gri) harmonikler; eşik 6 ile bileşen ayırma ve eşik taraması grafiği.
