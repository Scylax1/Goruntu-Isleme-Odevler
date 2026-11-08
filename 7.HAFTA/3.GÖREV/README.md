# Gürültülü Sinyali Gauss ve Medyan Filtreyle Temizleme

`SinyalFiltreleme.py`: Elle yazılmış iki tek boyutlu filtre.

- **`gauss_filtre(x, sigma)`:** Çekirdek `exp(−k² / 2σ²)`, k = −3σ … 3σ, toplamı 1 olacak şekilde normalize. Konvolüsyonla uygulanıyor. Doğrusal bir alçak geçiren filtre.
- **`medyan_filtre(x, pencere)`:** Penceredeki değerlerin ortancası. Doğrusal değil.
- **Kenarlar:** İki filtrede de sinyal kenarlarda ayna gibi yansıtılıyor.

Sinyal: A = 5, f = 4 Hz, N = 256. Başarı, temiz sinyale olan RMSE ile ölçüldü; her filtrenin genişliği tarandı.

## 1) Gauss gürültüsü (2. görevdeki `gurultu_ekle`, oran 0.3)

Gürültülü sinyalin RMSE'si 1.548.

| Gauss σ | 0.5 | 1 | 2 | 3 | **4** | 6 | 8 |
|---|---|---|---|---|---|---|---|
| RMSE | 1.27 | 0.89 | 0.69 | 0.62 | **0.61** | 0.73 | 1.01 |

| Medyan pencere | 3 | 5 | 7 | 9 | **11** | 21 | 31 |
|---|---|---|---|---|---|---|---|
| RMSE | 1.03 | 0.82 | 0.76 | 0.75 | **0.75** | 0.85 | 1.10 |

## 2) Dürtü (tuz-biber) gürültüsü (örneklerin %10'u ±3A'ya sıçrıyor)

Gürültülü sinyalin RMSE'si 4.970.

| Gauss σ | 0.5 | 1 | 2 | 4 | **6** | 8 |
|---|---|---|---|---|---|---|
| RMSE | 4.00 | 2.71 | 1.98 | 1.60 | **1.57** | 1.67 |

| Medyan pencere | 3 | **5** | 7 | 11 | 21 | 31 |
|---|---|---|---|---|---|---|
| RMSE | 2.12 | **0.24** | 0.29 | 0.38 | 0.64 | 1.05 |

## Yorum

- **Gauss gürültüsünde Gauss filtre daha iyi** (0.61'e karşı 0.75). Gauss gürültüsü her örneği biraz bozuyor; komşuların ağırlıklı ortalaması bu küçük ve simetrik sapmaları en iyi şekilde birbirine götürüyor. Medyan ise ortalamaya göre daha az örnekten bilgi kullanıyor ve basamaklı, köşeli bir çıktı veriyor.
- **Dürtü gürültüsünde medyan açık ara önde** (0.24'e karşı 1.57). Gauss filtre her dürtüyü komşularına yayıp çan biçiminde bir tümsek bırakıyor ve tepeleri aşağı çekiyor. Medyanda ise penceredeki dürtüler azınlıkta kaldıkça sıralamanın uçlarına itiliyor ve ortancaya hiç katılmıyor. Pencere 3'te sonuç kötü (2.12): yan yana iki dürtü 3'lük pencerede çoğunluğu alıyor. Pencere 5'te çoğunluk için 3 dürtü gerekiyor ve sinyal neredeyse tamamen kurtuluyor.
- **Aşırı geniş filtreler** (`ciktilar/asiri_genis_filtre.png`):
  - Gauss filtre sinüsün genliğini de düşürüyor. σ = 8'de 4 Hz'lik sinüs ×0.735'e iniyor; kuramsal kazanç `exp(−2(π f σ / N)²)`.
  - Medyan filtre tepeleri düzleştirip sinüsü kare dalgaya benzetiyor.
  - İkisinde de en iyi genişlik, gürültüyü bastırmak ile sinyali bozmamak arasındaki dengede.
- **Kenar etkisi:** Ayna yansıtma, sıfırdan başlayan sinüsün kenarında küçük bir sapma bırakıyor (t = 0 civarı). Sinüs n = 0 etrafında tek fonksiyon, yansıtma ise onu çift fonksiyon gibi uzatıyor.
