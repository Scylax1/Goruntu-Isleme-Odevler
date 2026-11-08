# X-ray Bagaj Görüntülerini Gri Tona Çevirip Yeniden Renklendirme

`XrayRenklendirme.py`, HUMS X-ray veri setinden 5 görüntüyü gri tona çevirir, güvenlik operatörleri için tasarlanmış bir renk paletiyle yeniden renklendirir ve sonucu orijinal renkli resimle karşılaştırır.

## Veri

- **Kaynak:** [HUMS X-ray Dataset](https://www.kaggle.com/datasets/orvile/hums-x-ray-dataset), lisans CC BY 4.0. Çift enerjili bir bagaj tarayıcısından alınmış, bıçak tespiti için hazırlanmış bir veri seti.
- **Klasör:** Ödevde "ThreadsRGB" diye geçen klasörün asıl adı `ThreatsRGB` (tehdit içeren 150 RGB görüntü).
- **Seçilen 5 görüntü (`veri/`):** Malzeme çeşitliliği için seçildi:
  - `image_0_1000`: bavulda çok sayıda bıçak
  - `image_0_1072`: standart test kiti (basamaklı bloklar, teller, bıçaklar)
  - `image_0_1212`: organik eşya arasında bıçaklar
  - `image_0_1302`: organik şişe ve bıçaklar
  - `image_0_1388`: yakın plan bıçaklar

## Orijinal renkler ne anlatıyor?

Tarayıcı her pikseli yüksek ve düşük iki enerjide ölçüyor. İki ölçümün oranı malzemenin **etkin atom numarasını (Z)**, toplam soğurma da **kalınlık × yoğunluğu** veriyor. Sektör standardı renk kodunda ton malzemeyi, parlaklık kalınlığı gösteriyor:

| Renk | Malzeme |
|---|---|
| Turuncu | Organik, düşük Z: plastik, kağıt, gıda, **patlayıcı** |
| Yeşil | Karışık / hafif inorganik: alüminyum, cam |
| Mavi | Metal, yüksek Z: çelik, bıçak, silah |
| Siyah | Nüfuz edilemeyen |

## Yöntem

1. **Gri dönüşüm:** `cv2.cvtColor(BGR2GRAY)`, Y = 0.299 R + 0.587 G + 0.114 B.
2. **Renklendirme:** Gri değer bir renk tablosuna (LUT) bakılarak renge çevriliyor. Tablo, 10 çapa noktası arasında doğrusal ara değerlemeyle 256 renge açılıyor:

   | Gri | 0 | 25 | 70 | 115 | 140 | 170 | 190 | 225 | 245 | 255 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | Renk | siyah | lacivert | mavi | açık mavi | yeşil | sarı-yeşil | turuncu | açık turuncu | krem | beyaz |

## Neden bu palet?

1. **Tanıdık kod.** Operatörler zaten turuncu / yeşil / mavi / siyah koduyla eğitiliyor. Aynı sıralama korunduğu için yeni bir şey öğrenmeleri gerekmiyor. Koyu (yoğun) bölgeler mavi-siyah, orta bölgeler yeşil, açık (ince) bölgeler turuncu, boş alan beyaz.
2. **Tehdit öne çıkıyor.** Metal bıçaklar en koyu gri değerlerde toplanıyor. Onlara açık zemine en zıt düşen doygun maviyi verdik; bıçaklar beyaz ve turuncu zeminde hemen göze çarpıyor.
3. **Parlaklık korunuyor.** Paletteki her rengin parlaklığı (luma), karşılığı olan gri değere en fazla **4** birim uzak ve tekdüze artıyor (`ciktilar/palet.png`). Renk, gri resmin bilgisini bozmuyor, yalnızca kategori ekliyor. Renkli resim tekrar griye çevrilirse neredeyse aynı gri resim çıkıyor.
4. **JET neden seçilmedi?** JET gibi gökkuşağı paletlerinin parlaklığı iniş-çıkış yapıyor: gri 159'da 226'ya çıkıp gri 255'te 38'e düşüyor. Bu, resimde olmayan sınırlar uyduruyor. Üstelik boş zemini koyu bordoya boyuyor; ekranın çoğu dikkat çeken koyu kırmızı oluyor ve asıl nesneler kayboluyor. Karşılaştırma resminde 4. sütunda görülüyor.

## Orijinal ile fark ve yorumu

Orijinal resmin her pikseli tonuna göre (HSV), bizim resmimizin her pikseli de gri aralığına göre 4 sınıfa ayrılıp karşılaştırıldı. 5 resmin toplamı:

| Orijinal → Bizim | arka plan | turuncu | yeşil | mavi/siyah |
|---|---|---|---|---|
| **arka plan** | **95.1%** | 3.7% | 0.9% | 0.3% |
| **turuncu** | 7.5% | **77.2%** | 12.3% | 3.0% |
| **yeşil** | 46.6% | 29.7% | **13.0%** | 10.7% |
| **mavi/siyah** | 0.0% | 6.4% | 32.6% | **61.0%** |

Toplam uyuşma **%75.7**; resim başına %69.0 ile %86.3 arasında.

**Neden fark var?** Gri resim yalnızca **ne kadar soğurulduğunu** (kalınlık × yoğunluk) taşıyor. Orijinaldeki rengi belirleyen **malzeme (Z) bilgisi**, iki enerjinin oranından geliyor ve griye çevirince geri dönmemek üzere kayboluyor. Bizim palet rengi yalnızca parlaklıktan tahmin edebiliyor. Bu yüzden:

- **İnce metal açık görünüyor ve yeşile ya da turuncuya kayıyor.** Orijinalde mavi olan piksellerin %39'u bizde yeşil ya da turuncu. `image_0_1388`'deki ortadaki bıçak orijinalde mavi (metal), bizde yeşil. Uyuşmayan pikseller haritasında bıçağın tamamı mavi olarak işaretli.
- **Kalın organik madde koyu görünüyor ve mavi ya da yeşile kayıyor.** `image_0_1072`'deki basamaklı turuncu test blokları ve `image_0_1302`'deki turuncu şişe bizde yeşil ve mavi görünüyor.
- **Yeşil sınıfı neredeyse tamamen kayboluyor** (yalnızca %13 doğru). Orijinaldeki yeşil, bavul kabuğu ve ince alüminyum gibi hafif malzemeler; bunlar ince olduğu için gri değeri çok açık. Bizde zemine (%47) ya da turuncuya (%30) karışıyor.
- **Uyuşan yerler:** Boş zemin (%95) ve kalın metal parçalar (bıçak sapları, kilitler, test kitindeki çelik bloklar) doğru renkte.

**Güvenlik açısından sonuç.** Tek kanallı gri resimden yapılan renklendirme, operatörün **yoğun nesneleri** hızla görmesini kolaylaştırıyor ve bıçakların büyük kısmını mavi-yeşil olarak öne çıkarıyor. Ancak **malzeme ayrımı yapamıyor**. En tehlikeli durum şu: kalın bir organik kütle (patlayıcı organiktir) bizim paletimizde metal gibi mavi görünebiliyor, ince bir bıçak da organik gibi turuncu-yeşil görünebiliyor. Orijinal çift enerji renklendirmesinin yerini tutamaz, yalnızca tek enerjili (gri) bir tarayıcıda okunabilirliği artırır.

## Çıktılar (`ciktilar/`)

- `renklendirme_karsilastirma.png`: Her resim için orijinal, gri, bizim renklendirme, JET ve sınıfı farklı pikseller. Farklı pikseller, orijinaldeki doğru sınıfının renginde gösteriliyor.
- `palet.png`: Bizim palet ve JET renk şeritleri, parlaklık eğrileri.
- `image_0_*_gri.png`, `image_0_*_renkli.png`: Her resmin gri ve renklendirilmiş hâli.
