"""
Görüntü İşleme Ödevi - Histogram Eşitleme (Histogram Equalization)
------------------------------------------------------------------
Düşük kontrastlı, tek kanallı (gri) bir görüntüyü OpenCV ile açar ve:

  1) 256 değerli histogramını çıkarır:      h[k] = k tonundaki piksel sayısı
  2) Kümülatif dağılımı (CDF) hesaplar:     cdf[k] = h[0] + h[1] + ... + h[k]
  3) CDF ile her tona yeni değer atar:
         yeni[k] = round( (cdf[k] - cdf_min) * 255 / (N - cdf_min) )
     (N = toplam piksel sayısı, cdf_min = sıfır olmayan ilk CDF değeri)
  4) Bu tabloyu pikseller üzerinde uygulayıp sonucu kaydeder.

Histogram, CDF ve eşitleme için cv2.calcHist, np.histogram, np.cumsum,
cv2.equalizeHist gibi hazır fonksiyonlar KULLANILMAZ; hepsi döngüyle yapılır.

Varsayılan girdi, kontrast germe ödevinde üretilen "dusuk_kontrast.png" resmidir. O resim
yoksa cameraman resminin tonları [100, 160] aralığına sıkıştırılarak üretilir.

Kullanım:
    python HistogramEsitleme.py [resim_yolu]

Herhangi bir tuşa basınca pencere kapanır.
"""

import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
KAYNAK_RESIM = os.path.join(KLASOR, "..", "2.HAFTA", "ciktilar", "cameraman.png")
DUSUK_KONTRAST_RESIM = os.path.join(CIKTI_KLASORU, "dusuk_kontrast.png")

# Düşük kontrastlı resim üretirken tonların sıkıştırılacağı aralık
DAR_ALT, DAR_UST = 100, 160


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma / yazma yardımcıları
# ---------------------------------------------------------------------------
def gri_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


def kaydet(yol, resim):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    cv2.imencode(".png", resim)[1].tofile(yol)
    print("Kaydedildi:", yol)


# ---------------------------------------------------------------------------
# 1) Düşük kontrastlı resmi hazırlama
# ---------------------------------------------------------------------------
def dusuk_kontrast_resim_olustur():
    """Cameraman resminin 0-255 tonlarını DAR_ALT-DAR_UST aralığına sıkıştırır."""
    resim = gri_oku(KAYNAK_RESIM)
    dar = DAR_ALT + resim.astype(np.float32) * (DAR_UST - DAR_ALT) / 255.0
    kaydet(DUSUK_KONTRAST_RESIM, dar.astype(np.uint8))


# ---------------------------------------------------------------------------
# 2) Histogram, CDF ve eşitleme (hazır fonksiyon yok)
# ---------------------------------------------------------------------------
def histogram_hesapla(resim):
    """Her pikseli gezip kendi tonunun sayacını 1 artırır (256 kutu)."""
    yukseklik, genislik = resim.shape
    histogram = [0] * 256
    for y in range(yukseklik):
        for x in range(genislik):
            histogram[int(resim[y, x])] += 1
    return histogram


def cdf_hesapla(histogram):
    """cdf[k] = h[0] + ... + h[k]  (kümülatif toplam)."""
    cdf = [0] * 256
    toplam = 0
    for k in range(256):
        toplam += histogram[k]
        cdf[k] = toplam
    return cdf


def esitleme_tablosu(cdf):
    """CDF'yi [0, 255] aralığına yayan dönüşüm tablosunu (LUT) çıkarır.

    yeni[k] = round( (cdf[k] - cdf_min) * 255 / (N - cdf_min) )
    En koyu ton 0'a, en parlak ton 255'e gider; aradaki tonlar
    kaç piksele sahip olduklarına göre (CDF eğimine göre) yayılır.
    """
    piksel_sayisi = cdf[255]

    # Sıfır olmayan ilk CDF değeri = resimdeki en koyu tonun piksel sayısı
    cdf_min = 0
    for k in range(256):
        if cdf[k] > 0:
            cdf_min = cdf[k]
            break

    tablo = [0] * 256
    payda = piksel_sayisi - cdf_min
    if payda == 0:
        # Resim tek renkse yayılacak bir dağılım yoktur
        return tablo

    for k in range(256):
        if cdf[k] < cdf_min:
            continue                       # resimde hiç olmayan koyu tonlar
        yeni = (cdf[k] - cdf_min) * 255.0 / payda
        tablo[k] = int(yeni + 0.5)         # en yakın tam sayıya yuvarla
    return tablo


def tabloyu_uygula(resim, tablo):
    """Her pikselin tonunu tablodaki karşılığıyla değiştirir."""
    yukseklik, genislik = resim.shape
    sonuc = np.zeros((yukseklik, genislik), dtype=np.uint8)
    for y in range(yukseklik):
        for x in range(genislik):
            sonuc[y, x] = tablo[int(resim[y, x])]
    return sonuc


# ---------------------------------------------------------------------------
# 3) Sonuçları gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def histogram_ciz(histogram, cdf, genislik, yukseklik=200):
    """Histogramı gri sütunlarla, CDF'yi kırmızı eğriyle aynı tuvale çizer."""
    tuval = np.full((yukseklik, genislik, 3), 255, dtype=np.uint8)
    en_cok = max(histogram)
    alan = yukseklik - 10

    for ton in range(256):
        boy = int(histogram[ton] / en_cok * alan)
        x = int(ton * genislik / 256)
        x2 = int((ton + 1) * genislik / 256)
        cv2.rectangle(tuval, (x, yukseklik - boy), (x2 - 1, yukseklik - 1), (60, 60, 60), -1)

    onceki = None
    for ton in range(256):
        nokta = (int((ton + 0.5) * genislik / 256),
                 yukseklik - 1 - int(cdf[ton] / cdf[255] * alan))
        if onceki is not None:
            cv2.line(tuval, onceki, nokta, (0, 0, 255), 2)
        onceki = nokta
    return tuval


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        yol = sys.argv[1]
    else:
        if not os.path.exists(DUSUK_KONTRAST_RESIM):
            dusuk_kontrast_resim_olustur()
        yol = DUSUK_KONTRAST_RESIM

    # 1) Düşük kontrastlı, tek kanallı resmi OpenCV ile aç
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape, "| Kanal sayısı: 1")

    # 2) 256 değerli histogram
    histogram = histogram_hesapla(resim)
    dolu_tonlar = [k for k in range(256) if histogram[k] > 0]
    print(f"Histogram -> kullanılan ton sayısı: {len(dolu_tonlar)} "
          f"| aralık: [{dolu_tonlar[0]}, {dolu_tonlar[-1]}]")

    # 3) Kümülatif dağılım (CDF)
    cdf = cdf_hesapla(histogram)
    print(f"CDF -> son değer (toplam piksel): {cdf[255]}")

    # 4) Histogram eşitleme
    tablo = esitleme_tablosu(cdf)
    sonuc = tabloyu_uygula(resim, tablo)

    yeni_histogram = histogram_hesapla(sonuc)
    yeni_cdf = cdf_hesapla(yeni_histogram)
    yeni_dolu = [k for k in range(256) if yeni_histogram[k] > 0]
    print(f"Eşitleme sonrası -> aralık: [{yeni_dolu[0]}, {yeni_dolu[-1]}]")

    # Sadece doğrulama amaçlı: OpenCV'nin hazır sonuçlarıyla karşılaştır
    hazir_hist = cv2.calcHist([resim], [0], None, [256], [0, 256]).ravel()
    hist_ayni = all(int(hazir_hist[k]) == histogram[k] for k in range(256))
    hazir = cv2.equalizeHist(resim)
    fark = np.abs(sonuc.astype(int) - hazir.astype(int)).max()
    print(f"Kontrol -> cv2.calcHist ile aynı mı: {hist_ayni} | "
          f"cv2.equalizeHist ile en büyük piksel farkı: {fark}")

    kaydet(os.path.join(CIKTI_KLASORU, "histogram_esitleme_sonuc.png"), sonuc)

    genislik = resim.shape[1]
    ust = np.hstack([etiket_ekle(resim, f"Orijinal [{dolu_tonlar[0]}, {dolu_tonlar[-1]}]"),
                     etiket_ekle(sonuc, f"Esitlenmis [{yeni_dolu[0]}, {yeni_dolu[-1]}]")])
    alt = np.hstack([histogram_ciz(histogram, cdf, genislik),
                     histogram_ciz(yeni_histogram, yeni_cdf, genislik)])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "histogram_esitleme_karsilastirma.png"), karsilastirma)

    pencere = "Histogram esitleme - Orijinal / Esitlenmis (kirmizi: CDF)"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
