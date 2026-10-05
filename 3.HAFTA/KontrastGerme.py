"""
Görüntü İşleme Ödevi - Kontrast Germe (Min-Maks Doğrusal Ölçekleme)
-------------------------------------------------------------------
Düşük kontrastlı, tek kanallı (gri) bir görüntüyü OpenCV ile açar,
içindeki minimum ve maksimum parlaklık değerlerini bulur ve bu değerleri
kullanarak tüm pikselleri [0, 255] aralığına doğrusal olarak ölçekler:

    yeni = (eski - min) * 255 / (maks - min)

Ölçekleme için cv2.normalize, np.interp gibi hazır fonksiyonlar
KULLANILMAZ; min/maks bulma ve ölçekleme piksel piksel döngüyle yapılır.

Düşük kontrastlı resim yoksa, 2. haftadaki cameraman resminin tonları
[100, 160] aralığına sıkıştırılarak "dusuk_kontrast.png" oluşturulur.

Kullanım:
    python KontrastGerme.py [resim_yolu]

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
# 2) Min / maks bulma ve doğrusal ölçekleme (hazır fonksiyon yok)
# ---------------------------------------------------------------------------
def min_maks_bul(resim):
    """Tüm pikselleri tek tek gezerek en küçük ve en büyük değeri bulur."""
    yukseklik, genislik = resim.shape
    en_kucuk = 255
    en_buyuk = 0
    for y in range(yukseklik):
        for x in range(genislik):
            deger = int(resim[y, x])
            if deger < en_kucuk:
                en_kucuk = deger
            if deger > en_buyuk:
                en_buyuk = deger
    return en_kucuk, en_buyuk


def dogrusal_olcekle(resim, en_kucuk, en_buyuk):
    """[min, maks] aralığını [0, 255] aralığına doğrusal olarak taşır.

    yeni = (eski - min) * 255 / (maks - min)
    min -> 0, maks -> 255 olur; aradaki değerler orantılı yayılır.
    """
    yukseklik, genislik = resim.shape
    sonuc = np.zeros((yukseklik, genislik), dtype=np.uint8)

    aralik = en_buyuk - en_kucuk
    if aralik == 0:
        # Resim tek renkse ölçeklenecek bir aralık yoktur
        return sonuc

    for y in range(yukseklik):
        for x in range(genislik):
            yeni = (int(resim[y, x]) - en_kucuk) * 255.0 / aralik
            sonuc[y, x] = int(yeni + 0.5)   # en yakın tam sayıya yuvarla
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


def histogram_ciz(resim, yukseklik=200):
    """256 sütunluk basit bir histogram görüntüsü çizer (genişlik = resim genişliği)."""
    sayac = np.zeros(256, dtype=np.int64)
    for deger in resim.ravel():
        sayac[deger] += 1

    genislik = resim.shape[1]
    tuval = np.full((yukseklik, genislik, 3), 255, dtype=np.uint8)
    en_cok = sayac.max()
    for ton in range(256):
        boy = int(sayac[ton] / en_cok * (yukseklik - 10))
        x = int(ton * genislik / 256)
        x2 = int((ton + 1) * genislik / 256)
        cv2.rectangle(tuval, (x, yukseklik - boy), (x2 - 1, yukseklik - 1), (60, 60, 60), -1)
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

    # 2) Minimum ve maksimum değerleri bul
    en_kucuk, en_buyuk = min_maks_bul(resim)
    print(f"Minimum: {en_kucuk} | Maksimum: {en_buyuk}")

    # 3) [min, maks] -> [0, 255] doğrusal ölçekleme
    sonuc = dogrusal_olcekle(resim, en_kucuk, en_buyuk)
    yeni_min, yeni_maks = min_maks_bul(sonuc)
    print(f"Ölçekleme sonrası -> Minimum: {yeni_min} | Maksimum: {yeni_maks}")

    # Sadece doğrulama amaçlı: OpenCV'nin hazır sonuçlarıyla karşılaştır
    cv_min, cv_maks, _, _ = cv2.minMaxLoc(resim)
    hazir = cv2.normalize(resim, None, 0, 255, cv2.NORM_MINMAX)
    fark = np.abs(sonuc.astype(int) - hazir.astype(int)).max()
    print(f"Kontrol -> cv2.minMaxLoc: ({int(cv_min)}, {int(cv_maks)}) | "
          f"cv2.normalize ile en büyük piksel farkı: {fark}")

    kaydet(os.path.join(CIKTI_KLASORU, "kontrast_germe_sonuc.png"), sonuc)

    ust = np.hstack([etiket_ekle(resim, f"Orijinal [{en_kucuk}, {en_buyuk}]"),
                     etiket_ekle(sonuc, f"Olceklenmis [{yeni_min}, {yeni_maks}]")])
    alt = np.hstack([histogram_ciz(resim), histogram_ciz(sonuc)])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "kontrast_germe_karsilastirma.png"), karsilastirma)

    pencere = "Kontrast germe - Orijinal / Olceklenmis"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
