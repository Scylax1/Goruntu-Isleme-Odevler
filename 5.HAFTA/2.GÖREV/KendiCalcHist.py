"""
Görüntü İşleme Ödevi - calcHist Fonksiyonunu Elle Yazma
-------------------------------------------------------
cv2.calcHist'in tek boyutlu (tek kanallı) hâlini baştan yazar ve renkli
resmin her kanalı için ayrı ayrı çalıştırır.

OpenCV'deki imza:

    cv2.calcHist(images, channels, mask, histSize, ranges)

Burada yazılan kendi_calc_hist aynı parametreleri, aynı sırayla alır ve
aynı biçimde (histSize x 1, float32) sonuç döndürür:

    images   : resim listesi (burada tek resim kullanılır, images[0])
    channels : [kanal indeksi]  (0: B, 1: G, 2: R; gri resimde 0)
    mask     : None ya da resimle aynı boyutta uint8 maske. Yalnızca
               maskenin 0'dan farklı olduğu pikseller sayılır.
    histSize : [kutu sayısı]
    ranges   : [alt, üst]  (üst sınır dahil değil)

Kutu hesabı (OpenCV'nin düzgün aralıklı histogramıyla aynı):

    ölçek = kutu_sayısı / (üst - alt)
    kutu  = floor((değer - alt) * ölçek)
    alt <= değer < üst değilse piksel sayılmaz.

Her piksel açık bir Python döngüsüyle dolaşılır. Kutu indeksleri her olası
değer (0..255) için önceden bir tabloda hesaplanır; böylece döngü içinde
çarpma / bölme yapılmaz. Sonuç cv2.calcHist ile karşılaştırılır.
Beklenen fark 0'dır.

Varsayılan girdi 1. görevdeki baboon.jpg resmidir.

Kullanım:
    python KendiCalcHist.py [resim_yolu]
"""

import math
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
KAYNAK_RESIM = os.path.join(KLASOR, "..", "1.GÖREV", "ciktilar", "baboon.jpg")

KANALLAR = [(0, "Mavi (B)", "tab:blue"),
            (1, "Yeşil (G)", "tab:green"),
            (2, "Kırmızı (R)", "tab:red")]


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma yardımcısı
# ---------------------------------------------------------------------------
def renkli_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Kendi calcHist fonksiyonumuz
# ---------------------------------------------------------------------------
def kendi_calc_hist(images, channels, mask, histSize, ranges):
    """cv2.calcHist'in tek kanallı, düzgün aralıklı karşılığı (uint8 resim)."""
    resim = images[0]
    kanal = channels[0]
    kutu_sayisi = histSize[0]
    alt, ust = ranges

    # Tek kanalı al (gri resimde kanal 0 resmin kendisidir)
    if resim.ndim == 2:
        if kanal != 0:
            raise ValueError("Gri resimde yalnızca kanal 0 vardır")
        duzlem = resim
    else:
        duzlem = resim[:, :, kanal]
    if mask is not None and mask.shape != duzlem.shape:
        raise ValueError("Maske resimle aynı boyutta olmalı")

    # Her olası piksel değeri (0..255) için kutu indeksi: -1 = aralık dışı
    olcek = kutu_sayisi / (ust - alt)
    kutu_tablosu = []
    for deger in range(256):
        if alt <= deger < ust:
            kutu = int(math.floor((deger - alt) * olcek))
            kutu_tablosu.append(min(kutu, kutu_sayisi - 1))
        else:
            kutu_tablosu.append(-1)

    hist = [0] * kutu_sayisi
    yukseklik, genislik = duzlem.shape
    satirlar = duzlem.tolist()
    maske_satirlari = mask.tolist() if mask is not None else None
    for y in range(yukseklik):
        satir = satirlar[y]
        maske_satiri = maske_satirlari[y] if maske_satirlari is not None else None
        for x in range(genislik):
            if maske_satiri is not None and maske_satiri[x] == 0:
                continue
            kutu = kutu_tablosu[satir[x]]
            if kutu >= 0:
                hist[kutu] += 1

    return np.array(hist, dtype=np.float32).reshape(kutu_sayisi, 1)


# ---------------------------------------------------------------------------
# Karşılaştırma
# ---------------------------------------------------------------------------
def karsilastir(resim):
    """Farklı ayarlarda kendi_calc_hist ile cv2.calcHist'i karşılaştırır."""
    yukseklik, genislik = resim.shape[:2]
    # Ortada daire şeklinde maske
    maske = np.zeros((yukseklik, genislik), np.uint8)
    cv2.circle(maske, (genislik // 2, yukseklik // 2), min(yukseklik, genislik) // 3, 255, -1)

    denemeler = [("256 kutu, [0, 256)", None, [256], [0, 256]),
                 ("32 kutu, [0, 256)", None, [32], [0, 256]),
                 ("10 kutu, [0, 256)", None, [10], [0, 256]),
                 ("50 kutu, [100, 200)", None, [50], [100, 200]),
                 ("256 kutu, daire maske", maske, [256], [0, 256])]

    print(f"\n{'Ayar':<24}{'Kanal':<13}{'Toplam':>9}{'Kendi süre':>12}{'cv2 süre':>10}{'En büyük fark':>15}")
    en_buyuk = 0.0
    for ad, m, boyut, aralik in denemeler:
        for kanal, kanal_adi, _ in KANALLAR:
            t0 = time.perf_counter()
            kendi = kendi_calc_hist([resim], [kanal], m, boyut, aralik)
            t1 = time.perf_counter()
            hazir = cv2.calcHist([resim], [kanal], m, boyut, aralik)
            t2 = time.perf_counter()
            assert kendi.shape == hazir.shape and kendi.dtype == hazir.dtype
            fark = float(np.abs(kendi - hazir).max())
            en_buyuk = max(en_buyuk, fark)
            print(f"{ad:<24}{kanal_adi:<13}{int(kendi.sum()):>9}{(t1 - t0) * 1000:>10.1f}ms"
                  f"{(t2 - t1) * 1000:>8.2f}ms{fark:>15.1f}")
    print(f"\nBütün denemelerde kendi_calc_hist ile cv2.calcHist arasındaki en büyük fark: {en_buyuk}")
    return maske


# ---------------------------------------------------------------------------
# Çizim
# ---------------------------------------------------------------------------
def ciz_ve_kaydet(resim, maske):
    fig, eksenler = plt.subplots(2, 4, figsize=(16, 8))

    eksenler[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    eksenler[0, 0].set_title("Orijinal")
    eksenler[1, 0].imshow(cv2.cvtColor(cv2.bitwise_and(resim, resim, mask=maske), cv2.COLOR_BGR2RGB))
    eksenler[1, 0].set_title("Maskeli bölge")
    for ax in eksenler[:, 0]:
        ax.axis("off")

    for i, (kanal, ad, renk) in enumerate(KANALLAR):
        ax = eksenler[0, i + 1]
        kendi = kendi_calc_hist([resim], [kanal], None, [256], [0, 256]).ravel()
        hazir = cv2.calcHist([resim], [kanal], None, [256], [0, 256]).ravel()
        ax.bar(np.arange(256), kendi, width=1.0, color=renk, label="kendi_calc_hist")
        ax.plot(np.arange(256), hazir, color="black", linewidth=0.8, linestyle="--", label="cv2.calcHist")
        ax.set_xlim(0, 255)
        ax.set_title(f"{ad}: 256 kutu (fark = {np.abs(kendi - hazir).max():.0f})")
        ax.legend(fontsize=8)

        ax = eksenler[1, i + 1]
        kendi = kendi_calc_hist([resim], [kanal], maske, [32], [0, 256]).ravel()
        hazir = cv2.calcHist([resim], [kanal], maske, [32], [0, 256]).ravel()
        merkezler = np.arange(32) * 8 + 4
        ax.bar(merkezler, kendi, width=7.0, color=renk, label="kendi_calc_hist")
        ax.plot(merkezler, hazir, "k.", label="cv2.calcHist")
        ax.set_xlim(0, 256)
        ax.set_title(f"{ad}: 32 kutu + maske (fark = {np.abs(kendi - hazir).max():.0f})")
        ax.set_xlabel("Parlaklık")
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = os.path.join(CIKTI_KLASORU, "kendi_calchist.png")
    fig.savefig(yol, dpi=110)
    print("Kaydedildi:", yol)
    plt.show()


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = renkli_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape)

    maske = karsilastir(resim)
    ciz_ve_kaydet(resim, maske)


if __name__ == "__main__":
    main()
