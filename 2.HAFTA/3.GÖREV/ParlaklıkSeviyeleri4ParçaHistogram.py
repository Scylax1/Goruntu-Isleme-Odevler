"""
Görüntü İşleme Ödevi - 4 Parçalı Parlaklık Deneyleri + Histogram
-----------------------------------------------------------------
ParlaklıkSeviyeleri4Parça.py'deki deneylerin aynısı (fotoğraf 2x2 bölünür,
her parçaya farklı ayar uygulanır), ama bu sefer her parçanın HİSTOGRAMI da
çıkarılır: 0-255 arasındaki her gri tondan kaç piksel olduğu tek tek sayılır.

    +-------------+-------------+
    |  Sol üst    |  Sağ üst    |
    +-------------+-------------+
    |  Sol alt    |  Sağ alt    |
    +-------------+-------------+

  1) Orijinal resim: 4 parçanın histogramı
  2) Nicemleme: her parçada farklı sayıda gri ton -> histogramda o kadar çubuk kalır
  3) Maksimum parlaklık: her parçada farklı en parlak değer -> çubuklar sola sıkışır
  4) Süre: histogram satır satır / sütun sütun sayılır, hazır fonksiyonla karşılaştırılır

Kullanım:
    python ParlaklıkSeviyeleri4ParçaHistogram.py

Pencereyi kapatınca bir sonraki grafiğe geçer.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

# Ortak fonksiyonlar 2. görevin dosyasında; Python o klasörü de arasın
sys.path.insert(0, os.path.join(KLASOR, "..", "2.GÖREV"))

from ParlaklıkSeviyeleri4Parça import (
    MAKS_PARLAKLIKLAR,
    SEVIYE_GRUPLARI,
    etiketle_ve_cizgi_ciz,
    maks_parlakligi_degistir,
    nicemle,
    parca_sinirlari,
    parcalara_uygula,
    resmi_indir,
    sure_olc,
)

PARCA_ADLARI = ["Sol üst", "Sağ üst", "Sol alt", "Sağ alt"]


# ---------------------------------------------------------------------------
# 1) Histogram: değerleri tek tek sayma
# ---------------------------------------------------------------------------
# Her parça için 256 kutulu bir sayaç tutulur. Her pikselin değeri kutu
# numarasıdır: değeri 200 olan piksel görülünce hist[200] bir artırılır.
def histogram_4parca_satir_satir(resim):
    histogramlar = []
    for y1, y2, x1, x2 in parca_sinirlari(resim):
        hist = [0] * 256
        for y in range(y1, y2):             # dış döngü: satırlar
            for x in range(x1, x2):         # iç döngü: sütunlar
                hist[resim[y, x]] += 1
        histogramlar.append(np.array(hist))
    return histogramlar


def histogram_4parca_sutun_sutun(resim):
    histogramlar = []
    for y1, y2, x1, x2 in parca_sinirlari(resim):
        hist = [0] * 256
        for x in range(x1, x2):             # dış döngü: sütunlar
            for y in range(y1, y2):         # iç döngü: satırlar
                hist[resim[y, x]] += 1
        histogramlar.append(np.array(hist))
    return histogramlar


def histogram_4parca_hazir(resim):
    """Karşılaştırma için NumPy'ın hazır sayma fonksiyonu (C ile yazılmış)."""
    return [np.bincount(resim[y1:y2, x1:x2].ravel(), minlength=256)
            for y1, y2, x1, x2 in parca_sinirlari(resim)]


def histogramlari_kontrol_et(resim, histogramlar):
    """Toplam sayı = parçadaki piksel sayısı mı, hazır fonksiyonla aynı mı?"""
    for (y1, y2, x1, x2), hist, hazir, ad in zip(parca_sinirlari(resim), histogramlar,
                                                  histogram_4parca_hazir(resim), PARCA_ADLARI):
        piksel_sayisi = (y2 - y1) * (x2 - x1)
        assert hist.sum() == piksel_sayisi
        assert np.array_equal(hist, hazir)
        print(f"  {ad:<8}: toplam {hist.sum()} piksel (= {y2 - y1}x{x2 - x1}), "
              f"kullanılan ton sayısı: {np.count_nonzero(hist)}")


# ---------------------------------------------------------------------------
# 2) Çizim: solda 4 parçalı resim, sağda aynı yerleşimde 4 histogram
# ---------------------------------------------------------------------------
def histogram_ciz_ve_kaydet(baslik, dosya_adi, resim, yazilar):
    histogramlar = histogram_4parca_satir_satir(resim)
    histogramlari_kontrol_et(resim, histogramlar)

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(baslik, fontsize=14)
    izgara = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

    # Sol: etiketli resim (OpenCV BGR -> matplotlib RGB)
    ax_resim = fig.add_subplot(izgara[:, 0])
    ax_resim.imshow(etiketle_ve_cizgi_ciz(resim, yazilar)[:, :, ::-1])
    ax_resim.axis("off")

    # Sağ: her parçanın histogramı, resimdeki yerine göre 2x2 dizilir
    for i, (hist, ad, yazi) in enumerate(zip(histogramlar, PARCA_ADLARI, yazilar)):
        ax = fig.add_subplot(izgara[i // 2, 1 + i % 2])
        ax.bar(range(256), hist, width=1.0, color="steelblue")
        ax.set_xlim(-1, 256)
        ax.set_title(f"{ad}: {yazi}", fontsize=10)
        ax.set_xlabel("Gri ton")
        ax.set_ylabel("Piksel sayısı")

    fig.tight_layout()
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = os.path.join(CIKTI_KLASORU, dosya_adi)
    fig.savefig(yol, dpi=120)
    print("Kaydedildi:", yol)
    plt.show()


# ---------------------------------------------------------------------------
# 3) Süre: satır satır / sütun sütun / hazır fonksiyon
# ---------------------------------------------------------------------------
def histogram_sure_deneyi(resim):
    print("\n--- Deney 4: Histogram hesaplama süresi (4 parça) ---")
    satir_suresi = sure_olc(histogram_4parca_satir_satir, resim)
    sutun_suresi = sure_olc(histogram_4parca_sutun_sutun, resim)
    hazir_suresi = sure_olc(histogram_4parca_hazir, resim)

    for a, b in zip(histogram_4parca_satir_satir(resim), histogram_4parca_sutun_sutun(resim)):
        assert np.array_equal(a, b)

    print(f"Satır satır  : {satir_suresi:.4f} s")
    print(f"Sütun sütun  : {sutun_suresi:.4f} s  (satır satıra göre {sutun_suresi / satir_suresi:.2f}x)")
    print(f"np.bincount  : {hazir_suresi:.6f} s  (satır satırdan {satir_suresi / hazir_suresi:.0f}x hızlı)")


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    resim = resmi_indir()
    print("Boyut:", resim.shape, "| Min:", resim.min(), "| Maks:", resim.max())

    # Deney 1: orijinal resmin 4 parçasının histogramı
    print("\n--- Deney 1: Orijinal resim histogramı ---")
    histogram_ciz_ve_kaydet("Orijinal resim - 4 parça histogram", "4parca_hist_0_orijinal.png",
                            resim, ["Orijinal"] * 4)

    # Deney 2: her parçada farklı gri seviye sayısı
    print("\n--- Deney 2: Nicemleme histogramı ---")
    for grup_no, seviyeler in enumerate(SEVIYE_GRUPLARI, start=1):
        sonuc = parcalara_uygula(resim, nicemle, seviyeler)
        yazilar = [f"{seviye} seviye ({int(np.log2(seviye))} bit)" for seviye in seviyeler]
        histogram_ciz_ve_kaydet(f"Gri seviye sayısı ({grup_no}/{len(SEVIYE_GRUPLARI)})",
                                f"4parca_hist_1_nicemleme_{grup_no}.png", sonuc, yazilar)

    # Deney 3: her parçada farklı maksimum parlaklık
    print("\n--- Deney 3: Maksimum parlaklık histogramı ---")
    sonuc = parcalara_uygula(resim, maks_parlakligi_degistir, MAKS_PARLAKLIKLAR)
    yazilar = [f"Maks = {maks}" for maks in MAKS_PARLAKLIKLAR]
    histogram_ciz_ve_kaydet("Maksimum parlaklık", "4parca_hist_2_maks_parlaklik.png", sonuc, yazilar)

    # Deney 4: histogram sayımının süresi
    histogram_sure_deneyi(resim)


if __name__ == "__main__":
    main()
