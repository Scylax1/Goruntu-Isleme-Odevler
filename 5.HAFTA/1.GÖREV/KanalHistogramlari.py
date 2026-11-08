"""
Görüntü İşleme Ödevi - Renkli Görüntünün Kanal Histogramları
------------------------------------------------------------
Renkli bir görüntüyü OpenCV ile okur ve her renk kanalı için ayrı ayrı
histogram çıkarır.

  - cv2.imread renkli resmi B, G, R sırasıyla (BGR) okur. Kanal 0 MAVİ,
    kanal 1 YEŞİL, kanal 2 KIRMIZI'dır. Matplotlib RGB beklediği için resim
    gösterilirken cv2.cvtColor(..., cv2.COLOR_BGR2RGB) ile çevrilir.
  - Her kanalın histogramı cv2.calcHist ile hesaplanır:

        cv2.calcHist([resim], [kanal], None, [256], [0, 256])

    [resim]   : resim listesi
    [kanal]   : histogramı çıkarılacak kanal indeksi (0: B, 1: G, 2: R)
    None      : maske yok, bütün pikseller sayılır
    [256]     : kutu (bin) sayısı, her gri seviye için bir kutu
    [0, 256]  : değer aralığı (üst sınır dahil değil)

  - Her kanal ayrıca tek kanallı gri resim olarak gösterilir: bir kanalın
    parlak olduğu yerler o rengin baskın olduğu yerlerdir.

Varsayılan girdi OpenCV'nin örnek resimlerinden baboon.jpg (512x512).

Kullanım:
    python KanalHistogramlari.py [resim_yolu]
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
KAYNAK_RESIM = os.path.join(CIKTI_KLASORU, "baboon.jpg")

# OpenCV kanal sırası: B, G, R
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
# Histogram
# ---------------------------------------------------------------------------
def kanal_histogramlari(resim):
    """Her kanal için cv2.calcHist ile 256 kutulu histogram (256,) dizisi."""
    return [cv2.calcHist([resim], [kanal], None, [256], [0, 256]).ravel()
            for kanal, _, _ in KANALLAR]


def istatistik_yazdir(resim, histogramlar):
    toplam = resim.shape[0] * resim.shape[1]
    print(f"{'Kanal':<12}{'Toplam':>9}{'Ort.':>8}{'Std':>8}{'Min':>6}{'Max':>6}{'En sık':>8}")
    for (kanal, ad, _), hist in zip(KANALLAR, histogramlar):
        degerler = resim[:, :, kanal]
        dolu = np.nonzero(hist)[0]
        print(f"{ad:<12}{int(hist.sum()):>9}{degerler.mean():>8.1f}{degerler.std():>8.1f}"
              f"{dolu[0]:>6}{dolu[-1]:>6}{int(np.argmax(hist)):>8}")
        # histogram toplamı piksel sayısına eşit olmalı
        assert hist.sum() == toplam


# ---------------------------------------------------------------------------
# Çizim
# ---------------------------------------------------------------------------
def ciz_ve_kaydet(resim, histogramlar):
    fig = plt.figure(figsize=(16, 8))
    izgara = fig.add_gridspec(2, 4)

    ax = fig.add_subplot(izgara[0, 0])
    ax.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    ax.set_title("Orijinal (BGR -> RGB)")
    ax.axis("off")

    for i, ((kanal, ad, renk), hist) in enumerate(zip(KANALLAR, histogramlar)):
        # Kanalın kendisi, tek kanallı gri resim olarak
        ax = fig.add_subplot(izgara[0, i + 1])
        ax.imshow(resim[:, :, kanal], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{ad} kanalı")
        ax.axis("off")

        # O kanalın histogramı
        ax = fig.add_subplot(izgara[1, i])
        ax.bar(np.arange(256), hist, width=1.0, color=renk)
        ax.set_xlim(0, 255)
        ax.set_title(f"{ad} histogramı")
        ax.set_xlabel("Parlaklık (0-255)")
        ax.set_ylabel("Piksel sayısı")

    # Üç histogram üst üste
    ax = fig.add_subplot(izgara[1, 3])
    for (kanal, ad, renk), hist in zip(KANALLAR, histogramlar):
        ax.plot(np.arange(256), hist, color=renk, label=ad, linewidth=1.2)
    ax.set_xlim(0, 255)
    ax.set_title("Üç kanal birlikte")
    ax.set_xlabel("Parlaklık (0-255)")
    ax.legend()

    fig.tight_layout()
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = os.path.join(CIKTI_KLASORU, "kanal_histogramlari.png")
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
    print("Resim:", yol, "| Boyut:", resim.shape, "| Tür:", resim.dtype)

    histogramlar = kanal_histogramlari(resim)
    istatistik_yazdir(resim, histogramlar)
    ciz_ve_kaydet(resim, histogramlar)


if __name__ == "__main__":
    main()
