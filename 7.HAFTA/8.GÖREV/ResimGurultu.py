"""
Görüntü İşleme Ödevi - Resme Gauss ve Tuz-Biber Gürültüsü Ekleme ve Harmonik Matrisleri
---------------------------------------------------------------------------------------
gauss_gurultu_ekle(resim, sigma, tohum=None)
    Her piksele bağımsız N(0, sigma^2) gürültü eklenir, sonuç 0..255'e kırpılır.
    sigma: gürültünün kuvveti (gri seviye cinsinden standart sapma).

tuz_biber_ekle(resim, oran, tuz_payi=0.5, tohum=None)
    Piksellerin 'oran' kadarı bozulur. Bozulanların 'tuz_payi' kadarı beyaz
    (255, tuz), gerisi siyah (0, biber) yapılır.
    oran: gürültü oranı [0, 1]; tuz_payi = 0.5 -> yarı tuz yarı biber.

Her iki resim için 7. görevdeki dft2 ile harmonik matrisi hesaplanır ve
orijinal resmin harmonik matrisiyle karşılaştırılır:
  - log(1 + |F|) görüntüleri yan yana
  - log genlik farkı: log|F_gürültülü| - log|F_orijinal|
  - radyal ortalama: merkeze aynı uzaklıktaki (aynı frekanstaki) harmoniklerin
    ortalama genliği, frekansa göre log ölçekte çizilir.

Beklenen: gürültü pikselden piksele bağımsız (beyaz) olduğu için harmonik
matrisinde BÜTÜN frekanslara eşit ve düz bir taban ekler. Doğal resmin
genliği frekansla hızla düştüğünden bu taban, düşük frekanslarda resmin
altında kalır ama yüksek frekanslarda resmin üstüne çıkar ve onu örter.
Beyaz gürültünün harmonik genliği yaklaşık sigma * sqrt(M N) olmalıdır.

Kullanım:
    python ResimGurultu.py [resim_yolu]
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

sys.path.insert(0, os.path.join(KLASOR, "..", "7.GÖREV"))
from DFT2D import KAYNAK_RESIM, dft2, gri_oku, log_genlik, merkeze_al  # noqa: E402


# ---------------------------------------------------------------------------
# Gürültü fonksiyonları
# ---------------------------------------------------------------------------
def gauss_gurultu_ekle(resim, sigma, tohum=None):
    """Her piksele N(0, sigma^2) gürültü ekler, 0..255'e kırpar."""
    if sigma < 0:
        raise ValueError("sigma negatif olamaz")
    uretec = np.random.default_rng(tohum)
    gurultulu = resim.astype(np.float64) + uretec.normal(0.0, sigma, resim.shape)
    return np.clip(np.rint(gurultulu), 0, 255).astype(np.uint8)


def tuz_biber_ekle(resim, oran, tuz_payi=0.5, tohum=None):
    """Piksellerin 'oran' kadarını rastgele 255 (tuz) ya da 0 (biber) yapar."""
    if not (0 <= oran <= 1 and 0 <= tuz_payi <= 1):
        raise ValueError("oran ve tuz_payi [0, 1] aralığında olmalı")
    uretec = np.random.default_rng(tohum)
    sonuc = resim.copy()
    zar = uretec.random(resim.shape)
    bozulan = zar < oran
    tuz = bozulan & (uretec.random(resim.shape) < tuz_payi)
    sonuc[bozulan] = 0
    sonuc[tuz] = 255
    return sonuc


# ---------------------------------------------------------------------------
# Spektrum analizi
# ---------------------------------------------------------------------------
def radyal_ortalama(F):
    """Merkeze uzaklığa (tam sayıya yuvarlanmış) göre ortalama |F|."""
    G = np.abs(merkeze_al(F))
    M, N = G.shape
    yy, xx = np.mgrid[0:M, 0:N]
    r = np.rint(np.hypot(yy - M // 2, xx - N // 2)).astype(int)
    toplam = np.bincount(r.ravel(), G.ravel())
    adet = np.bincount(r.ravel())
    sinir = min(M, N) // 2
    return toplam[:sinir] / adet[:sinir]


def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    M, N = resim.shape

    resimler = [("Orijinal", resim),
                ("Gauss sigma=15", gauss_gurultu_ekle(resim, 15, tohum=1)),
                ("Gauss sigma=40", gauss_gurultu_ekle(resim, 40, tohum=1)),
                ("Tuz-biber %5", tuz_biber_ekle(resim, 0.05, tohum=1)),
                ("Tuz-biber %20", tuz_biber_ekle(resim, 0.20, tohum=1))]

    F_orijinal = dft2(resim)
    r_orijinal = radyal_ortalama(F_orijinal)
    print(f"{'Resim':<16}{'ort. parlaklık':>15}{'eklenen std':>12}{'DC / MN':>9}"
          f"{'r 5-20':>10}{'r 100-200':>11}{'beklenen taban':>16}")
    analizler = []
    for ad, r in resimler:
        F = dft2(r)
        radyal = radyal_ortalama(F)
        eklenen = r.astype(np.float64) - resim
        # Beyaz gürültünün harmonik genliği ~ std * sqrt(MN) * sqrt(pi)/2 (Rayleigh ortalaması)
        beklenen = eklenen.std() * np.sqrt(M * N) * np.sqrt(np.pi) / 2
        print(f"{ad:<16}{r.mean():>15.2f}{eklenen.std():>12.1f}{F[0, 0].real / (M * N):>9.2f}"
              f"{radyal[5:21].mean():>10.0f}{radyal[100:201].mean():>11.0f}"
              f"{(beklenen if eklenen.std() > 0 else 0):>16.0f}")
        analizler.append((ad, r, F, radyal))

    # --- Resimler, harmonik matrisleri ve farklar ---
    fig, eksenler = plt.subplots(3, len(analizler), figsize=(4 * len(analizler), 12))
    alt, ust = np.percentile(log_genlik(F_orijinal), [1, 99.9])
    for i, (ad, r, F, _) in enumerate(analizler):
        eksenler[0, i].imshow(r, cmap="gray", vmin=0, vmax=255)
        eksenler[0, i].set_title(ad, fontsize=11)
        eksenler[1, i].imshow(log_genlik(F), cmap="inferno", vmin=alt, vmax=ust)
        eksenler[1, i].set_title("log(1 + |F|)", fontsize=10)
        fark = log_genlik(F) - log_genlik(F_orijinal)
        goruntu = eksenler[2, i].imshow(fark, cmap="RdBu_r", vmin=-4, vmax=4)
        eksenler[2, i].set_title("log|F| - log|F_orijinal|", fontsize=10)
    for ax in eksenler.ravel():
        ax.axis("off")
    fig.colorbar(goruntu, ax=eksenler[2, :], fraction=0.015, pad=0.01)
    yol_kayit = os.path.join(CIKTI_KLASORU, "gurultu_harmonik_matrisleri.png")
    fig.savefig(yol_kayit, dpi=70, bbox_inches="tight")
    print("Kaydedildi:", yol_kayit)

    # --- Radyal ortalama ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    renkler = ["black", "tab:blue", "tab:purple", "tab:orange", "tab:red"]
    for (ad, _, _, radyal), renk in zip(analizler, renkler):
        ax.semilogy(np.arange(1, len(radyal)), radyal[1:], color=renk, label=ad,
                    linewidth=1.8 if ad == "Orijinal" else 1.1)
    ax.set_xlabel("merkeze uzaklık (frekans, harmonik)")
    ax.set_ylabel("ortalama |F| (log)")
    ax.set_title("Radyal ortalama genlik: gürültü yüksek frekanslara düz bir taban ekliyor")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "radyal_spektrum.png")
    fig.savefig(yol_kayit, dpi=90)
    print("Kaydedildi:", yol_kayit)
    plt.show()


if __name__ == "__main__":
    main()
