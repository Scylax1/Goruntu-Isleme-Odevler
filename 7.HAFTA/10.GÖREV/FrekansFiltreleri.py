"""
Görüntü İşleme Ödevi - Frekans Uzayında Low-pass, High-pass, Band-pass, Band-reject
-----------------------------------------------------------------------------------
Bir resme ve gürültülü hâllerine frekans uzayında dört filtre uygular:

    G = idft2( dft2(resim) * H )         (7. görevdeki dft2 / idft2)

D(u, v): harmoniğin merkeze (DC'ye) uzaklığı. Filtreler Gauss biçimlidir:

  Low-pass  (alçak geçiren) : H = exp(-D^2 / (2 D0^2)),               D0 = 40
  High-pass (yüksek geçiren): H = 1 - exp(-D^2 / (2 D0^2)),           D0 = 20
  Band-pass (bant geçiren)  : H = exp(-((D^2 - C^2) / (D W))^2),       C = 60, W = 20
  Band-reject (bant söndüren): H = 1 - band-pass                        C = 60, W = 20

Gauss biçimi seçildi çünkü "ideal" (keskin kesimli, 0/1) filtre, uzay
uzayında sinc fonksiyonuna karşılık gelir ve kenarların çevresinde
halkalanma (ringing) yaratır. Bunu göstermek için ideal alçak geçiren
(D <= 40) de uygulanır.

Resimler:
  - Orijinal (cameraman)
  - Gauss gürültüsü sigma = 25          (8. görevdeki fonksiyon)
  - Tuz-biber %10                       (8. görevdeki fonksiyon)
  - Periyodik gürültü: 40 * sin(2 pi (40 y + 45 x) / 512) eklenmiş.
    Bu gürültü harmonik matrisinde merkezden D = sqrt(40^2 + 45^2) = 60.2
    uzaklıktaki iki noktada toplanır; band-reject (C = 60) onu tam
    hedefler. Band-reject filtresinin asıl kullanım alanı budur.

Ölçülenler (her filtre / resim için):
  - Ortalama parlaklık: DC'nin (H(0,0)) korunup korunmadığı.
  - ZNCC(çıktı, orijinal): sonucun orijinal resme benzerliği.
  - ZNCC(çıktı, aynı filtrenin temiz resimdeki çıktısı): filtre çıktısının
    gürültüden ne kadar etkilendiği (1 = hiç etkilenmemiş).
  - Kenar enerjisi oranı: ortalama Sobel büyüklüğü / orijinalinki.
    Kenar geçişlerinin keskinleşip keskinleşmediğini (>1) ya da
    yumuşadığını (<1) gösterir.

High-pass ve band-pass DC'yi (H(0,0) = 0) attığı için çıktıların
ortalaması ~0'dır ve negatif değerler içerir; görselleştirmede +128
kaydırılarak gösterilir.

Kullanım:
    python FrekansFiltreleri.py [resim_yolu]
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

for gorev in ("7.GÖREV", "8.GÖREV", "9.GÖREV"):
    sys.path.insert(0, os.path.join(KLASOR, "..", gorev))
from DFT2D import KAYNAK_RESIM, dft2, gri_oku, idft2, log_genlik, merkezden_geri  # noqa: E402
from ResimGurultu import gauss_gurultu_ekle, tuz_biber_ekle  # noqa: E402
from TersDFT_ZNCC import zncc  # noqa: E402

LP_D0 = 40
HP_D0 = 20
BANT_MERKEZ, BANT_GENISLIK = 60, 20
PERIYODIK = (40, 45, 40.0)       # (u0, v0, genlik)


# ---------------------------------------------------------------------------
# Filtreler (merkezlenmiş tasarlanır, DC köşeye geri taşınır)
# ---------------------------------------------------------------------------
def uzaklik_matrisi(boyut):
    M, N = boyut
    yy, xx = np.mgrid[0:M, 0:N]
    return np.hypot(yy - M // 2, xx - N // 2)


def alcak_geciren(boyut, d0):
    D = uzaklik_matrisi(boyut)
    return np.exp(-(D ** 2) / (2.0 * d0 ** 2))


def yuksek_geciren(boyut, d0):
    return 1.0 - alcak_geciren(boyut, d0)


def bant_geciren(boyut, merkez, genislik):
    D = uzaklik_matrisi(boyut)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.exp(-(((D ** 2 - merkez ** 2) / (D * genislik)) ** 2))
    H[D == 0] = 0.0                      # DC: band-pass'ta tamamen kapalı
    return H


def bant_sondiren(boyut, merkez, genislik):
    return 1.0 - bant_geciren(boyut, merkez, genislik)


def ideal_alcak_geciren(boyut, d0):
    return (uzaklik_matrisi(boyut) <= d0).astype(np.float64)


def filtrele(resim, H_merkezli, F=None):
    F = dft2(resim) if F is None else F
    return idft2(F * merkezden_geri(H_merkezli))


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def periyodik_gurultu_ekle(resim, u0, v0, genlik):
    M, N = resim.shape
    yy, xx = np.mgrid[0:M, 0:N]
    desen = genlik * np.sin(2 * np.pi * (u0 * yy / M + v0 * xx / N))
    return np.clip(np.rint(resim + desen), 0, 255).astype(np.uint8)


def kenar_enerjisi(resim):
    r = np.asarray(resim, np.float64)
    gx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


def gosterim(r, dc_yok):
    return np.clip(r + 128 if dc_yok else r, 0, 255)


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    orijinal = gri_oku(yol)
    if orijinal is None:
        sys.exit(f"Resim açılamadı: {yol}")
    boyut = orijinal.shape

    filtreler = [("Low-pass", alcak_geciren(boyut, LP_D0), False),
                 ("High-pass", yuksek_geciren(boyut, HP_D0), True),
                 ("Band-pass", bant_geciren(boyut, BANT_MERKEZ, BANT_GENISLIK), True),
                 ("Band-reject", bant_sondiren(boyut, BANT_MERKEZ, BANT_GENISLIK), False),
                 ("İdeal low-pass", ideal_alcak_geciren(boyut, LP_D0), False)]
    resimler = [("Orijinal", orijinal),
                ("Gauss sigma=25", gauss_gurultu_ekle(orijinal, 25, tohum=1)),
                ("Tuz-biber %10", tuz_biber_ekle(orijinal, 0.10, tohum=1)),
                ("Periyodik gürültü", periyodik_gurultu_ekle(orijinal, *PERIYODIK))]

    # Temiz resmin her filtredeki çıktısı (gürültü etkisini ölçmek için referans)
    F_orijinal = dft2(orijinal)
    temiz_ciktilar = {ad: filtrele(orijinal, H, F_orijinal) for ad, H, _ in filtreler}
    kenar_orijinal = kenar_enerjisi(orijinal)

    print(f"Orijinal: ortalama {orijinal.mean():.1f}, kenar enerjisi {kenar_orijinal:.1f}")
    print(f"\n{'Resim':<20}{'Filtre':<16}{'ortalama':>9}{'ZNCC/orij':>11}{'ZNCC/temiz-filtre':>19}{'kenar oranı':>13}")
    ciktilar = {}
    for r_ad, r in resimler:
        F = dft2(r)
        print(f"{r_ad:<20}{'(filtresiz)':<16}{r.mean():>9.1f}{zncc(orijinal, r):>11.4f}{'':>19}"
              f"{kenar_enerjisi(r) / kenar_orijinal:>13.2f}")
        for f_ad, H, _ in filtreler:
            cikti = filtrele(r, H, F)
            ciktilar[(r_ad, f_ad)] = cikti
            print(f"{'':<20}{f_ad:<16}{cikti.mean():>9.1f}{zncc(orijinal, cikti):>11.4f}"
                  f"{zncc(temiz_ciktilar[f_ad], cikti):>19.4f}{kenar_enerjisi(cikti) / kenar_orijinal:>13.2f}")

    # --- Filtrelerin transfer fonksiyonları ---
    fig, eksenler = plt.subplots(2, len(filtreler) + 1, figsize=(4 * (len(filtreler) + 1), 7.5))
    D = uzaklik_matrisi(boyut)
    merkez = boyut[0] // 2
    for i, (f_ad, H, _) in enumerate(filtreler):
        eksenler[0, i].imshow(H, cmap="gray", vmin=0, vmax=1)
        eksenler[0, i].set_title(f"{f_ad}: H(u, v)", fontsize=10)
        eksenler[0, i].axis("off")
        eksenler[1, i].plot(np.arange(merkez), H[merkez, merkez:merkez * 2])
        eksenler[1, i].set_ylim(-0.05, 1.05)
        eksenler[1, i].set_xlabel("D (merkeze uzaklık)")
        eksenler[1, i].set_title("kesit", fontsize=10)
        eksenler[1, i].axvline(np.hypot(*PERIYODIK[:2]), color="tab:red", linestyle=":", linewidth=1)
    periyodik = resimler[3][1]
    eksenler[0, -1].imshow(log_genlik(dft2(periyodik)), cmap="inferno")
    eksenler[0, -1].set_title("Periyodik gürültülü resmin\nharmonik matrisi", fontsize=10)
    eksenler[0, -1].axis("off")
    eksenler[1, -1].imshow(log_genlik(dft2(ciktilar[("Periyodik gürültü", "Band-reject")])), cmap="inferno")
    eksenler[1, -1].set_title("Band-reject sonrası", fontsize=10)
    eksenler[1, -1].axis("off")
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "filtre_transfer_fonksiyonlari.png")
    fig.savefig(yol_kayit, dpi=75)
    print("\nKaydedildi:", yol_kayit)

    # --- Bütün sonuçlar ---
    fig, eksenler = plt.subplots(len(resimler), len(filtreler) + 1,
                                 figsize=(3.6 * (len(filtreler) + 1), 3.9 * len(resimler)))
    for satir, (r_ad, r) in enumerate(resimler):
        eksenler[satir, 0].imshow(r, cmap="gray", vmin=0, vmax=255)
        eksenler[satir, 0].set_title(f"{r_ad}\nZNCC {zncc(orijinal, r):.3f}", fontsize=10)
        for sutun, (f_ad, _, dc_yok) in enumerate(filtreler, start=1):
            cikti = ciktilar[(r_ad, f_ad)]
            eksenler[satir, sutun].imshow(gosterim(cikti, dc_yok), cmap="gray", vmin=0, vmax=255)
            ek = " (+128)" if dc_yok else ""
            eksenler[satir, sutun].set_title(f"{f_ad}{ek}\nort. {cikti.mean():.0f}, "
                                             f"ZNCC/temiz-f {zncc(temiz_ciktilar[f_ad], cikti):.3f}",
                                             fontsize=9)
    for ax in eksenler.ravel():
        ax.axis("off")
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "frekans_filtreleri.png")
    fig.savefig(yol_kayit, dpi=65)
    print("Kaydedildi:", yol_kayit)

    # --- Yakın plan: halkalanma ve kenarlar ---
    kirp = (slice(60, 260), slice(140, 340))
    fig, eksenler = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (baslik, r) in zip(eksenler, [("Orijinal", orijinal),
                                         ("Gauss low-pass D0=40", ciktilar[("Orijinal", "Low-pass")]),
                                         ("İdeal low-pass D0=40: halkalanma", ciktilar[("Orijinal", "İdeal low-pass")]),
                                         ("High-pass D0=20 (+128)", ciktilar[("Orijinal", "High-pass")] + 128)]):
        ax.imshow(np.clip(r[kirp], 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(baslik, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "yakin_plan_halkalanma.png")
    fig.savefig(yol_kayit, dpi=85)
    print("Kaydedildi:", yol_kayit)
    plt.show()


if __name__ == "__main__":
    main()
