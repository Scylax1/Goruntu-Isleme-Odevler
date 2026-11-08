"""
Görüntü İşleme Ödevi - Ters DFT ile Gürültülü Resimden Orijinale Dönme ve ZNCC
------------------------------------------------------------------------------
8. görevdeki gürültülü resimler (Gauss sigma = 25, tuz-biber %10) üzerinde:

  A) Doğrudan ters DFT: idft2(dft2(gürültülü)).
     DFT kayıpsız ve tersinir bir dönüşümdür; harmonik matrisinde hiçbir
     şey değiştirilmezse ters DFT gürültülü resmin AYNISINI verir.
     Gürültü kendiliğinden gitmez.

  B) Frekans uzayında filtreleyip ters DFT:
        F = dft2(gürültülü)
        F' = F * H,   H(u, v) = exp(-D^2 / (2 D0^2))   (Gauss alçak geçiren)
        yeniden = idft2(F')
     D: harmoniğin merkeze uzaklığı. 8. görevde görüldüğü gibi resmin
     enerjisi düşük frekanslarda, gürültünün enerjisi ise her yerde düz.
     Yüksek frekansları bastırmak, gürültünün çoğunu resmin küçük bir
     kısmı pahasına atar. D0 taranır, ZNCC'yi en büyük yapan seçilir.

Benzerlik, Zero Mean Normalized Cross Correlation (ZNCC) ile ölçülür:

            sum (a - mean(a)) (b - mean(b))
  ZNCC = -------------------------------------------      -1 <= ZNCC <= 1
         sqrt( sum (a - mean(a))^2  sum (b - mean(b))^2 )

Karşılaştırma için diğer metrikler de elle hesaplanır:
  MSE, RMSE, MAE, PSNR, SAD, NCC (ortalama çıkarılmadan), SSIM.

Ayrıca metriklerin davranışını göstermek için orijinal resmin bilinen
bozulmaları da ölçülür: parlaklık +30, kontrast x0.5, 1 piksel kaydırma,
Gauss bulanıklaştırma, negatif. Böylece hangi metriğin neye duyarlı / neye
duyarsız olduğu sayılarla görülür.

Kullanım:
    python TersDFT_ZNCC.py [resim_yolu]
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

sys.path.insert(0, os.path.join(KLASOR, "..", "7.GÖREV"))
sys.path.insert(0, os.path.join(KLASOR, "..", "8.GÖREV"))
from DFT2D import KAYNAK_RESIM, dft2, gri_oku, idft2, merkezden_geri  # noqa: E402
from ResimGurultu import gauss_gurultu_ekle, tuz_biber_ekle  # noqa: E402

D0_TARAMA = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300]


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def zncc(a, b):
    a = np.asarray(a, np.float64) - np.mean(a)
    b = np.asarray(b, np.float64) - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def ncc(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, np.float64) - b)))


def sad(a, b):
    return float(np.sum(np.abs(np.asarray(a, np.float64) - b)))


def psnr(a, b, en_buyuk=255.0):
    hata = mse(a, b)
    return float("inf") if hata == 0 else float(10 * np.log10(en_buyuk ** 2 / hata))


def ssim(a, b, en_buyuk=255.0):
    """Yapısal benzerlik: 11x11, sigma 1.5 Gauss penceresiyle yerel parlaklık,
    kontrast ve yapı karşılaştırmasının ortalaması (Wang vd., 2004)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    c1, c2 = (0.01 * en_buyuk) ** 2, (0.03 * en_buyuk) ** 2
    pencere = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = pencere(a), pencere(b)
    var_a = pencere(a * a) - mu_a ** 2
    var_b = pencere(b * b) - mu_b ** 2
    kov = pencere(a * b) - mu_a * mu_b
    harita = ((2 * mu_a * mu_b + c1) * (2 * kov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(harita.mean())


METRIKLER = [("ZNCC", zncc, "{:.4f}"), ("NCC", ncc, "{:.4f}"), ("SSIM", ssim, "{:.4f}"),
             ("PSNR dB", psnr, "{:.2f}"), ("MSE", mse, "{:.1f}"), ("RMSE", lambda a, b: mse(a, b) ** 0.5, "{:.2f}"),
             ("MAE", mae, "{:.2f}"), ("SAD", sad, "{:.3g}")]


def metrik_satiri(ad, orijinal, resim):
    degerler = [bicim.format(fonksiyon(orijinal, resim)) for _, fonksiyon, bicim in METRIKLER]
    return f"{ad:<34}" + "".join(f"{d:>10}" for d in degerler)


def baslik_satiri():
    return f"{'':<34}" + "".join(f"{ad:>10}" for ad, _, _ in METRIKLER)


# ---------------------------------------------------------------------------
# Frekans uzayında Gauss alçak geçiren
# ---------------------------------------------------------------------------
def gauss_alcak_geciren(boyut, d0):
    """Merkezlenmemiş (DC köşede) Gauss transfer fonksiyonu H(u, v)."""
    M, N = boyut
    yy, xx = np.mgrid[0:M, 0:N]
    D = np.hypot(yy - M // 2, xx - N // 2)
    return merkezden_geri(np.exp(-(D ** 2) / (2.0 * d0 ** 2)))


def dft_ile_temizle(resim, d0, F=None):
    F = dft2(resim) if F is None else F
    return idft2(F * gauss_alcak_geciren(resim.shape, d0))


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    orijinal = gri_oku(yol)
    if orijinal is None:
        sys.exit(f"Resim açılamadı: {yol}")
    orijinal_f = orijinal.astype(np.float64)

    gurultululer = [("Gauss sigma=25", gauss_gurultu_ekle(orijinal, 25, tohum=1)),
                    ("Tuz-biber %10", tuz_biber_ekle(orijinal, 0.10, tohum=1))]

    sonuclar = []
    for ad, gurultulu in gurultululer:
        print(f"\n=== {ad} ===")
        F = dft2(gurultulu)
        dogrudan = idft2(F)
        print(f"A) Doğrudan ters DFT: gürültülü resimle en büyük fark {np.abs(dogrudan - gurultulu).max():.1e}")

        tarama = [(d0, zncc(orijinal_f, dft_ile_temizle(gurultulu, d0, F))) for d0 in D0_TARAMA]
        print("B) Gauss alçak geçiren, D0 taraması (ZNCC): " +
              ", ".join(f"{d0}: {z:.4f}" for d0, z in tarama))
        en_iyi_d0 = max(tarama, key=lambda x: x[1])[0]
        temiz = dft_ile_temizle(gurultulu, en_iyi_d0, F)
        medyan = cv2.medianBlur(gurultulu, 3)

        print(baslik_satiri())
        print(metrik_satiri("Gürültülü", orijinal_f, gurultulu))
        print(metrik_satiri("A) idft2(dft2(gürültülü))", orijinal_f, dogrudan))
        print(metrik_satiri(f"B) LP D0={en_iyi_d0} + idft2", orijinal_f, temiz))
        print(metrik_satiri("(uzamsal medyan 3x3, kıyas)", orijinal_f, medyan))
        sonuclar.append((ad, gurultulu, dogrudan, temiz, en_iyi_d0, tarama, medyan))

    # --- Metriklerin davranışı: bilinen bozulmalar ---
    bozulmalar = [("Aynı resim", orijinal_f),
                  ("Parlaklık +30", np.clip(orijinal_f + 30, 0, 255)),
                  ("Kontrast x0.5 (+64)", orijinal_f * 0.5 + 64),
                  ("1 piksel sağa kaydırma", np.roll(orijinal_f, 1, axis=1)),
                  ("Gauss bulanık sigma=2", cv2.GaussianBlur(orijinal_f, (0, 0), 2)),
                  ("Negatif (255 - f)", 255 - orijinal_f),
                  ("Gauss gürültü sigma=25", gauss_gurultu_ekle(orijinal, 25, tohum=1))]
    print("\nMetriklerin bilinen bozulmalara tepkisi (orijinale göre):")
    print(baslik_satiri())
    for ad, r in bozulmalar:
        print(metrik_satiri(ad, orijinal_f, r))

    # --- Çizim ---
    fig, eksenler = plt.subplots(len(sonuclar), 5, figsize=(22, 9))
    for satir, (ad, gurultulu, dogrudan, temiz, d0, tarama, medyan) in enumerate(sonuclar):
        paneller = [(orijinal, "Orijinal"),
                    (gurultulu, f"{ad}\nZNCC {zncc(orijinal_f, gurultulu):.4f}"),
                    (dogrudan, f"A) idft2(dft2(gürültülü))\nZNCC {zncc(orijinal_f, dogrudan):.4f}"),
                    (temiz, f"B) Gauss LP D0={d0} + idft2\nZNCC {zncc(orijinal_f, temiz):.4f}")]
        for sutun, (r, baslik) in enumerate(paneller):
            eksenler[satir, sutun].imshow(np.clip(r, 0, 255), cmap="gray", vmin=0, vmax=255)
            eksenler[satir, sutun].set_title(baslik, fontsize=10)
            eksenler[satir, sutun].axis("off")
        ax = eksenler[satir, 4]
        ax.semilogx([d for d, _ in tarama], [z for _, z in tarama], "o-")
        ax.axhline(zncc(orijinal_f, gurultulu), color="tab:gray", linestyle="--", label="gürültülü")
        ax.axhline(zncc(orijinal_f, medyan), color="tab:green", linestyle=":", label="medyan 3x3 (uzamsal)")
        ax.set_xlabel("D0 (log)")
        ax.set_ylabel("ZNCC")
        ax.set_title(f"{ad}: D0'a göre ZNCC", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "ters_dft_zncc.png")
    fig.savefig(yol_kayit, dpi=75)
    print("Kaydedildi:", yol_kayit)
    plt.show()


if __name__ == "__main__":
    main()
