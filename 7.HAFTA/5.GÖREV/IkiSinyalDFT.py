"""
Sinyal İşleme Ödevi - İki Gürültülü Sinyalin ve Toplamlarının DFT Analizi
-------------------------------------------------------------------------
  Sinyal 1: f = 50 Hz, A = 12, gürültü eklenmiş
  Sinyal 2: f = 30 Hz, A = 5,  gürültü eklenmiş
  Toplam  : sinyal 1 + sinyal 2

Önceki görevlerdeki fonksiyonlar kullanılır:
  sinus_doldur (1. görev), gurultu_ekle (2. görev), dft ve
  genlik_spektrumu (4. görev).

N = 512 örnek, fs = 512 Hz (1 saniye). Her iki frekans da tam sayı
olduğundan harmonikler tam k = 30 ve k = 50'ye düşer, sızıntı olmaz.

Gürültü oranı her iki sinyalde 0.3'tür; gurultu_ekle gürültüyü genliğe göre
ölçeklediği için sinyal 1'e sigma = 3.6, sinyal 2'ye sigma = 1.5 gürültü
eklenir. Toplamdaki gürültü bu iki bağımsız gürültünün toplamıdır:
sigma = sqrt(3.6^2 + 1.5^2) = 3.9.

DFT doğrusaldır: DFT(x1 + x2) = DFT(x1) + DFT(x2). Toplam sinyalin harmonik
vektörü, iki sinyalin harmonik vektörlerinin toplamına eşit çıkmalıdır;
program bunu sayısal olarak da kontrol eder.

Kullanım:
    python IkiSinyalDFT.py
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

for gorev in ("1.GÖREV", "2.GÖREV", "4.GÖREV"):
    sys.path.insert(0, os.path.join(KLASOR, "..", gorev))
from SinusSinyal import sinus_doldur  # noqa: E402
from GurultuEkle import gurultu_ekle  # noqa: E402
from DFT import dft, genlik_spektrumu  # noqa: E402

N = 512
GURULTU_ORANI = 0.3
SINYALLER = [(50, 12.0, 42), (30, 5.0, 7)]    # (frekans, genlik, gürültü tohumu)


def sinyal_uret(frekans, genlik, tohum):
    temiz = sinus_doldur(np.zeros(N), genlik, 0.0, frekans=frekans)
    return temiz, gurultu_ekle(temiz, GURULTU_ORANI, tohum=tohum)


def tepeler(genlik, adet=3):
    sirali = np.argsort(genlik[1:])[::-1][:adet] + 1
    return ", ".join(f"k={k}: {genlik[k]:.2f}" for k in sirali)


def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    t = np.arange(N) / N
    k = np.arange(N // 2 + 1)

    (temiz1, x1), (temiz2, x2) = [sinyal_uret(*s) for s in SINYALLER]
    toplam = x1 + x2
    toplam_temiz = temiz1 + temiz2

    X1, X2, X_toplam = dft(x1), dft(x2), dft(toplam)
    g1, g2, g_toplam = genlik_spektrumu(X1), genlik_spektrumu(X2), genlik_spektrumu(X_toplam)

    print(f"Sinyal 1 (50 Hz, A=12, gürültü sigma {np.std(x1 - temiz1):.2f}): en büyük harmonikler {tepeler(g1)}")
    print(f"Sinyal 2 (30 Hz, A=5,  gürültü sigma {np.std(x2 - temiz2):.2f}): en büyük harmonikler {tepeler(g2)}")
    print(f"Toplam   (gürültü sigma {np.std(toplam - toplam_temiz):.2f}): en büyük harmonikler {tepeler(g_toplam)}")
    taban = np.delete(g_toplam[1:], [29, 49])
    print(f"Toplamın gürültü tabanı: ortalama {taban.mean():.3f}, en büyük {taban.max():.3f}")
    print(f"Doğrusallık: |DFT(x1 + x2) - (DFT(x1) + DFT(x2))| en büyük = {np.abs(X_toplam - (X1 + X2)).max():.2e}")

    satirlar = [("Sinyal 1: 50 Hz, A = 12 (+ gürültü)", x1, temiz1, g1, "tab:blue"),
                ("Sinyal 2: 30 Hz, A = 5 (+ gürültü)", x2, temiz2, g2, "tab:green"),
                ("Toplam: sinyal 1 + sinyal 2", toplam, toplam_temiz, g_toplam, "tab:purple")]
    fig, eksenler = plt.subplots(3, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [1.3, 1]})
    for satir, (ad, x, temiz, g, renk) in enumerate(satirlar):
        ax = eksenler[satir, 0]
        ax.plot(t[:128], x[:128], ".-", color=renk, markersize=3, linewidth=0.7, label="gürültülü")
        ax.plot(t[:128], temiz[:128], "k", linewidth=1, label="gürültüsüz")
        ax.set_title(f"{ad}: ilk 0.25 s", fontsize=10)
        ax.set_ylim(-30, 30)
        ax.legend(fontsize=8, loc="upper right")
        ax = eksenler[satir, 1]
        isaret, sap, taban_cizgi = ax.stem(k, g)
        plt.setp(sap, linewidth=0.7, color=renk)
        plt.setp(isaret, markersize=2, color=renk)
        plt.setp(taban_cizgi, linewidth=0.4)
        for f, a, _ in SINYALLER:
            if g[f] > 2:
                ax.annotate(f"{f} Hz: {g[f]:.2f}", (f, g[f]), textcoords="offset points",
                            xytext=(8, -4), fontsize=9)
        ax.set_xlim(0, N // 2)
        ax.set_ylim(0, 14)
        ax.set_title(f"{ad}: harmonik vektörü (genlik)", fontsize=10)
    eksenler[-1, 0].set_xlabel("zaman (s)")
    eksenler[-1, 1].set_xlabel("harmonik k (Hz)")
    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "iki_sinyal_dft.png")
    fig.savefig(yol, dpi=90)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
