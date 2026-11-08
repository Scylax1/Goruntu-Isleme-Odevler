"""
Sinyal İşleme Ödevi - Eşik Tabanlı Frekans Filtresi
---------------------------------------------------
frekans_filtresi(vektor, esik) -> (yeniden üretilmiş sinyal, filtrelenmiş X, maske)

  1) X = dft(vektor)                         (4. görevdeki DFT)
  2) Her harmoniğin genliği: 2 |X[k]| / N   (DC ve Nyquist'te |X[k]| / N),
     yani o harmoniğin sinyaldeki sinüs genliği.
  3) Genliği esik'in altında kalan harmonikler sıfırlanır.
     Gerçel sinyalde k ve N-k aynı harmoniktir ve genlikleri eşittir; ikisi
     birlikte tutulur ya da birlikte atılır. Böylece ters dönüşüm gerçel kalır.
  4) x' = idft(X_filtreli)                   (4. görevdeki ters DFT)

Neden işe yarar? 4. görevde görüldüğü gibi beyaz gürültü bütün harmoniklere
küçük ve düz bir taban olarak dağılır (sigma = 3.6 için ~0.3), sinüsler ise
tek bir harmoniğe toplanır (12 ve 5). Eşik tabanın üstünde, sinüslerin
altında seçilirse yalnızca sinüslerin harmonikleri kalır. Gürültünün o
harmoniklere düşen küçük payı dışındaki her şey silinir.

Seçilen eşik 2.0'dır: gürültü tabanının en büyük değeri ~0.9, en zayıf
sinüs 5. Ayrıca eşik taranıp hatanın eşiğe göre değişimi çizilir.
Eşik 5'in üstüne çıkınca toplam sinyaldeki 30 Hz bileşeni de silinir:
aynı filtre, iki sinyali birbirinden ayırmak için de kullanılabilir.

Test sinyalleri 5. görevdekilerle aynıdır (aynı tohumlar).
Başarı, gürültüsüz sinyale olan RMSE ile ölçülür.

Kullanım:
    python FrekansFiltresi.py
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

for gorev in ("1.GÖREV", "2.GÖREV", "4.GÖREV", "5.GÖREV"):
    sys.path.insert(0, os.path.join(KLASOR, "..", gorev))
from DFT import dft, genlik_spektrumu, idft  # noqa: E402
from IkiSinyalDFT import N, SINYALLER, sinyal_uret  # noqa: E402

ESIK = 2.0
TARAMA = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 4.6, 4.7, 6.0, 9.0, 12.0, 12.5]


# ---------------------------------------------------------------------------
# Filtre
# ---------------------------------------------------------------------------
def harmonik_genlikleri(harmonikler):
    """Her k için o harmoniğin sinüs genliği (k ve N-k aynı değeri alır)."""
    n_ornek = len(harmonikler)
    genlik = 2.0 * np.abs(harmonikler) / n_ornek
    genlik[0] /= 2.0
    genlik[n_ornek // 2] /= 2.0
    return genlik


def frekans_filtresi(vektor, esik):
    """Genliği esik'in altındaki harmonikleri silip sinyali yeniden üretir."""
    harmonikler = dft(vektor)
    maske = harmonik_genlikleri(harmonikler) >= esik
    filtreli = np.where(maske, harmonikler, 0)
    return idft(filtreli), filtreli, maske


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def kalan_frekanslar(maske):
    return [int(k) for k in np.nonzero(maske[: len(maske) // 2 + 1])[0]]


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    (temiz1, x1), (temiz2, x2) = [sinyal_uret(*s) for s in SINYALLER]
    testler = [("Sinyal 1 (50 Hz, A=12)", x1, temiz1),
               ("Sinyal 2 (30 Hz, A=5)", x2, temiz2),
               ("Toplam (50 + 30 Hz)", x1 + x2, temiz1 + temiz2)]

    print(f"Eşik = {ESIK}")
    print(f"{'Sinyal':<26}{'kalan harmonikler':<20}{'RMSE önce':>10}{'RMSE sonra':>11}{'iyileşme':>10}")
    sonuclar = []
    for ad, x, temiz in testler:
        yeniden, filtreli, maske = frekans_filtresi(x, ESIK)
        once, sonra = rmse(x, temiz), rmse(yeniden, temiz)
        print(f"{ad:<26}{str(kalan_frekanslar(maske)):<20}{once:>10.3f}{sonra:>11.3f}{once / sonra:>9.1f}x")
        sonuclar.append((ad, x, temiz, yeniden, filtreli, maske))

    # Eşik taraması (toplam sinyal)
    ad, x, temiz = testler[2]
    X = dft(x)
    genlik = harmonik_genlikleri(X)
    print(f"\nEşik taraması ({ad}):")
    print(f"{'eşik':>6}{'kalan k':>22}{'RMSE (gürültüsüz toplama)':>28}")
    tarama_hatalari = []
    for esik in TARAMA:
        maske = genlik >= esik
        yeniden = idft(np.where(maske, X, 0))
        hata = rmse(yeniden, temiz)
        tarama_hatalari.append(hata)
        kalan = kalan_frekanslar(maske)
        kalan_yazi = str(kalan) if len(kalan) <= 4 else f"{len(kalan)} harmonik"
        print(f"{esik:>6}{kalan_yazi:>22}{hata:>28.3f}")

    # Eşik > 5: yalnızca 50 Hz kalır -> sinyal 1'in gürültüsüz hâline yakınlık
    yalniz_50, _, _ = frekans_filtresi(x, 6.0)
    print(f"\nToplam sinyal eşik 6 ile filtrelenince -> sinyal 1'in gürültüsüz hâline RMSE: "
          f"{rmse(yalniz_50, temiz1):.3f} (30 Hz bileşeni ayrıldı)")

    # --- Çizimler ---
    t = np.arange(N) / N
    k = np.arange(N // 2 + 1)
    fig, eksenler = plt.subplots(4, 2, figsize=(16, 14), gridspec_kw={"width_ratios": [1.3, 1]})
    for satir, (ad, x, temiz, yeniden, filtreli, maske) in enumerate(sonuclar):
        ax = eksenler[satir, 0]
        ax.plot(t[:128], x[:128], ".", color="tab:gray", markersize=3, label=f"gürültülü (RMSE {rmse(x, temiz):.2f})")
        ax.plot(t[:128], temiz[:128], "k", linewidth=1.5, label="gürültüsüz")
        ax.plot(t[:128], yeniden[:128], "--", color="tab:red", linewidth=1.2,
                label=f"filtrelenmiş (RMSE {rmse(yeniden, temiz):.2f})")
        ax.set_title(f"{ad}: ilk 0.25 s", fontsize=10)
        ax.set_ylim(-30, 30)
        ax.legend(fontsize=8, loc="upper right")

        ax = eksenler[satir, 1]
        g = genlik_spektrumu(dft(x))
        ax.vlines(k, 0, g, color="tab:gray", linewidth=0.7, label="atılan")
        tutulan = kalan_frekanslar(maske)
        ax.vlines(tutulan, 0, g[tutulan], color="tab:red", linewidth=2, label="tutulan")
        ax.axhline(ESIK, color="tab:blue", linestyle="--", linewidth=1, label=f"eşik {ESIK}")
        ax.set_xlim(0, N // 2)
        ax.set_ylim(0, 14)
        ax.set_title(f"{ad}: harmonikler", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    ax = eksenler[3, 0]
    ax.plot(t[:128], (x1 + x2)[:128], ".", color="tab:gray", markersize=3, label="gürültülü toplam")
    ax.plot(t[:128], temiz1[:128], "k", linewidth=1.5, label="sinyal 1 (gürültüsüz)")
    ax.plot(t[:128], yalniz_50[:128], "--", color="tab:red", linewidth=1.2, label="toplam, eşik 6 ile")
    ax.set_title("Eşik 6: toplamdan yalnızca 50 Hz kalır (30 Hz ayrılır)", fontsize=10)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("zaman (s)")
    ax.legend(fontsize=8, loc="upper right")

    ax = eksenler[3, 1]
    ax.semilogx(TARAMA, tarama_hatalari, "o-", color="tab:purple")
    ax.axhline(rmse(x1 + x2, temiz1 + temiz2), color="tab:gray", linestyle="--", label="filtresiz")
    ax.axvline(ESIK, color="tab:blue", linestyle="--", linewidth=1, label=f"seçilen eşik {ESIK}")
    ax.set_xlabel("eşik (log)")
    ax.set_ylabel("RMSE")
    ax.set_title("Toplam sinyal: eşiğe göre hata", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "frekans_filtresi.png")
    fig.savefig(yol, dpi=85)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
