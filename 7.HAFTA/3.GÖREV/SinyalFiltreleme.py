"""
Sinyal İşleme Ödevi - Gürültülü Sinyali Gauss ve Medyan Filtreyle Temizleme
---------------------------------------------------------------------------
2. görevdeki gürültülü sinüsü sırasıyla Gauss ve medyan filtreyle temizler.
İki filtre de tek boyutlu olarak elle yazılmıştır.

  gauss_filtre(vektor, sigma)
      Çekirdek: g[k] = exp(-k^2 / (2 sigma^2)), k = -3sigma .. 3sigma,
      toplamı 1 olacak şekilde normalize. Sinyal bu çekirdekle konvolüsyona
      sokulur: y[n] = sum_k g[k] * x[n - k]. Doğrusal bir alçak geçiren
      filtredir: komşuların ağırlıklı ortalamasını alır.

  medyan_filtre(vektor, pencere)
      y[n] = x[n - p/2 .. n + p/2] penceresindeki değerlerin ortancası.
      Doğrusal değildir: tek bir aşırı değer (dürtü) ortancayı değiştirmez.

  Kenarlarda pencere sinyalin dışına taşınca sinyal ayna gibi yansıtılır
  (x[-1] = x[1], x[N] = x[N-2]); sıfırla doldurmak kenarlarda yapay bir
  düşüş yaratırdı.

Temizlik, temiz sinyale olan RMSE (karekök ortalama kare hata) ile ölçülür.
Her filtrenin genişliği taranır ve en iyi değer raporlanır.

İki gürültü türüyle denenir:
  1) Gauss gürültüsü (2. görevdeki gurultu_ekle, oran = 0.3)
  2) Dürtü (tuz-biber) gürültüsü: örneklerin %10'u rastgele +-3A'ya sıçrar.
     Medyan filtrenin asıl gücü bu tür gürültüdedir.

Kullanım:
    python SinyalFiltreleme.py
"""

import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

sys.path.insert(0, os.path.join(KLASOR, "..", "1.GÖREV"))
sys.path.insert(0, os.path.join(KLASOR, "..", "2.GÖREV"))
from SinusSinyal import sinus_doldur  # noqa: E402
from GurultuEkle import gurultu_ekle  # noqa: E402

N, A, F = 256, 5.0, 4
GURULTU_ORANI = 0.3
DURTU_ORANI = 0.10
SIGMALAR = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
PENCERELER = [3, 5, 7, 9, 11, 15, 21, 31]


# ---------------------------------------------------------------------------
# Filtreler
# ---------------------------------------------------------------------------
def yansit(n, uzunluk):
    """Sinyal dışına taşan indeksi ayna gibi içeri yansıtır."""
    while n < 0 or n >= uzunluk:
        n = -n if n < 0 else 2 * (uzunluk - 1) - n
    return n


def gauss_cekirdegi(sigma):
    yaricap = max(1, int(math.ceil(3 * sigma)))
    cekirdek = [math.exp(-k * k / (2 * sigma * sigma)) for k in range(-yaricap, yaricap + 1)]
    toplam = sum(cekirdek)
    return [c / toplam for c in cekirdek], yaricap


def gauss_filtre(vektor, sigma):
    cekirdek, yaricap = gauss_cekirdegi(sigma)
    uzunluk = len(vektor)
    sonuc = np.zeros(uzunluk)
    for n in range(uzunluk):
        toplam = 0.0
        for i, k in enumerate(range(-yaricap, yaricap + 1)):
            toplam += cekirdek[i] * vektor[yansit(n - k, uzunluk)]
        sonuc[n] = toplam
    return sonuc


def medyan_filtre(vektor, pencere):
    if pencere % 2 == 0:
        raise ValueError("pencere tek sayı olmalı")
    yari = pencere // 2
    uzunluk = len(vektor)
    sonuc = np.zeros(uzunluk)
    for n in range(uzunluk):
        degerler = sorted(vektor[yansit(n + k, uzunluk)] for k in range(-yari, yari + 1))
        sonuc[n] = degerler[yari]
    return sonuc


def durtu_ekle(vektor, oran, genlik, tohum=None):
    uretec = random.Random(tohum)
    sonuc = np.array(vektor, dtype=np.float64)
    for n in range(len(sonuc)):
        if uretec.random() < oran:
            sonuc[n] = genlik if uretec.random() < 0.5 else -genlik
    return sonuc


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def tara(baslik, temiz, gurultulu):
    print(f"\n{baslik}: gürültülü RMSE = {rmse(gurultulu, temiz):.3f}")
    gauss_hatalari = [rmse(gauss_filtre(gurultulu, s), temiz) for s in SIGMALAR]
    medyan_hatalari = [rmse(medyan_filtre(gurultulu, p), temiz) for p in PENCERELER]
    print("  Gauss  sigma : " + "  ".join(f"{s:>5}" for s in SIGMALAR))
    print("         RMSE  : " + "  ".join(f"{h:5.3f}" for h in gauss_hatalari))
    print("  Medyan pencere:" + "  ".join(f"{p:>5}" for p in PENCERELER))
    print("         RMSE  : " + "  ".join(f"{h:5.3f}" for h in medyan_hatalari))
    en_iyi_sigma = SIGMALAR[int(np.argmin(gauss_hatalari))]
    en_iyi_pencere = PENCERELER[int(np.argmin(medyan_hatalari))]
    print(f"  En iyi: Gauss sigma = {en_iyi_sigma} (RMSE {min(gauss_hatalari):.3f}), "
          f"medyan pencere = {en_iyi_pencere} (RMSE {min(medyan_hatalari):.3f})")
    return gauss_hatalari, medyan_hatalari, en_iyi_sigma, en_iyi_pencere


def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    temiz = sinus_doldur(np.zeros(N), A, 0.0, frekans=F)
    t = np.arange(N) / N

    gauss_gurultulu = gurultu_ekle(temiz, GURULTU_ORANI, tohum=42)
    durtulu = durtu_ekle(temiz, DURTU_ORANI, 3 * A, tohum=3)

    # Gauss filtresinin sinüsün genliğini ne kadar düşürdüğü (kuramsal kazanç)
    for s in (2, 8):
        kazanc = math.exp(-2 * (math.pi * F * s / N) ** 2)
        print(f"Gauss sigma = {s}: {F} Hz sinüsün genliği x{kazanc:.3f} olur")

    sonuclar = []
    for baslik, gurultulu in (("1) Gauss gürültüsü (oran 0.3)", gauss_gurultulu),
                              ("2) Dürtü gürültüsü (%10, +-3A)", durtulu)):
        gh, mh, s, p = tara(baslik, temiz, gurultulu)
        sonuclar.append((baslik, gurultulu, gh, mh, s, p))

    fig, eksenler = plt.subplots(3, 2, figsize=(15, 11))
    for sutun, (baslik, gurultulu, gh, mh, s, p) in enumerate(sonuclar):
        ax = eksenler[0, sutun]
        ax.plot(t, gurultulu, ".", color="tab:gray", markersize=3, label="gürültülü")
        ax.plot(t, temiz, "k", linewidth=1, label="temiz")
        ax.set_title(f"{baslik}  (RMSE {rmse(gurultulu, temiz):.2f})", fontsize=10)
        ax.set_ylim(-3.3 * A, 3.3 * A)
        ax.legend(fontsize=8, loc="upper right")

        ax = eksenler[1, sutun]
        g = gauss_filtre(gurultulu, s)
        m = medyan_filtre(gurultulu, p)
        ax.plot(t, temiz, "k", linewidth=1, label="temiz")
        ax.plot(t, g, color="tab:blue", label=f"Gauss sigma={s} (RMSE {rmse(g, temiz):.2f})")
        ax.plot(t, m, color="tab:orange", label=f"Medyan p={p} (RMSE {rmse(m, temiz):.2f})")
        ax.set_title("En iyi ayarlarla filtrelenmiş", fontsize=10)
        ax.set_ylim(-1.8 * A, 1.8 * A)
        ax.legend(fontsize=8, loc="upper right")

        ax = eksenler[2, sutun]
        ax.plot(SIGMALAR, gh, "o-", color="tab:blue", label="Gauss (x: sigma)")
        ax.plot([(p_ - 1) / 4 for p_ in PENCERELER], mh, "s-", color="tab:orange",
                label="Medyan (x: (pencere-1)/4)")
        ax.axhline(rmse(gurultulu, temiz), color="tab:gray", linestyle="--", label="filtresiz")
        ax.set_xlabel("filtre genişliği: Gauss'ta sigma, medyanda (pencere - 1) / 4")
        ax.set_ylabel("RMSE")
        ax.set_title("Genişliğe göre hata", fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "gauss_medyan_filtre.png")
    fig.savefig(yol, dpi=90)
    print("\nKaydedildi:", yol)

    # Aşırı geniş filtrelerin etkisi
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, temiz, "k", linewidth=1.2, label="temiz")
    ax.plot(t, gauss_filtre(gauss_gurultulu, 8), color="tab:blue", label="Gauss sigma=8")
    ax.plot(t, medyan_filtre(gauss_gurultulu, 31), color="tab:orange", label="Medyan p=31")
    ax.set_title("Aşırı geniş filtre: Gauss genliği düşürür, medyan tepeleri düzleştirir", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "asiri_genis_filtre.png")
    fig.savefig(yol, dpi=90)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
