"""
Sinyal İşleme Ödevi - Vektöre Genliğe ve Orana Bağlı Rastgele Gürültü Ekleme
----------------------------------------------------------------------------
gurultu_ekle(vektor, oran, tohum=None) -> gürültülü yeni vektör

  vektor : gürültü eklenecek sinyal (değiştirilmez, kopyası döner)
  oran   : [0, 1] gürültü oranı
  tohum  : rastgele sayı üretecinin tohumu (aynı tohum -> aynı gürültü)

Gürültünün büyüklüğü sinyalin genliğine göre ayarlanır:

    genlik      = max |vektor[n] - ortalama|     (sinyalin tepe genliği)
    sigma       = oran * genlik
    gurultu[n] ~ N(0, sigma^2)                   (her eleman için bağımsız)
    sonuc[n]    = vektor[n] + gurultu[n]

  oran = 0   -> gürültü yok
  oran = 0.1 -> gürültünün standart sapması genliğin %10'u
  oran = 1   -> gürültünün standart sapması genlik kadar

Genliğe göre ölçeklemek, aynı oranın her sinyalde aynı "görsel kirlilik"
seviyesi vermesini sağlar: A = 1 ve A = 12 olan iki sinüse oran = 0.3
verildiğinde ikisinde de sinyal / gürültü oranı aynıdır.

Gürültü Gauss (normal) dağılımlıdır: doğadaki ölçüm gürültüsü
(algılayıcı, ısıl gürültü) genelde bu dağılıma uyar. Gauss sayıları
Box-Muller dönüşümüyle iki düzgün dağılımlı sayıdan elle üretilir:

    z = sqrt(-2 ln u1) * cos(2 pi u2),   u1, u2 ~ U(0, 1]

Sinyal / gürültü oranı (SNR) dB cinsinden de yazdırılır:

    SNR = 10 log10(sinyal_gücü / gürültü_gücü)

Sinüs için sinyal gücü A^2 / 2, gürültü gücü sigma^2 olduğundan
SNR = 10 log10(1 / (2 oran^2)). Örneğin oran = 0.1 -> 17 dB, oran = 1 -> -3 dB.

Kullanım:
    python GurultuEkle.py
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

# 1. görevdeki sinüs fonksiyonu
sys.path.insert(0, os.path.join(KLASOR, "..", "1.GÖREV"))
from SinusSinyal import sinus_doldur  # noqa: E402

ORANLAR = [0.0, 0.1, 0.3, 0.6, 1.0]


# ---------------------------------------------------------------------------
# Fonksiyon
# ---------------------------------------------------------------------------
def gauss_sayisi(uretec):
    """Box-Muller: iki düzgün dağılımlı sayıdan bir standart normal sayı."""
    u1 = 1.0 - uretec.random()          # (0, 1]: log(0) olmasın
    u2 = uretec.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def gurultu_ekle(vektor, oran, tohum=None):
    """Genliğe ve orana göre her elemana bağımsız Gauss gürültüsü ekler."""
    if not 0.0 <= oran <= 1.0:
        raise ValueError(f"oran [0, 1] aralığında olmalı, verilen: {oran}")
    uretec = random.Random(tohum)
    ortalama = sum(vektor) / len(vektor)
    genlik = max(abs(x - ortalama) for x in vektor)
    sigma = oran * genlik

    sonuc = np.array(vektor, dtype=np.float64)       # kopya: girdi değişmez
    for n in range(len(sonuc)):
        sonuc[n] += sigma * gauss_sayisi(uretec)
    return sonuc


def snr_db(temiz, gurultulu):
    gurultu = gurultulu - temiz
    guc_gurultu = np.mean(gurultu ** 2)
    return float("inf") if guc_gurultu == 0 else 10 * np.log10(np.mean(temiz ** 2) / guc_gurultu)


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    N, A, f = 256, 5.0, 4
    temiz = sinus_doldur(np.zeros(N), A, 0.0, frekans=f)
    t = np.arange(N) / N

    print(f"Sinyal: N = {N}, A = {A}, f = {f} Hz")
    print(f"{'oran':>6}{'sigma':>8}{'ölçülen std':>13}{'SNR (dB)':>10}{'kuramsal':>10}")
    fig, eksenler = plt.subplots(len(ORANLAR), 2, figsize=(14, 2.6 * len(ORANLAR)),
                                 gridspec_kw={"width_ratios": [3, 1]})
    for satir, oran in enumerate(ORANLAR):
        gurultulu = gurultu_ekle(temiz, oran, tohum=42)
        eklenen = gurultulu - temiz
        snr = snr_db(temiz, gurultulu)
        kuramsal = float("inf") if oran == 0 else 10 * np.log10(1 / (2 * oran ** 2))
        print(f"{oran:>6.1f}{oran * A:>8.2f}{eklenen.std():>13.2f}{snr:>10.1f}{kuramsal:>10.1f}")

        ax = eksenler[satir, 0]
        ax.plot(t, gurultulu, ".-", color="tab:red", linewidth=0.6, markersize=2.5, label="gürültülü")
        ax.plot(t, temiz, color="black", linewidth=1.2, label="temiz")
        ax.set_ylim(-A * 4, A * 4)
        ax.set_title(f"oran = {oran}  (sigma = {oran * A:.1f}, SNR = {snr:.1f} dB)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

        ax = eksenler[satir, 1]
        if oran > 0:
            ax.hist(eklenen, bins=30, color="tab:red", alpha=0.7, density=True)
            x = np.linspace(-4 * oran * A, 4 * oran * A, 200)
            ax.plot(x, np.exp(-x ** 2 / (2 * (oran * A) ** 2)) / (oran * A * np.sqrt(2 * np.pi)), "k")
            ax.set_title("eklenen gürültünün dağılımı", fontsize=9)
        else:
            ax.text(0.5, 0.5, "gürültü yok", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    eksenler[-1, 0].set_xlabel("zaman (s)")

    # Genliğe göre ölçekleme: A=1 ve A=12'de aynı oran -> aynı SNR
    for a in (1.0, 12.0):
        s = sinus_doldur(np.zeros(N), a, 0.0, frekans=f)
        print(f"A = {a:4.1f}, oran = 0.3 -> SNR = {snr_db(s, gurultu_ekle(s, 0.3, tohum=7)):.1f} dB")

    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "gurultu_oranlari.png")
    fig.savefig(yol, dpi=90)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
