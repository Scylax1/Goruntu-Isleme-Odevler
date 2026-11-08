"""
Sinyal İşleme Ödevi - DFT ile Frekans Uzayına Dönüşüm (elle)
------------------------------------------------------------
dft(vektor)  -> harmonik vektörü X (karmaşık, uzunluk N)
idft(X)      -> zaman uzayındaki sinyal (6. görevde kullanılır)

Ayrık Fourier Dönüşümü:

    X[k] = sum_{n=0}^{N-1} x[n] * e^(-j 2 pi k n / N)
         = sum_n x[n] * ( cos(2 pi k n / N) - j sin(2 pi k n / N) ),   k = 0..N-1

  X[k], sinyalin içindeki k. harmoniğin (vektör boyunca k tam devir yapan
  sinüs / kosinüs) karmaşık katsayısıdır:
     |X[k]|      : o harmoniğin ne kadar güçlü olduğu
     angle(X[k]) : fazı

  Sinyal 1 saniye (fs = N) ise k. harmonik k Hz'dir.

sin / cos tablosu: k*n çarpımı ne kadar büyük olursa olsun, açı
2 pi (k n mod N) / N'ye eşittir. Bu yüzden yalnızca N tane cos ve N tane sin
değeri hesaplanıp tabloda tutulur; döngü (k n) mod N indeksiyle tablodan
okur. N^2 kez cos / sin hesaplamaya gerek kalmaz.

Uzunluk 2'nin kuvveti olmalıdır (1. görevdeki kontrol).

Harmonik vektörünü yorumlamak için tek taraflı genlik spektrumu:

    genlik[0]   = |X[0]| / N          (ortalama, DC)
    genlik[k]   = 2 |X[k]| / N        k = 1 .. N/2 - 1
    genlik[N/2] = |X[N/2]| / N        (Nyquist)

  Gerçel bir sinyalde X[N-k] = X[k]* (eşlenik), yani k ve N-k aynı
  harmoniktir; ikisinin toplamı 2|X[k]|. N'ye bölünce A genlikli
  sinüsün harmoniği tam A çıkar.

Gürültüsüz ve gürültülü (2. görevdeki gurultu_ekle) sinüsler ayrı ayrı
dönüştürülüp harmonik vektörleri karşılaştırılır. Sonuç np.fft.fft ile
yalnızca kontrol için karşılaştırılır.

Kullanım:
    python DFT.py
"""

import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

sys.path.insert(0, os.path.join(KLASOR, "..", "1.GÖREV"))
sys.path.insert(0, os.path.join(KLASOR, "..", "2.GÖREV"))
from SinusSinyal import ikinin_kuvveti_mi, sinus_doldur  # noqa: E402
from GurultuEkle import gurultu_ekle  # noqa: E402

N, A, F = 512, 12.0, 50


# ---------------------------------------------------------------------------
# DFT / ters DFT
# ---------------------------------------------------------------------------
def _tablolar(n_ornek):
    acilar = [2.0 * math.pi * m / n_ornek for m in range(n_ornek)]
    return [math.cos(a) for a in acilar], [math.sin(a) for a in acilar]


def dft(vektor):
    """Harmonik vektörü X[k] = sum x[n] e^(-j 2 pi k n / N)."""
    n_ornek = len(vektor)
    if not ikinin_kuvveti_mi(n_ornek):
        raise ValueError(f"Vektör uzunluğu 2'nin kuvveti olmalı, verilen: {n_ornek}")
    cos_tablosu, sin_tablosu = _tablolar(n_ornek)
    x = [float(v) for v in vektor]

    harmonikler = np.zeros(n_ornek, dtype=np.complex128)
    for k in range(n_ornek):
        gercel, sanal = 0.0, 0.0
        for n in range(n_ornek):
            m = (k * n) % n_ornek
            gercel += x[n] * cos_tablosu[m]
            sanal -= x[n] * sin_tablosu[m]
        harmonikler[k] = complex(gercel, sanal)
    return harmonikler


def idft(harmonikler):
    """Ters DFT: x[n] = (1/N) sum X[k] e^(+j 2 pi k n / N). Gerçel kısmı döner."""
    n_ornek = len(harmonikler)
    cos_tablosu, sin_tablosu = _tablolar(n_ornek)
    gercel_k = [h.real for h in harmonikler]
    sanal_k = [h.imag for h in harmonikler]

    sinyal = np.zeros(n_ornek)
    for n in range(n_ornek):
        toplam = 0.0
        for k in range(n_ornek):
            m = (k * n) % n_ornek
            # (a + jb)(cos + j sin) ifadesinin gerçel kısmı: a cos - b sin
            toplam += gercel_k[k] * cos_tablosu[m] - sanal_k[k] * sin_tablosu[m]
        sinyal[n] = toplam / n_ornek
    return sinyal


def genlik_spektrumu(harmonikler):
    """Tek taraflı genlik: k = 0 .. N/2, A genlikli sinüs -> A."""
    n_ornek = len(harmonikler)
    genlik = np.abs(harmonikler[: n_ornek // 2 + 1]) / n_ornek
    genlik[1:n_ornek // 2] *= 2
    return genlik


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def ozet(ad, x, X):
    g = genlik_spektrumu(X)
    tepe = int(np.argmax(g[1:]) + 1)
    diger = np.delete(g[1:], tepe - 1)
    print(f"{ad:<34} tepe k = {tepe:3d}, genlik = {g[tepe]:6.3f}, faz = {np.degrees(np.angle(X[tepe])):7.1f} der"
          f" | diğer harmonikler: ort {diger.mean():.3f}, en büyük {diger.max():.3f}"
          f" | DC = {g[0]:.3f}")
    return g


def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    temiz = sinus_doldur(np.zeros(N), A, 0.0, frekans=F)
    gurultulu_03 = gurultu_ekle(temiz, 0.3, tohum=42)
    gurultulu_10 = gurultu_ekle(temiz, 1.0, tohum=42)
    kayan = sinus_doldur(np.zeros(N), A, 0.0, frekans=F + 0.5)     # tam sayı olmayan devir

    t0 = time.perf_counter()
    X_temiz = dft(temiz)
    print(f"dft({N}) süresi: {time.perf_counter() - t0:.2f} s")
    X_03, X_10, X_kayan = dft(gurultulu_03), dft(gurultulu_10), dft(kayan)

    # Kontrol: np.fft.fft ve ters dönüşüm
    print(f"np.fft.fft ile en büyük fark: {np.abs(X_03 - np.fft.fft(gurultulu_03)).max():.2e}")
    print(f"idft(dft(x)) - x en büyük fark: {np.abs(idft(X_03) - gurultulu_03).max():.2e}")
    # Parseval: zamandaki enerji = frekanstaki enerji / N
    print(f"Parseval: sum x^2 = {np.sum(gurultulu_03 ** 2):.1f}, sum |X|^2 / N = {np.sum(np.abs(X_03) ** 2) / N:.1f}")
    print()

    g_temiz = ozet(f"Gürültüsüz ({F} Hz, A={A})", temiz, X_temiz)
    g_03 = ozet("Gürültülü, oran 0.3 (sigma 3.6)", gurultulu_03, X_03)
    g_10 = ozet("Gürültülü, oran 1.0 (sigma 12)", gurultulu_10, X_10)
    g_kayan = ozet(f"Gürültüsüz, {F + 0.5} Hz (sızıntı)", kayan, X_kayan)
    for sigma, g in ((0.3 * A, g_03), (1.0 * A, g_10)):
        # Beyaz gürültünün her harmoniğe düşen beklenen genliği: 2 sigma / sqrt(N) * sqrt(pi)/2
        print(f"   sigma = {sigma:4.1f}: beklenen gürültü tabanı {sigma * math.sqrt(math.pi / N):.3f}, "
              f"ölçülen ort. {np.delete(g[1:], F - 1).mean():.3f}")

    k = np.arange(N // 2 + 1)
    fig, eksenler = plt.subplots(4, 2, figsize=(15, 13), gridspec_kw={"width_ratios": [1, 2]})
    for satir, (ad, x, g) in enumerate((("Gürültüsüz", temiz, g_temiz),
                                        ("Gürültülü, oran 0.3", gurultulu_03, g_03),
                                        ("Gürültülü, oran 1.0", gurultulu_10, g_10),
                                        (f"Gürültüsüz, f = {F + 0.5} Hz", kayan, g_kayan))):
        ax = eksenler[satir, 0]
        ax.plot(np.arange(64) / N, x[:64], ".-", markersize=3, linewidth=0.8)
        ax.set_title(f"{ad}: ilk 64 örnek", fontsize=10)
        ax.set_ylim(-3.5 * A, 3.5 * A)
        ax = eksenler[satir, 1]
        isaret, sap, taban = ax.stem(k, g)
        plt.setp(sap, linewidth=0.7)
        plt.setp(isaret, markersize=2)
        plt.setp(taban, linewidth=0.4)
        ax.set_xlim(0, N // 2)
        ax.set_ylim(0, A * 1.15)
        ax.set_title(f"{ad}: harmonik genlikleri (tepe k={int(np.argmax(g[1:]) + 1)}, {g[1:].max():.2f})",
                     fontsize=10)
        ax.set_ylabel("genlik")
    eksenler[-1, 1].set_xlabel("harmonik k (= Hz, fs = N)")
    eksenler[-1, 0].set_xlabel("zaman (s)")
    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "dft_harmonikler.png")
    fig.savefig(yol, dpi=90)
    print("Kaydedildi:", yol)

    # Gürültü tabanını görmek için log ölçek
    fig, ax = plt.subplots(figsize=(12, 4))
    for ad, g, renk in (("gürültüsüz", g_temiz, "black"), ("oran 0.3", g_03, "tab:orange"),
                        ("oran 1.0", g_10, "tab:red")):
        ax.semilogy(k, np.maximum(g, 1e-16), color=renk, linewidth=0.8, label=ad)
    ax.set_xlim(0, N // 2)
    ax.set_xlabel("harmonik k")
    ax.set_ylabel("genlik (log)")
    ax.set_title("Log ölçekte harmonik genlikleri: gürültü bütün harmoniklere düz bir taban ekler")
    ax.legend()
    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "dft_log.png")
    fig.savefig(yol, dpi=90)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
