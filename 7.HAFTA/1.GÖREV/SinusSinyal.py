"""
Sinyal İşleme Ödevi - Vektörü Sinüs Değerleriyle Dolduran Fonksiyon
-------------------------------------------------------------------
sinus_doldur(vektor, genlik, faz, frekans=..., periyot=..., ornekleme_frekansi=...)

Parametre olarak verilen vektörün her elemanını, istenen sinüsün o
örnekteki değeriyle doldurur (vektör YERİNDE değiştirilir):

    vektor[n] = genlik * sin(2 * pi * frekans * t_n + faz),   t_n = n / fs

  genlik             : sinüsün tepe değeri
  faz                : radyan cinsinden başlangıç fazı (n = 0'daki açı)
  frekans / periyot  : ikisinden biri verilir. periyot = 1 / frekans (saniye)
  ornekleme_frekansi : fs, saniyedeki örnek sayısı. Verilmezse fs = N alınır,
                       yani vektör tam 1 saniyelik sinyali tutar. Böylece
                       "frekans" doğrudan "vektör boyunca kaç tam devir" olur
                       ve DFT'de k. harmonik k Hz'e denk gelir.

Vektörün uzunluğu 2'nin kuvveti olmalıdır (FFT / DFT için alışılmış boyut).
Değilse ValueError verilir. Kontrol: n > 0 ve n & (n - 1) == 0.
(2'nin kuvvetinin ikili gösteriminde tek bir 1 biti vardır; n - 1 o bitin
altındaki bütün bitleri 1 yapar, bu yüzden ikisinin VE'si 0 olur.)

Örnekleme kuramına (Nyquist) göre frekans fs / 2'den küçük olmalıdır.
Büyükse sinüs örneklerde daha düşük bir frekans gibi görünür (aliasing).
Fonksiyon bu durumda uyarı yazar; bunun bir örneği de görselleştirilir.

Görselleştirme: örnekler (vektörün elemanları) sap grafiği (stem) olarak,
gerçek sürekli sinüs de çok sık örneklenmiş ince bir eğri olarak çizilir.

Kullanım:
    python SinusSinyal.py
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")


# ---------------------------------------------------------------------------
# Fonksiyon
# ---------------------------------------------------------------------------
def ikinin_kuvveti_mi(n):
    return n > 0 and (n & (n - 1)) == 0


def sinus_doldur(vektor, genlik, faz, frekans=None, periyot=None, ornekleme_frekansi=None):
    """vektor'ü A*sin(2*pi*f*n/fs + faz) değerleriyle yerinde doldurur ve döndürür."""
    n_ornek = len(vektor)
    if not ikinin_kuvveti_mi(n_ornek):
        raise ValueError(f"Vektör uzunluğu 2'nin kuvveti olmalı, verilen: {n_ornek}")
    if (frekans is None) == (periyot is None):
        raise ValueError("frekans ya da periyot'tan yalnızca biri verilmeli")
    if periyot is not None:
        if periyot <= 0:
            raise ValueError("periyot pozitif olmalı")
        frekans = 1.0 / periyot

    fs = n_ornek if ornekleme_frekansi is None else ornekleme_frekansi
    if frekans >= fs / 2:
        print(f"  UYARI: frekans ({frekans} Hz) >= fs/2 ({fs / 2} Hz), örtüşme (aliasing) olur")

    acisal_adim = 2.0 * math.pi * frekans / fs      # iki örnek arasındaki açı (radyan)
    for n in range(n_ornek):
        vektor[n] = genlik * math.sin(acisal_adim * n + faz)
    return vektor


# ---------------------------------------------------------------------------
# Görselleştirme
# ---------------------------------------------------------------------------
def ciz(ax, vektor, genlik, faz, frekans, fs, baslik):
    """Örnekler (stem) + gerçek sürekli sinüs (ince eğri)."""
    n_ornek = len(vektor)
    t = np.arange(n_ornek) / fs
    t_surekli = np.linspace(0, n_ornek / fs, 4000, endpoint=False)
    ax.plot(t_surekli, genlik * np.sin(2 * np.pi * frekans * t_surekli + faz),
            color="tab:gray", linewidth=0.9, label="sürekli sinüs")
    if frekans >= fs / 2:
        # Örneklerin aslında izlediği düşük frekans: f - fs * round(f / fs)
        f_ortusen = frekans - fs * round(frekans / fs)
        ax.plot(t_surekli, genlik * np.sin(2 * np.pi * f_ortusen * t_surekli + faz),
                "r--", linewidth=1.2, label=f"örneklerin göründüğü: {f_ortusen:g} Hz")
    isaret, sap, taban = ax.stem(t, vektor, label=f"vektör ({n_ornek} örnek)")
    plt.setp(sap, linewidth=0.8)
    plt.setp(isaret, markersize=3)
    plt.setp(taban, linewidth=0.5, color="black")
    ax.set_title(baslik, fontsize=10)
    ax.set_xlabel("zaman (s)")
    ax.set_ylabel("genlik")
    ax.legend(fontsize=8, loc="upper right")


def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)

    # (N, genlik, faz, frekans, periyot, fs, başlık)
    ornekler = [
        (64, 1.0, 0.0, 2, None, None, "N=64, A=1, faz=0, f=2 Hz (fs=N=64)"),
        (64, 3.0, math.pi / 2, None, 0.25, None, "N=64, A=3, faz=pi/2, periyot=0.25 s (f=4 Hz)"),
        (128, 2.0, -math.pi / 4, 5, None, 128, "N=128, A=2, faz=-pi/4, f=5 Hz"),
        (512, 12.0, 0.0, 50, None, None, "N=512, A=12, f=50 Hz (5. görevdeki sinyal)"),
        (64, 1.0, 0.0, 6, None, 32, "N=64, f=6 Hz, fs=32 Hz: iyi örneklenmiş"),
        (64, 1.0, 0.0, 30, None, 32, "Örtüşme: f=30 Hz, fs=32 Hz -> örnekler -2 Hz gibi"),
    ]

    fig, eksenler = plt.subplots(3, 2, figsize=(14, 10))
    for ax, (n, a, faz, f, periyot, fs, baslik) in zip(eksenler.ravel(), ornekler):
        vektor = np.zeros(n)                       # doldurulacak vektör
        sinus_doldur(vektor, a, faz, frekans=f, periyot=periyot, ornekleme_frekansi=fs)
        f_gercek = f if f is not None else 1 / periyot
        fs_gercek = n if fs is None else fs
        print(f"{baslik}\n   ilk 6 eleman: {np.round(vektor[:6], 3)}  "
              f"| min {vektor.min():.3f}  max {vektor.max():.3f}")
        if n == 512:
            # Uzun vektörde sapları seyrek göstermek için ilk 0.1 saniye
            ciz(ax, vektor[:64], a, faz, f_gercek, fs_gercek, baslik + " (ilk 64 örnek)")
        else:
            ciz(ax, vektor, a, faz, f_gercek, fs_gercek, baslik)

    # Doğrulama: numpy ile aynı formül
    v = sinus_doldur(np.zeros(256), 2.5, 0.3, frekans=7)
    beklenen = 2.5 * np.sin(2 * np.pi * 7 * np.arange(256) / 256 + 0.3)
    print(f"Doğrulama (N=256, numpy ile): en büyük fark = {np.abs(v - beklenen).max():.2e}")

    # 2'nin kuvveti olmayan uzunluk reddedilir
    try:
        sinus_doldur(np.zeros(100), 1, 0, frekans=1)
    except ValueError as hata:
        print("N=100 ile çağrı:", hata)

    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "sinus_ornekleri.png")
    fig.savefig(yol, dpi=100)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
