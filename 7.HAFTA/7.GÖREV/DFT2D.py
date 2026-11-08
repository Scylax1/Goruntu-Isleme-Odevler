"""
Görüntü İşleme Ödevi - 2 Boyutlu DFT (elle)
-------------------------------------------
dft2(resim)    -> harmonik matrisi F (karmaşık, M x N)
idft2(F)       -> resim (9. ve 10. görevlerde kullanılır)
merkeze_al(F)  -> F'nin dörtte birlik bloklarını yer değiştirip DC'yi
                  (sıfır frekansı) ortaya getirir (fftshift karşılığı)

2D DFT tanımı:

    F[u, v] = sum_{m=0}^{M-1} sum_{n=0}^{N-1} f[m, n] e^(-j 2 pi (u m / M + v n / N))

Üstel ifade iki çarpana ayrılır: e^(-j2pi um/M) * e^(-j2pi vn/N). Bu yüzden
2D DFT, önce her satıra 1D DFT, sonra her sütuna 1D DFT uygulamaktır
(ayrılabilirlik):

    F = W_M  f  W_N^T,     W_N[k, n] = e^(-j 2 pi k n / N)   (DFT matrisi)

W_N, 4. görevdeki gibi yalnızca N tane cos / sin değerinden kurulur:
W_N[k, n] = COS[(k n) mod N] - j SIN[(k n) mod N]. 1D DFT'yi her satıra
Python döngüsüyle uygulamak 512x512 bir resimde ~20 s sürerdi; matris
çarpımıyla aynı toplamlar milisaniyeler içinde hesaplanır.

Tanımın kendisi (dört iç içe döngü) dft2_dongu fonksiyonunda da yazılmıştır.
Küçük bir resimde matris yoluyla aynı sonucu verdiği kontrol edilir.

Ters dönüşüm:  f = (1 / MN) conj(W_M) F conj(W_N)^T

Harmonik matrisinin yorumu (merkezlenmiş, log(1 + |F|) ile gösterilir):
  - Merkez: DC, resmin ortalama parlaklığı x MN.
  - Merkeze yakın: yavaş değişen bileşenler (geniş alanlar, genel parlaklık).
  - Kenarlara doğru: hızlı değişen bileşenler (kenarlar, ince doku, gürültü).
  - Resimdeki bir kenar, harmonik matrisinde KENARA DİK bir çizgi verir.
  - Doğal resimlerin genliği frekansla hızla (~1/f) azalır.

Gösterilenler: yapay desenler (dikey / eğik çizgiler, kare, daire) ve
cameraman resmi; genlik ve faz. Fazın önemi: bir resmin genliği ile
diğerinin fazı birleştirilince sonuç FAZIN geldiği resme benzer.

Kullanım:
    python DFT2D.py [resim_yolu]
"""

import math
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
KAYNAK_RESIM = os.path.join(KLASOR, "..", "..", "2.HAFTA", "1.GÖREV", "ciktilar", "cameraman.png")


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma yardımcısı
# ---------------------------------------------------------------------------
def gri_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# 2D DFT
# ---------------------------------------------------------------------------
def dft_matrisi(n_boyut):
    """W[k, n] = e^(-j 2 pi k n / N), N tane cos / sin değerinden kurulur."""
    cos_tablosu = np.array([math.cos(2 * math.pi * m / n_boyut) for m in range(n_boyut)])
    sin_tablosu = np.array([math.sin(2 * math.pi * m / n_boyut) for m in range(n_boyut)])
    indeks = np.outer(np.arange(n_boyut), np.arange(n_boyut)) % n_boyut
    return cos_tablosu[indeks] - 1j * sin_tablosu[indeks]


def dft2(resim):
    """Harmonik matrisi F = W_M f W_N^T (satırlara ve sütunlara 1D DFT)."""
    f = np.asarray(resim, dtype=np.float64)
    W_M, W_N = dft_matrisi(f.shape[0]), dft_matrisi(f.shape[1])
    return W_M @ f @ W_N.T


def idft2(harmonikler):
    """Ters 2D DFT, gerçel kısmı döner."""
    M, N = harmonikler.shape
    W_M, W_N = dft_matrisi(M), dft_matrisi(N)
    return np.real(np.conj(W_M) @ harmonikler @ np.conj(W_N).T) / (M * N)


def dft2_dongu(resim):
    """Tanımın kendisi: dört iç içe döngü. Yalnızca küçük resimler için."""
    M, N = resim.shape
    F = np.zeros((M, N), dtype=np.complex128)
    for u in range(M):
        for v in range(N):
            toplam = 0j
            for m in range(M):
                for n in range(N):
                    aci = -2 * math.pi * (u * m / M + v * n / N)
                    toplam += resim[m, n] * complex(math.cos(aci), math.sin(aci))
            F[u, v] = toplam
    return F


def merkeze_al(F):
    """DC'yi (0, 0) köşesinden ortaya taşır: dört bloğu çapraz yer değiştirir."""
    M, N = F.shape
    return np.roll(np.roll(F, M // 2, axis=0), N // 2, axis=1)


def merkezden_geri(F):
    M, N = F.shape
    return np.roll(np.roll(F, -(M // 2), axis=0), -(N // 2), axis=1)


def log_genlik(F):
    """Gösterim için: merkezlenmiş log(1 + |F|)."""
    return np.log1p(np.abs(merkeze_al(F)))


# ---------------------------------------------------------------------------
# Yapay desenler
# ---------------------------------------------------------------------------
def yapay_desenler(boyut=256):
    y, x = np.mgrid[0:boyut, 0:boyut]
    dikey = 127.5 + 127.5 * np.sin(2 * np.pi * 8 * x / boyut)              # 8 devir yatayda
    egik = 127.5 + 127.5 * np.sin(2 * np.pi * (6 * x + 12 * y) / boyut)    # eğik
    kare = np.zeros((boyut, boyut))
    kare[boyut // 2 - 20:boyut // 2 + 20, boyut // 2 - 20:boyut // 2 + 20] = 255
    daire = (((x - boyut / 2) ** 2 + (y - boyut / 2) ** 2) < 30 ** 2) * 255.0
    return [("Dikey çizgiler (8 devir)", dikey), ("Eğik çizgiler (6, 12)", egik),
            ("Kare 40x40", kare), ("Daire r=30", daire)]


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape)

    # Doğrulama 1: tanım (dört döngü) = matris yolu, küçük resimde
    kucuk = cv2.resize(resim, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float64)
    t0 = time.perf_counter()
    F_dongu = dft2_dongu(kucuk)
    t1 = time.perf_counter()
    print(f"16x16: dört döngülü tanım {t1 - t0:.2f} s, matris yoluyla en büyük fark "
          f"{np.abs(F_dongu - dft2(kucuk)).max():.2e}")

    # Doğrulama 2: büyük resimde np.fft.fft2 ve ters dönüşüm
    t0 = time.perf_counter()
    F = dft2(resim)
    t1 = time.perf_counter()
    print(f"{resim.shape[0]}x{resim.shape[1]}: dft2 süresi {(t1 - t0) * 1000:.0f} ms | "
          f"np.fft.fft2 ile göreli fark {np.abs(F - np.fft.fft2(resim)).max() / np.abs(F).max():.2e} | "
          f"idft2(dft2(f)) - f en büyük {np.abs(idft2(F) - resim).max():.2e}")
    print(f"F[0, 0] = {F[0, 0].real:.0f} = ortalama ({resim.mean():.2f}) x {resim.size}")

    # Harmonik enerjisinin merkezde toplanması
    M, N = resim.shape
    yy, xx = np.mgrid[0:M, 0:N]
    uzaklik = np.hypot(yy - M // 2, xx - N // 2)
    enerji = np.abs(merkeze_al(F)) ** 2
    for r in (5, 20, 50):
        print(f"   Merkeze {r:2d} harmonikten yakın bölge: enerjinin %{100 * enerji[uzaklik <= r].sum() / enerji.sum():.2f}'i")

    # --- Yapay desenler ---
    desenler = yapay_desenler()
    fig, eksenler = plt.subplots(2, len(desenler), figsize=(16, 8))
    for i, (ad, d) in enumerate(desenler):
        eksenler[0, i].imshow(d, cmap="gray")
        eksenler[0, i].set_title(ad, fontsize=10)
        Fd = dft2(d)
        if i < 2:
            # Saf sinüslerde harmonikler tek piksel: merkezin +-24'ü büyütülerek gösterilir
            r, c = d.shape[0] // 2, d.shape[1] // 2
            eksenler[1, i].imshow(log_genlik(Fd)[r - 24:r + 25, c - 24:c + 25], cmap="inferno",
                                  extent=(-24.5, 24.5, 24.5, -24.5), interpolation="nearest")
            eksenler[1, i].set_title("log(1 + |F|), merkezin +-24'ü", fontsize=10)
        else:
            eksenler[1, i].imshow(log_genlik(Fd), cmap="inferno")
            eksenler[1, i].set_title("log(1 + |F|), merkezlenmiş", fontsize=10)
        guclu = np.argwhere(np.abs(merkeze_al(Fd)) > 0.01 * np.abs(Fd).max())
        if len(guclu) <= 3:
            konumlar = [(int(u) - d.shape[0] // 2, int(v) - d.shape[1] // 2) for u, v in guclu]
            print(f"{ad}: harmonikler merkeze göre (u, v) = {konumlar}")
    for ax in eksenler.ravel():
        ax.axis("off")
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "yapay_desenler_dft.png")
    fig.savefig(yol_kayit, dpi=80)
    print("Kaydedildi:", yol_kayit)

    # --- Cameraman: genlik, faz ve faz / genlik takası ---
    desen = desenler[3][1]
    desen = cv2.resize(desen, (N, M))
    F_desen = dft2(desen)
    genlik_cam_faz_daire = idft2(np.abs(F) * np.exp(1j * np.angle(F_desen)))
    genlik_daire_faz_cam = idft2(np.abs(F_desen) * np.exp(1j * np.angle(F)))

    fig, eksenler = plt.subplots(2, 3, figsize=(15, 10))
    eksenler[0, 0].imshow(resim, cmap="gray")
    eksenler[0, 0].set_title("Orijinal")
    eksenler[0, 1].imshow(log_genlik(F), cmap="inferno")
    eksenler[0, 1].set_title("Harmonik matrisi: log(1 + |F|)")
    eksenler[0, 2].imshow(np.angle(merkeze_al(F)), cmap="twilight")
    eksenler[0, 2].set_title("Faz: angle(F)")
    eksenler[1, 0].imshow(idft2(F), cmap="gray")
    eksenler[1, 0].set_title(f"idft2(dft2(f)), en büyük fark {np.abs(idft2(F) - resim).max():.1e}")
    eksenler[1, 1].imshow(genlik_daire_faz_cam, cmap="gray")
    eksenler[1, 1].set_title("Dairenin genliği + cameraman'in fazı")
    eksenler[1, 2].imshow(genlik_cam_faz_daire, cmap="gray")
    eksenler[1, 2].set_title("Cameraman'in genliği + dairenin fazı")
    for ax in eksenler.ravel():
        ax.axis("off")
    fig.tight_layout()
    yol_kayit = os.path.join(CIKTI_KLASORU, "cameraman_dft.png")
    fig.savefig(yol_kayit, dpi=80)
    print("Kaydedildi:", yol_kayit)
    plt.show()


if __name__ == "__main__":
    main()
