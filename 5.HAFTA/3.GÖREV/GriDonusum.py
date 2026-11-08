"""
Görüntü İşleme Ödevi - RGB Görüntüyü Gri Tonlu Görüntüye Dönüştürme
-------------------------------------------------------------------
Renkli bir görüntüyü tek kanallı gri görüntüye iki yolla dönüştürür ve
iki sonucun eşleşen piksellerinin mutlak farklarının toplamını (SAD)
hata değeri olarak raporlar:

  1) OpenCV:  cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
  2) Kendi fonksiyonumuz: kendi_bgr2gray(resim)

"Doğru şekilde" dönüştürmek için iki şeye dikkat etmek gerekir:

  a) Kanal sırası. cv2.imread resmi B, G, R sırasıyla okur. Kırmızının
     ağırlığı mavi kanala uygulanırsa sonuç yanlış olur.

  b) Ağırlıklar. Göz yeşile en duyarlı, maviye en az duyarlıdır. Bu yüzden
     üç kanalın düz ortalaması değil, ITU-R BT.601 parlaklık (luma)
     ağırlıkları kullanılır:

        Y = 0.299 R + 0.587 G + 0.114 B

     Düz ortalamada (R+G+B)/3 saf mavi (0, 0, 255) ile saf yeşil
     (0, 255, 0) aynı griye (85) düşer; oysa yeşil gözümüze çok daha
     parlak görünür. Luma ile mavi 29, yeşil 150 olur.

OpenCV 8 bitlik resimlerde bu formülü kayan noktayla değil, 15 bitlik sabit
noktalı (fixed-point) tamsayı aritmetiğiyle hesaplar:

        Y = (B*3735 + G*19235 + R*9798 + 16384) >> 15

  9798  = round(0.299 * 2^15),  19235 = round(0.587 * 2^15),
  3735  = 2^15 - 9798 - 19235   (round(0.114 * 2^15) = 3736 değil!
          Toplam tam 32768 = 2^15 olsun, beyaz 255'e gitsin diye mavinin
          katsayısı 1 eksik alınmış.)
  + 16384 (= 2^14) yuvarlama içindir, >> 15 de 2^15'e bölmedir.

Bu katsayılar OpenCV 4.x'in kaynağındaki değerlerdir. Bütün 2^24 = 16 777 216
BGR rengi üzerinde denenip cv2.cvtColor ile bire bir aynı sonucu verdiği
doğrulandı. Eski 14 bitlik katsayılar (1868, 9617, 4899) 43 864 renkte,
3736'lı 15 bitlik katsayılar 64 173 renkte 1 farklı sonuç verir.

kendi_bgr2gray bu sabit noktalı yolu piksel piksel uygular, bu yüzden
OpenCV ile hata 0 çıkar. Karşılaştırma için kayan noktalı formül, eski 14
bitlik katsayılar, yanlış kanal sırası ve düz ortalama da hesaplanıp
hataları raporlanır.

Varsayılan girdi 1. görevdeki baboon.jpg resmidir.

Kullanım:
    python GriDonusum.py [resim_yolu]
"""

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
KAYNAK_RESIM = os.path.join(KLASOR, "..", "1.GÖREV", "ciktilar", "baboon.jpg")

# BT.601 luma ağırlıkları
R_AGIRLIK, G_AGIRLIK, B_AGIRLIK = 0.299, 0.587, 0.114

# 15 bitlik sabit nokta karşılıkları (OpenCV 4.x ile aynı)
KAYDIRMA = 15
R_TAM = round(R_AGIRLIK * (1 << KAYDIRMA))   # 9798
G_TAM = round(G_AGIRLIK * (1 << KAYDIRMA))   # 19235
B_TAM = (1 << KAYDIRMA) - R_TAM - G_TAM      # 3735 (toplam tam 2^15 olsun)
YARIM = 1 << (KAYDIRMA - 1)                  # 16384


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma / yazma yardımcıları
# ---------------------------------------------------------------------------
def renkli_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_COLOR)


def kaydet(yol, resim):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    cv2.imencode(".png", resim)[1].tofile(yol)
    print("Kaydedildi:", yol)


# ---------------------------------------------------------------------------
# Gri dönüşüm fonksiyonları
# ---------------------------------------------------------------------------
def kendi_bgr2gray(resim):
    """BGR -> gri, OpenCV ile aynı 15 bit sabit noktalı luma, piksel piksel."""
    yukseklik, genislik = resim.shape[:2]
    gri = np.zeros((yukseklik, genislik), dtype=np.uint8)
    pikseller = resim.tolist()
    for y in range(yukseklik):
        satir = pikseller[y]
        gri_satir = [0] * genislik
        for x in range(genislik):
            b, g, r = satir[x]                      # OpenCV sırası: B, G, R
            gri_satir[x] = (b * B_TAM + g * G_TAM + r * R_TAM + YARIM) >> KAYDIRMA
        gri[y] = gri_satir
    return gri


def kayan_noktali_gri(resim):
    """Y = 0.299 R + 0.587 G + 0.114 B, kayan noktalı, en yakına yuvarlama."""
    b, g, r = [resim[:, :, i].astype(np.float64) for i in range(3)]
    return np.clip(np.rint(R_AGIRLIK * r + G_AGIRLIK * g + B_AGIRLIK * b), 0, 255).astype(np.uint8)


def eski_14bit(resim):
    """Eski OpenCV sürümlerinin 14 bitlik katsayıları: 1868, 9617, 4899."""
    b, g, r = [resim[:, :, i].astype(np.int64) for i in range(3)]
    return ((b * 1868 + g * 9617 + r * 4899 + 8192) >> 14).astype(np.uint8)


def yanlis_kanal_sirasi(resim):
    """HATALI: resmi RGB sanıp 0.299'u B'ye, 0.114'ü R'ye uygulamak."""
    b, g, r = [resim[:, :, i].astype(np.float64) for i in range(3)]
    return np.clip(np.rint(R_AGIRLIK * b + G_AGIRLIK * g + B_AGIRLIK * r), 0, 255).astype(np.uint8)


def duz_ortalama(resim):
    """HATALI: üç kanalın düz ortalaması (gözün duyarlılığını hesaba katmaz)."""
    return np.rint(resim.astype(np.float64).mean(axis=2)).astype(np.uint8)


def hata(a, b):
    """Eşleşen piksellerin mutlak farklarının toplamı (SAD) ve diğer ölçüler."""
    fark = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return int(fark.sum()), int(fark.max()), int(np.count_nonzero(fark)), fark


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = renkli_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    toplam_piksel = resim.shape[0] * resim.shape[1]
    print("Resim:", yol, "| Boyut:", resim.shape)
    print(f"Sabit nokta katsayıları: R = {R_TAM}, G = {G_TAM}, B = {B_TAM} "
          f"(toplam {R_TAM + G_TAM + B_TAM} = 2^{KAYDIRMA})")

    t0 = time.perf_counter()
    opencv_gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    t1 = time.perf_counter()
    kendi_gri = kendi_bgr2gray(resim)
    t2 = time.perf_counter()
    print(f"Süre: cv2.cvtColor {(t1 - t0) * 1000:.2f} ms | kendi_bgr2gray {(t2 - t1) * 1000:.0f} ms")
    print(f"Çıktılar: OpenCV {opencv_gri.shape} {opencv_gri.dtype} | "
          f"kendi {kendi_gri.shape} {kendi_gri.dtype} (tek kanal)")

    sonuclar = [("kendi_bgr2gray (sabit nokta)", kendi_gri),
                ("Kayan noktalı formül", kayan_noktali_gri(resim)),
                ("14 bit katsayılar", eski_14bit(resim)),
                ("HATALI: kanal sırası ters", yanlis_kanal_sirasi(resim)),
                ("HATALI: düz ortalama", duz_ortalama(resim))]

    print(f"\nOpenCV COLOR_BGR2GRAY ile karşılaştırma ({toplam_piksel} piksel):")
    print(f"{'Yöntem':<32}{'SAD (hata)':>12}{'Ort. fark':>11}{'En büyük':>10}{'Farklı piksel':>15}")
    farklar = []
    for ad, gri in sonuclar:
        sad, en_buyuk, farkli, fark = hata(opencv_gri, gri)
        farklar.append(fark)
        print(f"{ad:<32}{sad:>12}{sad / toplam_piksel:>11.3f}{en_buyuk:>10}{farkli:>15}")

    # Düz ortalamanın neden yanlış olduğu: saf renkler
    print("\nSaf renklerin gri karşılığı (OpenCV / düz ortalama):")
    for ad, bgr in (("Kırmızı", (0, 0, 255)), ("Yeşil", (0, 255, 0)), ("Mavi", (255, 0, 0))):
        piksel = np.uint8([[bgr]])
        print(f"   {ad:<8} -> {cv2.cvtColor(piksel, cv2.COLOR_BGR2GRAY)[0, 0]:3d} / {duz_ortalama(piksel)[0, 0]:3d}")

    kaydet(os.path.join(CIKTI_KLASORU, "gri_opencv.png"), opencv_gri)
    kaydet(os.path.join(CIKTI_KLASORU, "gri_kendi.png"), kendi_gri)

    fig, eksenler = plt.subplots(2, 4, figsize=(16, 8.5))
    eksenler[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    eksenler[0, 0].set_title("Orijinal")
    eksenler[0, 1].imshow(opencv_gri, cmap="gray", vmin=0, vmax=255)
    eksenler[0, 1].set_title("cv2.cvtColor BGR2GRAY")
    # Üst satır: OpenCV ve kendi fonksiyonumuz; alt satır: iki hatalı yöntem
    yerlesim = [(eksenler[0, 2], eksenler[0, 3], 0),
                (eksenler[1, 0], eksenler[1, 1], 3),
                (eksenler[1, 2], eksenler[1, 3], 4)]
    for resim_ax, fark_ax, i in yerlesim:
        ad, gri = sonuclar[i]
        resim_ax.imshow(gri, cmap="gray", vmin=0, vmax=255)
        resim_ax.set_title(f"{ad}\nSAD = {int(farklar[i].sum())}", fontsize=10)
        goruntu = fark_ax.imshow(farklar[i], cmap="magma", vmin=0, vmax=max(1, int(farklar[i].max())))
        fark_ax.set_title(f"|OpenCV - {ad}|", fontsize=10)
        fig.colorbar(goruntu, ax=fark_ax, fraction=0.046)
    for ax in eksenler.ravel():
        ax.axis("off")
    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "gri_karsilastirma.png")
    fig.savefig(yol, dpi=100)
    print("Kaydedildi:", yol)
    plt.show()


if __name__ == "__main__":
    main()
