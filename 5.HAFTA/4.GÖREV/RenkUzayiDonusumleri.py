"""
Görüntü İşleme Ödevi - Renk Uzayı Dönüşümlerini Elle Yazma
----------------------------------------------------------
Bir renk uzayındaki görüntüyü başka bir renk uzayına dönüştüren
fonksiyonları baştan yazar ve sonuçları OpenCV'nin cv2.cvtColor çıktısıyla
sayısal olarak karşılaştırır. Beklenen hata sıfırdır.

Yazılan dönüşümler (hepsi 8 bit, uint8 girdi -> uint8 çıktı):

    BGR -> GRAY        Y = 0.299 R + 0.587 G + 0.114 B
    BGR -> YCrCb       Y, Cr = 0.713 (R - Y) + 128, Cb = 0.564 (B - Y) + 128
    BGR -> YUV         Y, U = 0.492 (B - Y) + 128,  V = 0.877 (R - Y) + 128
    BGR -> XYZ         sRGB (D65) -> CIE XYZ, 3x3 matris
    BGR -> HSV         H: 0..179 (derece / 2), S, V: 0..255
    BGR -> HSV_FULL    H: 0..255 (360 derece 256'ya ölçeklenir)
    YCrCb -> BGR       ters dönüşüm

Hata neden tam olarak sıfır olabiliyor?
  OpenCV 8 bitlik dönüşümleri kayan noktayla değil, sabit noktalı (fixed-
  point) tamsayı aritmetiğiyle yapar: katsayılar 2^n ile çarpılıp
  yuvarlanır, sonuç (x + 2^(n-1)) >> n ile geri ölçeklenir. Kayan noktalı
  formül bazı piksellerde 1 farklı sonuç verir. Bu yüzden fonksiyonlar
  OpenCV'nin kullandığı katsayılar ve kaydırmalarla yazıldı:

    GRAY  : 15 bit, R 9798, G 19235, B 3735
    YCrCb : 14 bit, Y için R 4899, G 9617, B 1868; Cr 11682, Cb 9241
    YUV   : 14 bit, aynı Y; U 8061, V 14369
    XYZ   : 12 bit, matris katsayıları round(m * 4096)
    HSV   : S ve H için önceden hesaplanmış bölme tabloları (12 bit):
            S = diff * round(255 * 2^12 / V) >> 12
            H = h' * round(H_aralığı * 2^12 / (6 * diff)) >> 12
    YCrCb -> BGR : 14 bit, Cr->R 22987, Cr->G -11698, Cb->G -5636, Cb->B 29049

"Fonksiyon pointer" karşılığı:
  C++'ta dönüşümler, aynı imzaya sahip fonksiyonların adreslerini tutan
  bir fonksiyon pointer dizisiyle seçilirdi:

      typedef void (*DonusumFn)(const Mat&, Mat&);
      DonusumFn tablo[] = { bgr2gray, bgr2ycrcb, ... };
      tablo[i](girdi, cikti);

  Python'da fonksiyonlar birinci sınıf nesnelerdir: bir fonksiyonun adı
  (parantezsiz) o fonksiyona bir referanstır ve pointer gibi listeye
  konup başka fonksiyona parametre olarak verilebilir. DONUSUMLER
  tablosu (ad, kendi_fonksiyonumuz, opencv_kodu) üçlülerini tutar.
  donustur(resim, fonksiyon) da hangi dönüşümü çalıştıracağını bilmeden,
  aldığı referansı çağırır.

İki test yapılır:
  1) baboon.jpg üzerinde (gerçek resim)
  2) Bütün 2^24 = 16 777 216 BGR renginin her birini bir kez içeren
     4096x4096'lık yapay resim üzerinde. Böylece "bu resimde tutuyor"
     değil, "her olası girdide tutuyor" gösterilmiş olur.

Kullanım:
    python RenkUzayiDonusumleri.py [resim_yolu]
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


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma yardımcısı
# ---------------------------------------------------------------------------
def renkli_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def kanallar(resim):
    """uint8 resmin üç kanalını, taşma olmasın diye int32 olarak döndürür."""
    return [resim[:, :, i].astype(np.int32) for i in range(3)]


def olcekle(x, n):
    """Sabit noktadan geri ölçekleme, en yakına yuvarlayarak: (x + 2^(n-1)) >> n"""
    return (x + (1 << (n - 1))) >> n


def doyur(x):
    """0..255 dışına taşan değerleri kırpıp uint8'e çevirir (saturate_cast)."""
    return np.clip(x, 0, 255).astype(np.uint8)


def birlestir(*kanal_listesi):
    return np.stack(kanal_listesi, axis=-1)


# ---------------------------------------------------------------------------
# Dönüşüm fonksiyonları (hepsi aynı imza: uint8 resim -> uint8 resim)
# ---------------------------------------------------------------------------
# Y (luma) katsayıları, 14 bit: YCrCb ve YUV için
Y_R14, Y_G14, Y_B14 = 4899, 9617, 1868


def bgr2gray(resim):
    b, g, r = kanallar(resim)
    return doyur(olcekle(b * 3735 + g * 19235 + r * 9798, 15))


def _luma14(b, g, r):
    return olcekle(b * Y_B14 + g * Y_G14 + r * Y_R14, 14)


def bgr2ycrcb(resim):
    b, g, r = kanallar(resim)
    y = _luma14(b, g, r)
    orta = 128 << 14                                  # 128 ofseti, sabit noktada
    cr = olcekle((r - y) * 11682 + orta, 14)          # 0.713 * 2^14
    cb = olcekle((b - y) * 9241 + orta, 14)           # 0.564 * 2^14
    return birlestir(doyur(y), doyur(cr), doyur(cb))


def bgr2yuv(resim):
    b, g, r = kanallar(resim)
    y = _luma14(b, g, r)
    orta = 128 << 14
    u = olcekle((b - y) * 8061 + orta, 14)            # 0.492 * 2^14
    v = olcekle((r - y) * 14369 + orta, 14)           # 0.877 * 2^14
    return birlestir(doyur(y), doyur(u), doyur(v))


# sRGB (D65) -> XYZ matrisi
XYZ_MATRIS = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]
XYZ_TAM = [[int(round(m * (1 << 12))) for m in satir] for satir in XYZ_MATRIS]


def bgr2xyz(resim):
    b, g, r = kanallar(resim)
    sonuc = [doyur(olcekle(r * kr + g * kg + b * kb, 12)) for kr, kg, kb in XYZ_TAM]
    return birlestir(*sonuc)


# HSV bölme tabloları: döngüde bölme yapmamak için önceden hesaplanır (12 bit)
S_TABLOSU = np.array([0] + [int(round((255 << 12) / v)) for v in range(1, 256)], np.int64)


def _h_tablosu(h_araligi):
    return np.array([0] + [int(round((h_araligi << 12) / (6.0 * d))) for d in range(1, 256)], np.int64)


H_TABLOSU_180 = _h_tablosu(180)
H_TABLOSU_256 = _h_tablosu(256)


def _bgr2hsv(resim, h_araligi, h_tablosu):
    b, g, r = [k.astype(np.int64) for k in kanallar(resim)]
    v = np.maximum(np.maximum(b, g), r)
    fark = v - np.minimum(np.minimum(b, g), r)
    s = olcekle(fark * S_TABLOSU[v], 12)
    # Ton: en büyük kanal R ise (G - B), G ise (B - R) + 2 fark, B ise (R - G) + 4 fark
    # (fark birimi 60 dereceye karşılık gelir)
    h = np.where(v == r, g - b, np.where(v == g, b - r + 2 * fark, r - g + 4 * fark))
    h = olcekle(h * h_tablosu[fark], 12)
    h = np.where(h < 0, h + h_araligi, h)
    return birlestir(doyur(h), doyur(s), doyur(v))


def bgr2hsv(resim):
    return _bgr2hsv(resim, 180, H_TABLOSU_180)


def bgr2hsv_full(resim):
    return _bgr2hsv(resim, 256, H_TABLOSU_256)


def ycrcb2bgr(resim):
    y, cr, cb = kanallar(resim)
    cr, cb = cr - 128, cb - 128
    b = y + olcekle(cb * 29049, 14)                    # 1.773 * 2^14
    g = y + olcekle(cb * -5636 + cr * -11698, 14)      # -0.344, -0.714
    r = y + olcekle(cr * 22987, 14)                    # 1.403 * 2^14
    return birlestir(doyur(b), doyur(g), doyur(r))


# ---------------------------------------------------------------------------
# "Fonksiyon pointer" tablosu: (ad, kendi fonksiyonumuzun referansı, OpenCV kodu)
# ---------------------------------------------------------------------------
DONUSUMLER = [
    ("BGR -> GRAY", bgr2gray, cv2.COLOR_BGR2GRAY),
    ("BGR -> YCrCb", bgr2ycrcb, cv2.COLOR_BGR2YCrCb),
    ("BGR -> YUV", bgr2yuv, cv2.COLOR_BGR2YUV),
    ("BGR -> XYZ", bgr2xyz, cv2.COLOR_BGR2XYZ),
    ("BGR -> HSV", bgr2hsv, cv2.COLOR_BGR2HSV),
    ("BGR -> HSV_FULL", bgr2hsv_full, cv2.COLOR_BGR2HSV_FULL),
    ("YCrCb -> BGR", ycrcb2bgr, cv2.COLOR_YCrCb2BGR),
]


def donustur(resim, donusum_fonksiyonu):
    """Hangi dönüşüm olduğunu bilmeden, verilen fonksiyon referansını çağırır
    (C++'taki fonksiyon pointer'ı üzerinden çağrının karşılığı)."""
    return donusum_fonksiyonu(resim)


def hata_olc(kendi, opencv):
    """Kanal başına mutlak fark toplamı, toplam SAD, en büyük fark, farklı piksel sayısı."""
    fark = np.abs(kendi.astype(np.int32) - opencv.astype(np.int32))
    if fark.ndim == 2:
        fark = fark[:, :, None]
    kanal_sad = [int(fark[:, :, i].sum()) for i in range(fark.shape[2])]
    return kanal_sad, sum(kanal_sad), int(fark.max()), int(np.count_nonzero(fark.max(axis=2)))


def tum_renkler_resmi():
    """2^24 BGR renginin her birini bir kez içeren 4096x4096 resim."""
    sayi = np.arange(1 << 24, dtype=np.uint32)
    return np.stack([(sayi >> 16) & 255, (sayi >> 8) & 255, sayi & 255],
                    axis=-1).astype(np.uint8).reshape(4096, 4096, 3)


def test_et(baslik, resim):
    print(f"\n{baslik}  ({resim.shape[0]}x{resim.shape[1]} = {resim.shape[0] * resim.shape[1]} piksel)")
    print(f"{'Dönüşüm':<18}{'Kanal SAD (1, 2, 3)':<26}{'Toplam SAD':>11}{'En büyük':>10}"
          f"{'Farklı px':>11}{'Kendi':>10}{'OpenCV':>9}")
    toplam = 0
    sonuclar = {}
    for ad, fonksiyon, kod in DONUSUMLER:
        t0 = time.perf_counter()
        kendi = donustur(resim, fonksiyon)
        t1 = time.perf_counter()
        opencv = cv2.cvtColor(resim, kod)
        t2 = time.perf_counter()
        assert kendi.shape == opencv.shape and kendi.dtype == opencv.dtype, ad
        kanal_sad, sad, en_buyuk, farkli = hata_olc(kendi, opencv)
        toplam += sad
        sonuclar[ad] = (kendi, opencv)
        print(f"{ad:<18}{str(kanal_sad):<26}{sad:>11}{en_buyuk:>10}{farkli:>11}"
              f"{(t1 - t0) * 1000:>8.1f}ms{(t2 - t1) * 1000:>7.1f}ms")
    print(f"Bütün dönüşümlerde toplam hata: {toplam}")
    return sonuclar, toplam


# ---------------------------------------------------------------------------
# Çizim
# ---------------------------------------------------------------------------
KANAL_ADLARI = {"BGR -> YCrCb": ("Y", "Cr", "Cb"), "BGR -> YUV": ("Y", "U", "V"),
                "BGR -> XYZ": ("X", "Y", "Z"), "BGR -> HSV": ("H", "S", "V"),
                "BGR -> HSV_FULL": ("H", "S", "V"), "YCrCb -> BGR": ("B", "G", "R")}


def ciz_ve_kaydet(resim, sonuclar):
    satirlar = [ad for ad, _, _ in DONUSUMLER if ad in KANAL_ADLARI]
    fig, eksenler = plt.subplots(len(satirlar) + 1, 4, figsize=(13, 3.1 * (len(satirlar) + 1)))
    eksenler[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    eksenler[0, 0].set_title("Orijinal")
    gri_kendi, gri_cv = sonuclar["BGR -> GRAY"]
    eksenler[0, 1].imshow(gri_kendi, cmap="gray", vmin=0, vmax=255)
    eksenler[0, 1].set_title("GRAY (kendi)")
    eksenler[0, 2].imshow(gri_cv, cmap="gray", vmin=0, vmax=255)
    eksenler[0, 2].set_title("GRAY (OpenCV)")
    fark = np.abs(gri_kendi.astype(int) - gri_cv.astype(int))
    eksenler[0, 3].imshow(fark, cmap="magma", vmin=0, vmax=1)
    eksenler[0, 3].set_title(f"|fark|, toplam = {fark.sum()}")

    for satir, ad in enumerate(satirlar, start=1):
        kendi, opencv = sonuclar[ad]
        for k in range(3):
            eksenler[satir, k].imshow(kendi[:, :, k], cmap="gray", vmin=0, vmax=255)
            eksenler[satir, k].set_title(f"{ad}: {KANAL_ADLARI[ad][k]} (kendi)", fontsize=9)
        fark = np.abs(kendi.astype(int) - opencv.astype(int)).sum(axis=2)
        eksenler[satir, 3].imshow(fark, cmap="magma", vmin=0, vmax=1)
        eksenler[satir, 3].set_title(f"|kendi - OpenCV|, toplam = {fark.sum()}", fontsize=9)
    for ax in eksenler.ravel():
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = os.path.join(CIKTI_KLASORU, "renk_uzaylari.png")
    fig.savefig(yol, dpi=90)
    print("Kaydedildi:", yol)
    plt.show()


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = renkli_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")

    sonuclar, _ = test_et(f"1) Resim: {os.path.basename(yol)}", resim)
    test_et("2) Bütün 2^24 BGR rengi", tum_renkler_resmi())
    ciz_ve_kaydet(resim, sonuclar)


if __name__ == "__main__":
    main()
