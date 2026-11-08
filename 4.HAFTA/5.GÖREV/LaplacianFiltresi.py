"""
Görüntü İşleme Ödevi - Gauss Yumuşatma + 3x3 Laplacian ile Kenar Bulma
----------------------------------------------------------------------
Tek kanallı (gri) bir görüntüyü OpenCV ile açar ve üç adımda işler:

  1) Gauss filtresi ile hafif yumuşatma (3x3, sigma ~ 0.85):

                 | 1 2 1 |
        G = 1/16 | 2 4 2 |
                 | 1 2 1 |

     Laplacian ikinci türevdir; gürültüyü birinci türev filtrelerinden
     (Sobel, Prewitt) de fazla büyütür. Bu yüzden önce yumuşatmak şarttır.
     Gauss + Laplacian birlikte "LoG" (Laplacian of Gaussian) olarak bilinir.

  2) 3x3 Laplacian filtresi:

            | 0  1  0 |
        L = | 1 -4  1 |      L = d2f/dx2 + d2f/dy2
            | 0  1  0 |

     Yatay [1 -2 1] ve dikey [1 -2 1] ikinci türevlerin toplamıdır.
     Düz bölgede ve sabit eğimli rampada 0 verir. Bir kenarın bir yanında
     pozitif, öbür yanında negatif değer üretir; kenarın kendisi, işaretin
     değiştiği yerdedir (sıfır geçişi).

  3) Sıfır geçişi (zero crossing) ile kenar bulma:
     Bir piksel ile sağ ya da alt komşusu arasında Laplacian işaret
     değiştiriyorsa ve bu sıçrama yeterince büyükse (> SIFIR_GECIS_ESIGI)
     o piksel kenar sayılır. Böylece kenarlar 1 piksel inceliğinde çıkar.
     Eşik, gürültüden doğan küçük işaret değişimlerini eler.

Filtreler konvolüsyonla, döngüyle piksel piksel uygulanır;
cv2.GaussianBlur, cv2.filter2D, cv2.Laplacian benzeri hazır fonksiyonlar
yalnızca sonucu doğrulamak için çağrılır.

Karşılaştırma için Laplacian, yumuşatılmamış resme de uygulanır.

Varsayılan girdi 2. haftadaki (1. görev) cameraman resmidir.

Kullanım:
    python LaplacianFiltresi.py [resim_yolu]

Herhangi bir tuşa basınca pencere kapanır.
"""

import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
KAYNAK_RESIM = os.path.join(KLASOR, "..", "..", "2.HAFTA", "1.GÖREV", "ciktilar", "cameraman.png")

# 3x3 Gauss çekirdeği (binom katsayıları 1-2-1'in dış çarpımı), toplamı 1
GAUSS_CEKIRDEGI = [[1 / 16, 2 / 16, 1 / 16],
                   [2 / 16, 4 / 16, 2 / 16],
                   [1 / 16, 2 / 16, 1 / 16]]

# 3x3 Laplacian (4 komşulu), toplamı 0 (düz bölgede sonuç 0)
LAPLACIAN_CEKIRDEGI = [[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]]

# Sıfır geçişinde iki komşu arasındaki Laplacian farkı bunu aşmalı
SIFIR_GECIS_ESIGI = 24


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma / yazma yardımcıları
# ---------------------------------------------------------------------------
def gri_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


def kaydet(yol, resim):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    cv2.imencode(".png", resim)[1].tofile(yol)
    print("Kaydedildi:", yol)


# ---------------------------------------------------------------------------
# 1) Konvolüsyon (hazır fonksiyon yok)
# ---------------------------------------------------------------------------
def yansit(i, n):
    """Resmin dışına taşan indeksi kenardan geri yansıtır (OpenCV'deki BORDER_REFLECT_101).

    -1 -> 1,  n -> n-2  (kenar pikselinin kendisi tekrar edilmez)
    """
    if i < 0:
        return -i
    if i >= n:
        return 2 * n - 2 - i
    return i


def konvolusyon(resim, cekirdek):
    """Çekirdeği resmin her pikselinin üzerine oturtup ağırlıklı toplamı alır.

        sonuc(y, x) = toplam  K(i, j) * resim(y - i, x - j)      i, j = -1..1

    Sonuç float döner; Laplacian'da negatif değerler de çıkabildiği için
    uint8'e çevirme işi çağıran tarafa bırakılır.

    Gauss ve Laplacian çekirdekleri simetrik olduğundan konvolüsyondaki
    180 derece çevirme sonucu değiştirmez.
    """
    yukseklik, genislik = resim.shape
    yaricap = len(cekirdek) // 2           # 3x3 için 1
    sonuc = np.zeros((yukseklik, genislik), dtype=np.float64)

    for y in range(yukseklik):
        for x in range(genislik):
            toplam = 0.0
            for i in range(-yaricap, yaricap + 1):
                yy = yansit(y - i, yukseklik)
                for j in range(-yaricap, yaricap + 1):
                    xx = yansit(x - j, genislik)
                    toplam += cekirdek[i + yaricap][j + yaricap] * int(resim[yy, xx])
            sonuc[y, x] = toplam
    return sonuc


def gauss_yumusat(resim):
    """3x3 Gauss konvolüsyonu; sonuç yine 0-255 arası bir resim (uint8)."""
    # x.5 değerleri yukarı yuvarlanır (np.round ise çift sayıya yuvarlardı)
    return np.clip(np.floor(konvolusyon(resim, GAUSS_CEKIRDEGI) + 0.5), 0, 255).astype(np.uint8)


def laplacian(resim):
    """3x3 Laplacian konvolüsyonu; işaretli ham ikinci türev (-1020 .. +1020) döner."""
    return konvolusyon(resim, LAPLACIAN_CEKIRDEGI)


# ---------------------------------------------------------------------------
# 2) Sıfır geçişi ile kenar bulma
# ---------------------------------------------------------------------------
def sifir_gecisleri(lap, esik=SIFIR_GECIS_ESIGI):
    """Laplacian'ın işaret değiştirdiği yerleri kenar olarak işaretler.

    Her piksel sağ ve alt komşusuyla kıyaslanır. İşaretler zıtsa ve
    aradaki fark eşikten büyükse, sıfıra (kenara) daha yakın olan piksel
    işaretlenir; böylece kenar tek piksel kalınlığında olur.
    """
    kenar = np.zeros(lap.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = lap[:lap.shape[0] - dy, :lap.shape[1] - dx]
        b = lap[dy:, dx:]
        gecis = (a * b < 0) & (np.abs(a - b) > esik)
        a_yakin = gecis & (np.abs(a) <= np.abs(b))
        b_yakin = gecis & (np.abs(a) > np.abs(b))
        kenar[:lap.shape[0] - dy, :lap.shape[1] - dx] |= a_yakin
        kenar[dy:, dx:] |= b_yakin
    return kenar


# ---------------------------------------------------------------------------
# 3) Sonuçları gösterme
# ---------------------------------------------------------------------------
def mutlak_goruntu(lap, kazanc=4):
    """|Laplacian| değerini kazançla çarpıp 255'te kırparak gösterilebilir resme çevirir.

    Yumuşatılmış resimde |L| küçük kaldığı için (çoğu < 60) kazanç olmadan
    kenarlar zor seçilir.
    """
    return np.clip(np.abs(lap) * kazanc, 0, 255).astype(np.uint8)


def isaretli_goruntu(lap):
    """İşaretli Laplacian'ı gri zemin üzerinde gösterir: 0 -> 128, negatif koyu, pozitif açık."""
    return np.clip(128 + lap, 0, 255).astype(np.uint8)


def kenar_goruntu(kenar):
    return np.where(kenar, 255, 0).astype(np.uint8)


def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def istatistik_yazdir(ad, lap, kenar):
    mutlak = np.abs(lap)
    print(f"{ad:<24} -> ortalama |L|: {mutlak.mean():6.2f} | en büyük |L|: {mutlak.max():5.0f} | "
          f"sıfır geçişi kenar pikseli: {np.count_nonzero(kenar)}")


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM

    # 1) Tek kanallı resmi OpenCV ile aç
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape, "| Kanal sayısı: 1")

    # 2) Gauss ile hafif yumuşat
    baslangic = time.perf_counter()
    yumusak = gauss_yumusat(resim)
    print(f"Gauss konvolüsyonu süresi: {time.perf_counter() - baslangic:.2f} s")

    # 3) Laplacian'ı yumuşatılmış resme (ve karşılaştırma için ham resme) uygula
    baslangic = time.perf_counter()
    lap = laplacian(yumusak)
    print(f"Laplacian konvolüsyonu süresi: {time.perf_counter() - baslangic:.2f} s")
    lap_ham = laplacian(resim)

    # Sadece doğrulama amaçlı: OpenCV'nin hazır sonuçlarıyla karşılaştır.
    hazir_gauss = cv2.GaussianBlur(resim, (3, 3), 0)
    fark = np.abs(yumusak.astype(int) - hazir_gauss.astype(int))
    print(f"Kontrol -> cv2.GaussianBlur ile en büyük piksel farkı: {fark.max()} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")
    # ksize=1 iken cv2.Laplacian tam olarak yukarıdaki 4 komşulu çekirdeği kullanır
    hazir_lap = cv2.Laplacian(yumusak, cv2.CV_64F, ksize=1)
    fark = np.abs(lap - hazir_lap)
    print(f"Kontrol -> cv2.Laplacian ile en büyük fark: {fark.max():.0f} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")

    # 4) Sıfır geçişleriyle kenarları bul
    kenar = sifir_gecisleri(lap)
    kenar_ham = sifir_gecisleri(lap_ham)
    kenar_esiksiz = sifir_gecisleri(lap, esik=0)
    istatistik_yazdir("Laplacian (ham resim)", lap_ham, kenar_ham)
    istatistik_yazdir("Gauss + Laplacian", lap, kenar)
    print(f"Eşiksiz sıfır geçişi (Gauss sonrası): {np.count_nonzero(kenar_esiksiz)} piksel")

    kaydet(os.path.join(CIKTI_KLASORU, "gauss_yumusatilmis.png"), yumusak)
    kaydet(os.path.join(CIKTI_KLASORU, "laplacian_sonuc.png"), mutlak_goruntu(lap))
    kaydet(os.path.join(CIKTI_KLASORU, "laplacian_kenarlar.png"), kenar_goruntu(kenar))

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(yumusak, "Gauss 3x3"),
                     etiket_ekle(isaretli_goruntu(lap), "Gauss+Laplacian (isaretli)")])
    orta = np.hstack([etiket_ekle(mutlak_goruntu(lap_ham), "|Laplacian| x4 (yumusatmasiz)"),
                      etiket_ekle(mutlak_goruntu(lap), "|Gauss+Laplacian| x4"),
                      etiket_ekle(kenar_goruntu(kenar_esiksiz), "Sifir gecisi (esiksiz)")])
    alt = np.hstack([etiket_ekle(kenar_goruntu(kenar_ham), f"Sifir gecisi > {SIFIR_GECIS_ESIGI} yumusatmasiz"),
                     etiket_ekle(kenar_goruntu(kenar), f"Sifir gecisi > {SIFIR_GECIS_ESIGI}"),
                     etiket_ekle(np.where(kenar[..., None], [0, 0, 255], cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR))
                                 .astype(np.uint8), "Kenarlar resim uzerinde")])
    karsilastirma = np.vstack([ust, orta, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "laplacian_karsilastirma.png"), karsilastirma)

    pencere = "Gauss + 3x3 Laplacian filtresi"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
