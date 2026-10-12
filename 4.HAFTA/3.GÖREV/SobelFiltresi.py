"""
Görüntü İşleme Ödevi - Gauss Yumuşatma + 3x3 Yatay ve Dikey Sobel Filtreleri
----------------------------------------------------------------------------
Tek kanallı (gri) bir görüntüyü OpenCV ile açar ve iki adımda işler:

  1) Gauss filtresi ile hafif yumuşatma (3x3, sigma ~ 0.85):

                 | 1 2 1 |
        G = 1/16 | 2 4 2 |
                 | 1 2 1 |

     Sobel bir türev filtresidir; türev, gürültüdeki küçük piksel
     dalgalanmalarını da büyütür. Önce yumuşatınca bu sahte kenarlar azalır.

  2) 3x3 Sobel filtreleri:

             | -1  0  1 |                 | -1 -2 -1 |
        Sx = | -2  0  2 |            Sy = |  0  0  0 |
             | -1  0  1 |                 |  1  2  1 |

     Sx yatay yöndeki parlaklık değişimini ölçer -> DİKEY kenarları bulur.
     Sy dikey yöndeki parlaklık değişimini ölçer -> YATAY kenarları bulur.

     Prewitt'ten tek farkı orta satır/sütunun 2 ile ağırlıklandırılmasıdır:
     Sobel = [1 2 1] yumuşatma x [-1 0 1] türev. Yani türeve dik yönde
     küçük bir Gauss benzeri yumuşatma zaten içinde vardır.

     İki yönün sonucu birleştirilerek kenar büyüklüğü elde edilir:

        |G| = sqrt(Sx^2 + Sy^2)

Tüm filtreler konvolüsyonla, döngüyle piksel piksel uygulanır;
cv2.GaussianBlur, cv2.filter2D, cv2.Sobel benzeri hazır fonksiyonlar
yalnızca sonucu doğrulamak için çağrılır.

Karşılaştırma için Sobel, yumuşatılmamış resme de uygulanır.

Varsayılan girdi 2. haftadaki (1. görev) cameraman resmidir.

Kullanım:
    python SobelFiltresi.py [resim_yolu]

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

# Yatay Sobel (x yönünde türev, dikey kenarlar), toplamı 0
SOBEL_X = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

# Dikey Sobel (y yönünde türev, yatay kenarlar), toplamı 0
SOBEL_Y = [[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]]

# |Sobel| bu değeri aşan piksel "kenar" sayılır (sadece istatistik için)
KENAR_ESIGI = 80


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

    Sonuç float döner; Sobel'de negatif değerler de çıkabildiği için
    uint8'e çevirme işi çağıran tarafa bırakılır.

    Konvolüsyonda çekirdek 180 derece çevrilerek uygulanır. Gauss çekirdeği
    simetrik olduğu için fark etmez; Sobel çekirdekleri çevrilince işaretleri
    tersine döner. Kenarın büyüklüğü |sonuç| aynı kalır.
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


def sobel(resim):
    """Yatay ve dikey Sobel konvolüsyonları; işaretli ham türevler (-1020 .. +1020) döner."""
    return konvolusyon(resim, SOBEL_X), konvolusyon(resim, SOBEL_Y)


def buyukluk(gx, gy):
    """İki yönlü türevden kenar büyüklüğü: sqrt(gx^2 + gy^2)."""
    return np.sqrt(gx ** 2 + gy ** 2)


def mutlak_goruntu(turev):
    """|türev| değerini 255'te kırparak gösterilebilir resme çevirir (cv2.convertScaleAbs gibi)."""
    return np.clip(np.abs(turev), 0, 255).astype(np.uint8)


def isaretli_goruntu(turev):
    """İşaretli türevi gri zemin üzerinde gösterir: 0 -> 128, negatif koyu, pozitif açık."""
    return np.clip(128 + turev / 4, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2) Sonuçları gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def istatistik_yazdir(ad, deger):
    mutlak = np.abs(deger)
    print(f"{ad:<26} -> ortalama: {mutlak.mean():6.2f} | en büyük: {mutlak.max():6.0f} | "
          f"> {KENAR_ESIGI} olan piksel: {np.count_nonzero(mutlak > KENAR_ESIGI)}")


def dogrula(ad, benim, cekirdek, girdi):
    """cv2.filter2D çekirdeği çevirmez (korelasyon yapar); konvolüsyonla aynı
    sonucu vermesi için çekirdeği 180 derece çevirip veriyoruz."""
    cevrik = cv2.flip(np.array(cekirdek, np.float64), -1)
    hazir = cv2.filter2D(girdi.astype(np.float64), -1, cevrik)
    fark = np.abs(benim - hazir)
    print(f"Kontrol -> cv2.filter2D ({ad}) ile en büyük fark: {fark.max():.0f} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")


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

    # 3) Sobel'i yumuşatılmış resme (ve karşılaştırma için ham resme) uygula
    baslangic = time.perf_counter()
    gx, gy = sobel(yumusak)
    print(f"Sobel (x ve y) konvolüsyon süresi: {time.perf_counter() - baslangic:.2f} s")
    gx_ham, gy_ham = sobel(resim)
    g = buyukluk(gx, gy)
    g_ham = buyukluk(gx_ham, gy_ham)

    # Sadece doğrulama amaçlı: OpenCV'nin hazır sonuçlarıyla karşılaştır.
    hazir_gauss = cv2.GaussianBlur(resim, (3, 3), 0)
    fark = np.abs(yumusak.astype(int) - hazir_gauss.astype(int))
    print(f"Kontrol -> cv2.GaussianBlur ile en büyük piksel farkı: {fark.max()} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")
    dogrula("Sobel X", gx, SOBEL_X, yumusak)
    dogrula("Sobel Y", gy, SOBEL_Y, yumusak)

    # cv2.Sobel korelasyon yapar; bizim konvolüsyon sonucumuzun tam tersi işaretlidir
    hazir_sobel_x = cv2.Sobel(yumusak, cv2.CV_64F, 1, 0, ksize=3)
    hazir_sobel_y = cv2.Sobel(yumusak, cv2.CV_64F, 0, 1, ksize=3)
    print(f"Kontrol -> cv2.Sobel ile en büyük fark: "
          f"x: {np.abs(gx + hazir_sobel_x).max():.0f}, y: {np.abs(gy + hazir_sobel_y).max():.0f}")

    # 4) Yumuşatmanın etkisi: sahte (küçük) kenar tepkileri azalmalı
    istatistik_yazdir("|Sobel X| (ham resim)", gx_ham)
    istatistik_yazdir("|Sobel X| (Gauss sonrası)", gx)
    istatistik_yazdir("|Sobel Y| (ham resim)", gy_ham)
    istatistik_yazdir("|Sobel Y| (Gauss sonrası)", gy)
    istatistik_yazdir("Büyüklük (ham resim)", g_ham)
    istatistik_yazdir("Büyüklük (Gauss sonrası)", g)

    kenar_x = mutlak_goruntu(gx)
    kenar_y = mutlak_goruntu(gy)
    kenar = mutlak_goruntu(g)
    kaydet(os.path.join(CIKTI_KLASORU, "gauss_yumusatilmis.png"), yumusak)
    kaydet(os.path.join(CIKTI_KLASORU, "sobel_x.png"), kenar_x)
    kaydet(os.path.join(CIKTI_KLASORU, "sobel_y.png"), kenar_y)
    kaydet(os.path.join(CIKTI_KLASORU, "sobel_buyukluk.png"), kenar)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(yumusak, "Gauss 3x3"),
                     etiket_ekle(mutlak_goruntu(g_ham), "|G| (yumusatmasiz)")])
    orta = np.hstack([etiket_ekle(isaretli_goruntu(gx), "Sobel X (isaretli)"),
                      etiket_ekle(kenar_x, "|Sobel X| dikey kenar"),
                      etiket_ekle(mutlak_goruntu(gx_ham), "|Sobel X| yumusatmasiz")])
    alt = np.hstack([etiket_ekle(isaretli_goruntu(gy), "Sobel Y (isaretli)"),
                     etiket_ekle(kenar_y, "|Sobel Y| yatay kenar"),
                     etiket_ekle(mutlak_goruntu(gy_ham), "|Sobel Y| yumusatmasiz")])
    son = np.hstack([etiket_ekle(kenar, "|G| = sqrt(X^2+Y^2)"),
                     etiket_ekle(np.where(g > KENAR_ESIGI, 255, 0).astype(np.uint8),
                                 f"|G| > {KENAR_ESIGI}"),
                     etiket_ekle(np.where(g_ham > KENAR_ESIGI, 255, 0).astype(np.uint8),
                                 f"|G| > {KENAR_ESIGI} yumusatmasiz")])
    karsilastirma = np.vstack([ust, orta, alt, son])
    kaydet(os.path.join(CIKTI_KLASORU, "sobel_karsilastirma.png"), karsilastirma)

    pencere = "Gauss + 3x3 yatay/dikey Sobel filtresi"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
