"""
Görüntü İşleme Ödevi - Gauss Yumuşatma + 3x3 Prewitt Filtresi (Yatay Kenar Bulma)
----------------------------------------------------------------------------------
Tek kanallı (gri) bir görüntüyü OpenCV ile açar ve iki adımda işler:

  1) Gauss filtresi ile hafif yumuşatma (3x3, sigma ~ 0.85):

                 | 1 2 1 |
        G = 1/16 | 2 4 2 |
                 | 1 2 1 |

     Prewitt bir türev filtresidir; türev, gürültüdeki küçük piksel
     dalgalanmalarını da büyütür. Önce yumuşatınca bu sahte kenarlar azalır.

  2) 3x3 Prewitt filtresi:

            | -1 -1 -1 |
        P = |  0  0  0 |      sonuç ~ alttaki satır - üstteki satır
            |  1  1  1 |

     Dikey yöndeki parlaklık değişimini (dikey türevi) ölçer, bu yüzden
     YATAY kenarlarda büyük, düz bölgelerde ~0 değer verir. Sonuç negatif
     de olabilir (parlaktan koyuya geçiş); gösterirken mutlak değeri alınır.
     (1. görevdeki [-1 0 1] çekirdeğinin 90 derece döndürülmüş hâlidir.)

Her iki filtre de konvolüsyonla, döngüyle piksel piksel uygulanır;
cv2.GaussianBlur, cv2.filter2D, cv2.Prewitt benzeri hazır fonksiyonlar
yalnızca sonucu doğrulamak için çağrılır.

Karşılaştırma için Prewitt, yumuşatılmamış resme de uygulanır.

Varsayılan girdi 2. haftadaki (1. görev) cameraman resmidir.

Kullanım:
    python PrewittYatayKenar.py [resim_yolu]

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

# Hocanın verdiği 3x3 Prewitt çekirdeği, toplamı 0 (düz bölgede sonuç 0)
PREWITT_CEKIRDEGI = [[-1, -1, -1],
                     [ 0,  0,  0],
                     [ 1,  1,  1]]

# |Prewitt| bu değeri aşan piksel "kenar" sayılır (sadece istatistik için)
KENAR_ESIGI = 60


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

    Sonuç float döner; Prewitt'te negatif değerler de çıkabildiği için
    uint8'e çevirme işi çağıran tarafa bırakılır.

    Konvolüsyonda çekirdek 180 derece çevrilerek uygulanır. Gauss çekirdeği
    simetrik olduğu için fark etmez; Prewitt çekirdeği çevrilince işareti
    tersine döner (üst - alt olur). Kenarın büyüklüğü |sonuç| aynı kalır.
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


def prewitt(resim):
    """3x3 Prewitt konvolüsyonu; işaretli ham türev (-765 .. +765) döner."""
    return konvolusyon(resim, PREWITT_CEKIRDEGI)


def mutlak_goruntu(turev):
    """|türev| değerini 255'te kırparak gösterilebilir resme çevirir (cv2.convertScaleAbs gibi)."""
    return np.clip(np.abs(turev), 0, 255).astype(np.uint8)


def isaretli_goruntu(turev):
    """İşaretli türevi gri zemin üzerinde gösterir: 0 -> 128, negatif koyu, pozitif açık."""
    return np.clip(128 + turev / 2, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2) Sonuçları gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def istatistik_yazdir(ad, turev):
    mutlak = np.abs(turev)
    print(f"{ad:<22} -> ortalama |P|: {mutlak.mean():6.2f} | en büyük |P|: {mutlak.max():5.0f} | "
          f"|P| > {KENAR_ESIGI} olan piksel: {np.count_nonzero(mutlak > KENAR_ESIGI)}")


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

    # 3) Prewitt'i yumuşatılmış resme (ve karşılaştırma için ham resme) uygula
    baslangic = time.perf_counter()
    turev_yumusak = prewitt(yumusak)
    print(f"Prewitt konvolüsyonu süresi: {time.perf_counter() - baslangic:.2f} s")
    turev_ham = prewitt(resim)

    # Sadece doğrulama amaçlı: OpenCV'nin hazır sonuçlarıyla karşılaştır.
    # cv2.filter2D çekirdeği çevirmez (korelasyon yapar); konvolüsyonla aynı
    # sonucu vermesi için çekirdeği 180 derece çevirip veriyoruz.
    hazir_gauss = cv2.GaussianBlur(resim, (3, 3), 0)
    fark = np.abs(yumusak.astype(int) - hazir_gauss.astype(int))
    print(f"Kontrol -> cv2.GaussianBlur ile en büyük piksel farkı: {fark.max()} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")

    cevrik_prewitt = cv2.flip(np.array(PREWITT_CEKIRDEGI, np.float64), -1)
    hazir_prewitt = cv2.filter2D(yumusak.astype(np.float64), -1, cevrik_prewitt)
    fark = np.abs(turev_yumusak - hazir_prewitt)
    print(f"Kontrol -> cv2.filter2D (Prewitt) ile en büyük fark: {fark.max():.0f} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")

    # 4) Yumuşatmanın etkisi: sahte (küçük) kenar tepkileri azalmalı
    istatistik_yazdir("Prewitt (ham resim)", turev_ham)
    istatistik_yazdir("Gauss + Prewitt", turev_yumusak)

    kenar_ham = mutlak_goruntu(turev_ham)
    kenar_yumusak = mutlak_goruntu(turev_yumusak)
    kaydet(os.path.join(CIKTI_KLASORU, "gauss_yumusatilmis.png"), yumusak)
    kaydet(os.path.join(CIKTI_KLASORU, "prewitt_sonuc.png"), kenar_yumusak)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(yumusak, "Gauss 3x3"),
                     etiket_ekle(isaretli_goruntu(turev_yumusak), "Gauss+Prewitt (isaretli)")])
    alt = np.hstack([etiket_ekle(kenar_ham, "|Prewitt| (yumusatmasiz)"),
                     etiket_ekle(kenar_yumusak, "|Gauss+Prewitt|"),
                     etiket_ekle(np.where(kenar_yumusak > KENAR_ESIGI, 255, 0).astype(np.uint8),
                                 f"|Gauss+Prewitt| > {KENAR_ESIGI}")])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "prewitt_karsilastirma.png"), karsilastirma)

    pencere = "Gauss + 3x3 Prewitt filtresi (yatay kenar)"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
