"""
Görüntü İşleme Ödevi - OpenCV Canny Kenar Bulucu
------------------------------------------------
Tek kanallı (gri) bir görüntüde OpenCV'nin hazır Canny fonksiyonuyla
kenarları bulur.

Canny algoritmasının adımları (cv2.Canny 2-5. adımları kendi içinde yapar):

  1) Gauss ile yumuşatma. cv2.Canny bunu YAPMAZ, bu yüzden önce
     cv2.GaussianBlur (3x3) uygulanır; önceki görevlerle aynı ön yumuşatma.
  2) Sobel ile gradyan: Gx, Gy, büyüklük |G| ve yön atan2(Gy, Gx).
     L2gradient=True ile |G| = sqrt(Gx^2 + Gy^2) kullanılır
     (varsayılan |Gx| + |Gy| yaklaşımıdır).
  3) Maksimum olmayanı bastırma (non-maximum suppression): her piksel,
     gradyan yönündeki iki komşusundan büyük değilse silinir. Sobel'in
     kalın kenar bandı tek piksel kalınlığında bir çizgiye iner.
  4) Çift eşik: |G| > YUKSEK_ESIK güçlü kenar, DUSUK_ESIK < |G| < YUKSEK_ESIK
     zayıf kenar, gerisi atılır.
  5) Histerezis: zayıf kenarlar yalnızca bir güçlü kenara bağlıysa kalır.
     Böylece gerçek kenarlar kopmadan izlenir, tek başına duran gürültü
     noktaları atılır.

Karşılaştırma için histerezis olmadan tek eşikle (DUSUK = YUKSEK) çalıştırılan
Canny ve farklı eşik çiftleri de gösterilir.

Varsayılan girdi 2. haftadaki (1. görev) cameraman resmidir.

Kullanım:
    python CannyKenar.py [resim_yolu]

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

# Canny'nin önerdiği gibi yüksek/düşük oranı 2:1 - 3:1 arası.
# |G| burada ham Sobel ölçeğinde (normalize değil): 150 gri seviyelik keskin
# bir kenar |G| ~ 600 verir.
DUSUK_ESIK = 50
YUKSEK_ESIK = 150


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
# Canny
# ---------------------------------------------------------------------------
def canny(resim, dusuk=DUSUK_ESIK, yuksek=YUKSEK_ESIK):
    """3x3 Gauss + cv2.Canny (3x3 Sobel, L2 gradyan). 0/255 kenar haritası döner."""
    yumusak = cv2.GaussianBlur(resim, (3, 3), 0)
    return cv2.Canny(yumusak, dusuk, yuksek, apertureSize=3, L2gradient=True)


# ---------------------------------------------------------------------------
# Gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM

    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape, "| Kanal sayısı: 1")

    # Çalışma süresi: tek ölçüm gürültülü olduğu için 50 kez çalıştırıp ortanca alınır
    sureler = []
    for _ in range(50):
        baslangic = time.perf_counter()
        kenar = canny(resim)
        sureler.append(time.perf_counter() - baslangic)
    print(f"Gauss + Canny süresi (50 ölçümün ortancası): {np.median(sureler) * 1000:.2f} ms")
    print(f"Canny ({DUSUK_ESIK}, {YUKSEK_ESIK}) kenar pikseli: {np.count_nonzero(kenar)}")

    # Histerezisin etkisi: tek eşikle (düşük = yüksek) çalıştırınca ne oluyor?
    sadece_yuksek = canny(resim, YUKSEK_ESIK, YUKSEK_ESIK)
    sadece_dusuk = canny(resim, DUSUK_ESIK, DUSUK_ESIK)
    print(f"Tek eşik {YUKSEK_ESIK}: {np.count_nonzero(sadece_yuksek)} piksel | "
          f"tek eşik {DUSUK_ESIK}: {np.count_nonzero(sadece_dusuk)} piksel")
    # Bağlı bileşen sayısı: kopuk kenar parçalarının ölçüsü
    for ad, harita in (("histerezis", kenar), (f"tek eşik {YUKSEK_ESIK}", sadece_yuksek),
                       (f"tek eşik {DUSUK_ESIK}", sadece_dusuk)):
        parca = cv2.connectedComponents(harita, connectivity=8)[0] - 1
        print(f"   {ad:<14} -> kenar parçası sayısı: {parca:5d} | "
              f"parça başına ortalama piksel: {np.count_nonzero(harita) / parca:5.1f}")

    kaydet(os.path.join(CIKTI_KLASORU, "canny_sonuc.png"), kenar)

    # Histerezis ile eklenen kenarlar: tek eşik 150'de olmayıp çift eşikte olanlar (kırmızı)
    renkli = cv2.cvtColor(sadece_yuksek, cv2.COLOR_GRAY2BGR)
    renkli[(kenar > 0) & (sadece_yuksek == 0)] = (0, 0, 255)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(kenar, f"Canny ({DUSUK_ESIK}, {YUKSEK_ESIK})"),
                     etiket_ekle(np.where(kenar[..., None] > 0, [0, 0, 255],
                                          cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)).astype(np.uint8),
                                 "Kenarlar resim uzerinde")])
    alt = np.hstack([etiket_ekle(sadece_dusuk, f"Tek esik {DUSUK_ESIK} (histerezissiz)"),
                     etiket_ekle(sadece_yuksek, f"Tek esik {YUKSEK_ESIK} (histerezissiz)"),
                     etiket_ekle(renkli, "Histerezisle eklenen (kirmizi)")])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "canny_karsilastirma.png"), karsilastirma)

    pencere = "Gauss + OpenCV Canny"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
