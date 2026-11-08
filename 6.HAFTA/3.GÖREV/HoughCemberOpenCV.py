"""
Görüntü İşleme Ödevi - Canny + Medyan + OpenCV Hough Çember Bulma
-----------------------------------------------------------------
Bir görüntüde çemberleri OpenCV'nin cv2.HoughCircles fonksiyonuyla bulur.

cv2.HoughCircles (HOUGH_GRADIENT yöntemi) üç boyutlu (a, b, r) oylama
matrisini doğrudan kurmaz; "Hough gradyan yöntemi" denen iki aşamalı bir
yol izler:

  1) Girdi resmine kendi içinde Canny uygular (yüksek eşik = param1,
     düşük eşik = param1 / 2) ve Sobel ile her kenar pikselinin gradyan
     yönünü bulur.
  2) Her kenar pikseli, gradyan doğrultusu üzerindeki (minRadius ..
     maxRadius uzaklığındaki) noktalara MERKEZ oyu verir. Çemberin bütün
     kenar piksellerinin gradyanı merkezden geçtiği için merkezde oylar
     birikir (2 boyutlu oylama).
  3) Oyu param2'yi geçen ve birbirinden en az minDist uzak olan merkezler
     için, kenar piksellerinin merkeze uzaklıklarının histogramından en
     çok desteklenen yarıçap seçilir.

Ödevde istenen sıra:

     Gri resim -> 3x3 Gauss -> Canny -> 3x3 medyan (kenar resmine)
               -> cv2.HoughCircles

Kenar resmi ikili (0/255) olduğu için HoughCircles'in içindeki Canny, kenar
pikselleri dışında hiçbir şey bulamaz; yani asıl kenarları bizim Canny'miz
belirler.

1. görevdeki gözlem burada da geçerli: 1 piksellik Canny kenarı üzerinde
3x3 medyan kenarları siler (pencerede 3/9 kenar pikseli var, çoğunluk değil).
Bu yüzden sırası değiştirilmiş yol da çalıştırılır:

     Gri resim -> 3x3 medyan -> 3x3 Gauss -> Canny -> cv2.HoughCircles

Bulunan çemberler YEŞİL, merkezleri KIRMIZI ile çizilir. Yapay test resmi
kullanıldığı için gerçek çemberler bilinir ve sonuç onlarla karşılaştırılır.

Kullanım:
    python HoughCemberOpenCV.py [resim_yolu]

Herhangi bir tuşa basınca pencere kapanır.
"""

import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

DUSUK_ESIK = 60
YUKSEK_ESIK = 150
MEDYAN_BOYUTU = 3

# cv2.HoughCircles parametreleri
DP = 1              # oylama matrisi çözünürlüğü = resim çözünürlüğü
EN_AZ_UZAKLIK = 20  # iki merkez arası en az uzaklık (minDist)
PARAM1 = 100        # içteki Canny'nin yüksek eşiği (ikili resimde fark etmez)
PARAM2 = 24         # merkez oy eşiği: küçüldükçe daha çok (ve sahte) çember
EN_KUCUK_R = 10
EN_BUYUK_R = 80

# Test resmindeki gerçek çemberler: (x, y, r, gri seviye)
GERCEK_CEMBERLER = [(90, 90, 40, 200), (220, 80, 30, 170), (320, 110, 50, 230),
                    (100, 220, 35, 30), (240, 215, 55, 150), (355, 255, 22, 210)]

YESIL = (0, 255, 0)
KIRMIZI = (0, 0, 255)


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
# Test resmi
# ---------------------------------------------------------------------------
def test_resmi_olustur():
    """Altı dolu daire + çember olmayan bir kare, Gauss ve tuz-biber gürültüsü."""
    rng = np.random.default_rng(1)
    resim = np.full((300, 400), 90, np.uint8)
    cv2.rectangle(resim, (150, 130), (190, 170), 220, -1)   # çember değil: tuzak
    for x, y, r, gri in GERCEK_CEMBERLER:
        cv2.circle(resim, (x, y), r, gri, -1, cv2.LINE_AA)

    gurultulu = resim.astype(np.float64) + rng.normal(0, 10, resim.shape)
    resim = np.clip(gurultulu, 0, 255).astype(np.uint8)
    zar = rng.random(resim.shape)
    resim[zar < 0.02] = 0
    resim[zar > 0.98] = 255
    return resim


# ---------------------------------------------------------------------------
# Kenar ve Hough
# ---------------------------------------------------------------------------
def canny(resim):
    yumusak = cv2.GaussianBlur(resim, (3, 3), 0)
    return cv2.Canny(yumusak, DUSUK_ESIK, YUKSEK_ESIK, apertureSize=3, L2gradient=True)


def cemberleri_bul_ciz(resim, kenar):
    cemberler = cv2.HoughCircles(kenar, cv2.HOUGH_GRADIENT, DP, EN_AZ_UZAKLIK,
                                 param1=PARAM1, param2=PARAM2,
                                 minRadius=EN_KUCUK_R, maxRadius=EN_BUYUK_R)
    cemberler = [] if cemberler is None else cemberler[0]
    cizim = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for x, y, r in cemberler:
        merkez = (int(round(x)), int(round(y)))
        cv2.circle(cizim, merkez, int(round(r)), YESIL, 2, cv2.LINE_AA)
        cv2.circle(cizim, merkez, 2, KIRMIZI, -1)
    return cemberler, cizim


def rapor(baslik, kenar, cemberler, gercek_var):
    print(f"\n{baslik}")
    print(f"  Kenar pikseli: {np.count_nonzero(kenar)}")
    print(f"  cv2.HoughCircles -> {len(cemberler)} çember")
    for x, y, r in cemberler:
        satir = f"      merkez = ({x:6.1f}, {y:6.1f})  r = {r:5.1f}"
        if gercek_var:
            # en yakın gerçek çember ve hata
            gx, gy, gr, _ = min(GERCEK_CEMBERLER, key=lambda c: np.hypot(c[0] - x, c[1] - y))
            satir += (f"   | gerçek ({gx}, {gy}) r = {gr}  ->  merkez hatası "
                      f"{np.hypot(gx - x, gy - y):4.1f}, r hatası {r - gr:+5.1f}")
        print(satir)


# ---------------------------------------------------------------------------
# Gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return kopya


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    gercek_var = len(sys.argv) == 1
    if not gercek_var:
        resim = gri_oku(sys.argv[1])
        if resim is None:
            sys.exit(f"Resim açılamadı: {sys.argv[1]}")
    else:
        resim = test_resmi_olustur()
        kaydet(os.path.join(CIKTI_KLASORU, "test_resmi.png"), resim)
    print("Boyut:", resim.shape)

    # --- A) Ödevdeki sıra: Canny -> medyan (kenar resmine) -> HoughCircles ---
    kenar = canny(resim)
    kenar_medyan = cv2.medianBlur(kenar, MEDYAN_BOYUTU)
    a_cember, a_cizim = cemberleri_bul_ciz(resim, kenar_medyan)
    rapor("A) Canny -> medyan -> HoughCircles", kenar_medyan, a_cember, gercek_var)
    print(f"  Medyan, {np.count_nonzero(kenar)} kenar pikselinin "
          f"{np.count_nonzero((kenar > 0) & (kenar_medyan == 0))} tanesini sildi.")

    # --- B) Sırası değiştirilmiş: medyan -> Canny -> HoughCircles ---
    kenar_b = canny(cv2.medianBlur(resim, MEDYAN_BOYUTU))
    b_cember, b_cizim = cemberleri_bul_ciz(resim, kenar_b)
    rapor("B) medyan -> Canny -> HoughCircles", kenar_b, b_cember, gercek_var)

    kaydet(os.path.join(CIKTI_KLASORU, "A_canny.png"), kenar)
    kaydet(os.path.join(CIKTI_KLASORU, "A_canny_medyan.png"), kenar_medyan)
    kaydet(os.path.join(CIKTI_KLASORU, "A_hough_cember.png"), a_cizim)
    kaydet(os.path.join(CIKTI_KLASORU, "B_medyan_canny.png"), kenar_b)
    kaydet(os.path.join(CIKTI_KLASORU, "B_hough_cember.png"), b_cizim)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(kenar, "A: Canny"),
                     etiket_ekle(kenar_medyan, "A: Canny -> medyan 3x3"),
                     etiket_ekle(a_cizim, f"A: HoughCircles ({len(a_cember)})")])
    alt = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(cv2.medianBlur(resim, MEDYAN_BOYUTU), "B: medyan 3x3"),
                     etiket_ekle(kenar_b, "B: medyan -> Canny"),
                     etiket_ekle(b_cizim, f"B: HoughCircles ({len(b_cember)})")])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "hough_cember_karsilastirma.png"), karsilastirma)

    pencere = "Canny + Medyan + OpenCV HoughCircles"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
