"""
Görüntü İşleme Ödevi - Canny + Medyan + OpenCV Hough Doğru Bulma
----------------------------------------------------------------
Bir görüntüde doğruları OpenCV'nin iki farklı Hough fonksiyonuyla bulur:

  1) cv2.HoughLines  (standart Hough dönüşümü)
     Her kenar pikseli, (rho, theta) uzayında bir sinüs eğrisine oy verir.
     Sonuç sonsuz uzunlukta doğrulardır: rho = x*cos(theta) + y*sin(theta).
     Çizilirken resmin bir ucundan öbür ucuna uzatılır. KIRMIZI ile çizilir.

  2) cv2.HoughLinesP (olasılıksal Hough dönüşümü)
     Kenar piksellerinin rastgele bir alt kümesiyle oylama yapar.
     Bulduğu her doğru boyunca pikselleri izleyip uç noktaları
     (x1, y1, x2, y2) olan doğru PARÇALARI döndürür. minLineLength'ten kısa
     parçaları atar, maxLineGap'ten küçük boşlukları birleştirir.
     YEŞİL ile çizilir.

İşlem sırası (ödevde istendiği gibi):

     Gri resim -> 3x3 Gauss -> Canny -> 3x3 medyan (kenar resmine) -> Hough

ÖNEMLİ GÖZLEM: Canny kenarları 1 piksel inceliğindedir. İkili (0/255) bir
resimde 3x3 medyan, 9 pikselin çoğunluk oyudur: bir piksel ancak 9
komşusundan en az 5'i kenarsa kenar kalır. İnce bir doğru üzerindeki
pikselin penceresinde yalnızca 3 kenar pikseli vardır (3/9), bu yüzden
medyan doğruları SİLER. Gürültüden doğan küçük kapalı halkalar ve kümeler
ise yoğun oldukları için kalır. Yani medyanın kenar resmine uygulanması
istenen etkinin tersini yapar.

Bunu göstermek için aynı program bir de sırası değiştirilmiş yolu çalıştırır:

     Gri resim -> 3x3 medyan -> 3x3 Gauss -> Canny -> Hough

Burada medyan, tuz-biber gürültüsünü kenar bulunmadan önce temizler;
doğrular sağlam kalır.

Varsayılan girdi, program içinde çizilen yapay bir test resmidir
(dikdörtgen, üçgen, iki kalın çizgi + Gauss ve tuz-biber gürültüsü).
Doğruların nerede olduğu bilindiği için sonuç gözle kolayca denetlenir.

Kullanım:
    python HoughDogruOpenCV.py [resim_yolu]

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

# Canny eşikleri (önceki haftadaki gibi 1:2 - 1:3 oranı)
DUSUK_ESIK = 60
YUKSEK_ESIK = 150

MEDYAN_BOYUTU = 3

# cv2.HoughLines: rho çözünürlüğü 1 piksel, theta çözünürlüğü 1 derece.
# Bir doğrunun kabul edilmesi için gereken en az oy (kenar pikseli) sayısı.
HOUGH_ESIK = 70

# cv2.HoughLinesP
HOUGHP_ESIK = 60
EN_KISA_PARCA = 60   # bundan kısa parçalar atılır (piksel)
EN_BUYUK_BOSLUK = 10  # aynı doğru üzerindeki bu kadar boşluk birleştirilir

KIRMIZI = (0, 0, 255)
YESIL = (0, 255, 0)


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
    """Doğruları bilinen 400x300 yapay resim: dikdörtgen, üçgen, iki çizgi + gürültü."""
    rng = np.random.default_rng(0)
    resim = np.full((300, 400), 70, np.uint8)
    cv2.rectangle(resim, (40, 40), (170, 140), 200, -1)
    cv2.fillPoly(resim, [np.array([[220, 260], [300, 60], [380, 260]])], 160)
    cv2.line(resim, (10, 280), (390, 200), 230, 3)   # eğik açık çizgi
    cv2.line(resim, (60, 170), (60, 290), 20, 3)     # dikey koyu çizgi

    gurultulu = resim.astype(np.float64) + rng.normal(0, 10, resim.shape)
    resim = np.clip(gurultulu, 0, 255).astype(np.uint8)
    # %2 tuz, %2 biber
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


def hough_dogrulari_ciz(resim, kenar):
    """İki Hough fonksiyonunu çalıştırır, sonuçları ayrı ve ortak resimlere çizer."""
    dogrular = cv2.HoughLines(kenar, 1, np.pi / 180, HOUGH_ESIK)
    parcalar = cv2.HoughLinesP(kenar, 1, np.pi / 180, HOUGHP_ESIK,
                               minLineLength=EN_KISA_PARCA, maxLineGap=EN_BUYUK_BOSLUK)
    dogrular = [] if dogrular is None else dogrular[:, 0]
    parcalar = [] if parcalar is None else parcalar[:, 0]

    renkli = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    standart, olasiliksal, ortak = renkli.copy(), renkli.copy(), renkli.copy()

    uzunluk = 2 * int(np.hypot(*resim.shape))  # doğruyu resmin dışına taşacak kadar uzat
    for rho, theta in dogrular:
        # (x0, y0): doğrunun orijine en yakın noktası; (-sin, cos): doğrunun yönü
        c, s = np.cos(theta), np.sin(theta)
        x0, y0 = rho * c, rho * s
        p1 = (int(round(x0 - uzunluk * s)), int(round(y0 + uzunluk * c)))
        p2 = (int(round(x0 + uzunluk * s)), int(round(y0 - uzunluk * c)))
        cv2.line(standart, p1, p2, KIRMIZI, 1, cv2.LINE_AA)
        cv2.line(ortak, p1, p2, KIRMIZI, 1, cv2.LINE_AA)

    for x1, y1, x2, y2 in parcalar:
        cv2.line(olasiliksal, (x1, y1), (x2, y2), YESIL, 2, cv2.LINE_AA)
        cv2.line(ortak, (x1, y1), (x2, y2), YESIL, 2, cv2.LINE_AA)

    return dogrular, parcalar, standart, olasiliksal, ortak


def rapor(baslik, kenar, dogrular, parcalar):
    print(f"\n{baslik}")
    print(f"  Kenar pikseli: {np.count_nonzero(kenar)}")
    print(f"  cv2.HoughLines  -> {len(dogrular)} doğru")
    for rho, theta in dogrular:
        print(f"      rho = {rho:7.1f}  theta = {np.degrees(theta):6.1f} derece")
    print(f"  cv2.HoughLinesP -> {len(parcalar)} doğru parçası")
    for x1, y1, x2, y2 in parcalar:
        print(f"      ({x1:3d},{y1:3d}) - ({x2:3d},{y2:3d})  uzunluk {np.hypot(x2 - x1, y2 - y1):5.1f}")


# ---------------------------------------------------------------------------
# Gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return kopya


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        resim = gri_oku(sys.argv[1])
        if resim is None:
            sys.exit(f"Resim açılamadı: {sys.argv[1]}")
    else:
        resim = test_resmi_olustur()
        kaydet(os.path.join(CIKTI_KLASORU, "test_resmi.png"), resim)
    print("Boyut:", resim.shape)

    # --- A) Ödevdeki sıra: Canny -> medyan (kenar resmine) -> Hough ---
    kenar = canny(resim)
    kenar_medyan = cv2.medianBlur(kenar, MEDYAN_BOYUTU)
    a_dogru, a_parca, a_std, a_olas, a_ortak = hough_dogrulari_ciz(resim, kenar_medyan)
    rapor("A) Canny -> medyan -> Hough", kenar_medyan, a_dogru, a_parca)
    print(f"  Medyan, {np.count_nonzero(kenar)} kenar pikselinin "
          f"{np.count_nonzero((kenar > 0) & (kenar_medyan == 0))} tanesini sildi.")

    # --- B) Sırası değiştirilmiş: medyan -> Canny -> Hough ---
    kenar_b = canny(cv2.medianBlur(resim, MEDYAN_BOYUTU))
    b_dogru, b_parca, b_std, b_olas, b_ortak = hough_dogrulari_ciz(resim, kenar_b)
    rapor("B) medyan -> Canny -> Hough", kenar_b, b_dogru, b_parca)

    kaydet(os.path.join(CIKTI_KLASORU, "A_canny.png"), kenar)
    kaydet(os.path.join(CIKTI_KLASORU, "A_canny_medyan.png"), kenar_medyan)
    kaydet(os.path.join(CIKTI_KLASORU, "A_hough_sonuc.png"), a_ortak)
    kaydet(os.path.join(CIKTI_KLASORU, "B_medyan_canny.png"), kenar_b)
    kaydet(os.path.join(CIKTI_KLASORU, "B_hough_sonuc.png"), b_ortak)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(kenar, "A: Canny"),
                     etiket_ekle(kenar_medyan, "A: Canny -> medyan 3x3"),
                     etiket_ekle(a_ortak, f"A: HoughLines {len(a_dogru)} / P {len(a_parca)}")])
    alt = np.hstack([etiket_ekle(kenar_b, "B: medyan -> Canny"),
                     etiket_ekle(b_std, f"B: HoughLines ({len(b_dogru)}, kirmizi)"),
                     etiket_ekle(b_olas, f"B: HoughLinesP ({len(b_parca)}, yesil)"),
                     etiket_ekle(b_ortak, "B: ikisi birlikte")])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "hough_dogru_karsilastirma.png"), karsilastirma)

    pencere = "Canny + Medyan + OpenCV Hough Dogru"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
