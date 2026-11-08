"""
Görüntü İşleme Ödevi - Hough Dönüşümü ile Doğru Bulma (tamamen elle)
--------------------------------------------------------------------
Hazır cv2.HoughLines kullanmadan Hough doğru dönüşümünü baştan yazar.

Doğrunun normal (dik) gösterimi:

        rho = x * cos(theta) + y * sin(theta)

  theta : doğrunun normalinin x ekseniyle yaptığı açı, [0, 180) derece
  rho   : orijinden (sol üst köşe) doğruya olan işaretli uzaklık,
          [-D, +D], D = resmin köşegen uzunluğu

Görüntü uzayındaki bir doğru, (rho, theta) uzayında TEK bir noktadır.
Görüntü uzayındaki bir nokta (x, y) ise, kendisinden geçen bütün doğrulara
karşılık (rho, theta) uzayında bir SİNÜS eğrisidir. Aynı doğru üzerindeki
noktaların eğrileri, o doğrunun (rho, theta) noktasında kesişir.

Adımlar:

  1) Kenar bulma: 3x3 medyan (tuz-biber için) -> 3x3 Gauss -> Canny.

  2) sin / cos tablosu: 180 açının cos ve sin değerleri döngüden ÖNCE bir
     kez hesaplanıp listelerde tutulur. Oylama döngüsü her kenar pikseli
     için 180 açıyı dolaşır; tablo olmasa her seferinde yeniden cos / sin
     hesaplanırdı (kenar_sayısı x 180 kez).

  3) Oylama matrisi H[rho_indeksi, theta_indeksi], 0'larla dolu
     (2D+1) x 180 boyutlu bir tamsayı matrisi. Her kenar pikseli (x, y)
     için her theta'da:

            rho = x * COS[t] + y * SIN[t]
            H[round(rho) + D, t] += 1          <- oy

     Oylama, açık iki iç içe Python döngüsüyle yapılır.

  4) Tepe bulma: H'de eşik değeri (ESIK) geçen ve çevresindeki
     (2*KOMSULUK+1)^2 pencerede en büyük olan hücreler doğru kabul edilir.
     Bu yerel maksimum şartı, kalın / dalgalı kenarların aynı doğruya
     yakın birçok hücreye oy vermesinden doğan tekrarları önler.

  5) Oylama matrisi resim olarak kaydedilir:
       - ham hali (0..max -> 0..255)
       - log ölçekli renkli hali (küçük oylar da görünsün diye)
       - bulunan tepeler işaretlenmiş hali
     Ayrıca tek bir kenar pikselinin verdiği oyların (tek bir sinüs) ve
     aynı doğru üzerindeki birkaç pikselin eğrilerinin tek noktada
     kesiştiği bir gösterim de kaydedilir.

Doğrulama için sonuç cv2.HoughLines ile karşılaştırılır (yalnızca kontrol).

Varsayılan girdi 1. görevdeki yapay test resmidir (aynı tohumla yeniden
üretilir).

Kullanım:
    python HoughDogruElle.py [resim_yolu]

Herhangi bir tuşa basınca pencere kapanır.
"""

import math
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

DUSUK_ESIK = 60
YUKSEK_ESIK = 150

THETA_SAYISI = 180   # 1 derecelik adım, theta = 0, 1, ..., 179 derece
ESIK = 70            # bir doğru için gereken en az oy
KOMSULUK = 6         # yerel maksimum penceresi: +-6 piksel rho, +-6 derece theta

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
# Test resmi (1. görevdekiyle aynı)
# ---------------------------------------------------------------------------
def test_resmi_olustur():
    rng = np.random.default_rng(0)
    resim = np.full((300, 400), 70, np.uint8)
    cv2.rectangle(resim, (40, 40), (170, 140), 200, -1)
    cv2.fillPoly(resim, [np.array([[220, 260], [300, 60], [380, 260]])], 160)
    cv2.line(resim, (10, 280), (390, 200), 230, 3)
    cv2.line(resim, (60, 170), (60, 290), 20, 3)

    gurultulu = resim.astype(np.float64) + rng.normal(0, 10, resim.shape)
    resim = np.clip(gurultulu, 0, 255).astype(np.uint8)
    zar = rng.random(resim.shape)
    resim[zar < 0.02] = 0
    resim[zar > 0.98] = 255
    return resim


def kenar_bul(resim):
    temiz = cv2.medianBlur(resim, 3)
    yumusak = cv2.GaussianBlur(temiz, (3, 3), 0)
    return cv2.Canny(yumusak, DUSUK_ESIK, YUKSEK_ESIK, apertureSize=3, L2gradient=True)


# ---------------------------------------------------------------------------
# Hough dönüşümü
# ---------------------------------------------------------------------------
def sin_cos_tablosu():
    """theta = 0..179 derece için cos ve sin değerleri, bir kez hesaplanır."""
    thetalar = [math.radians(t) for t in range(THETA_SAYISI)]
    cos_tablosu = [math.cos(t) for t in thetalar]
    sin_tablosu = [math.sin(t) for t in thetalar]
    return thetalar, cos_tablosu, sin_tablosu


def hough_oylama(kenar, cos_tablosu, sin_tablosu):
    """Oylama matrisini doldurur. Satır: rho + D, sütun: theta (derece)."""
    yukseklik, genislik = kenar.shape
    D = int(math.ceil(math.hypot(yukseklik, genislik)))
    H = np.zeros((2 * D + 1, THETA_SAYISI), dtype=np.int32)

    # Kenar piksellerinin koordinatları (y, x)
    ys, xs = np.nonzero(kenar)
    for x, y in zip(xs.tolist(), ys.tolist()):
        for t in range(THETA_SAYISI):
            rho = x * cos_tablosu[t] + y * sin_tablosu[t]
            H[int(round(rho)) + D, t] += 1          # oy ver
    return H, D


def tepeleri_bul(H, esik=ESIK, komsuluk=KOMSULUK):
    """Eşiği geçen ve penceresinde en büyük olan hücreleri (oy, rho_i, theta_i) döner."""
    satir, sutun = H.shape
    tepeler = []
    adaylar = np.argwhere(H >= esik)
    for r, t in adaylar:
        oy = H[r, t]
        en_buyuk = True
        for dr in range(-komsuluk, komsuluk + 1):
            for dt in range(-komsuluk, komsuluk + 1):
                rr, tt = r + dr, t + dt
                # theta 180'de başa döner: (rho, 180+t) = (-rho, t)
                if tt < 0:
                    tt, rr = tt + sutun, (satir - 1) - rr
                elif tt >= sutun:
                    tt, rr = tt - sutun, (satir - 1) - rr
                if not (0 <= rr < satir) or (dr == 0 and dt == 0):
                    continue
                # eşit oylarda yalnızca ilkini tut (aynı tepeyi iki kez sayma)
                if H[rr, tt] > oy or (H[rr, tt] == oy and (rr, tt) < (r, t)):
                    en_buyuk = False
                    break
            if not en_buyuk:
                break
        if en_buyuk:
            tepeler.append((int(oy), int(r), int(t)))
    tepeler.sort(reverse=True)
    return tepeler


def dogru_ciz(resim, rho, theta, renk, kalinlik=1):
    uzunluk = 2 * int(math.hypot(*resim.shape[:2]))
    c, s = math.cos(theta), math.sin(theta)
    x0, y0 = rho * c, rho * s
    p1 = (int(round(x0 - uzunluk * s)), int(round(y0 + uzunluk * c)))
    p2 = (int(round(x0 + uzunluk * s)), int(round(y0 - uzunluk * c)))
    cv2.line(resim, p1, p2, renk, kalinlik, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Oylama matrisini resim olarak gösterme
# ---------------------------------------------------------------------------
def oylama_resimleri(H, D, tepeler):
    """Ham, log-renkli ve tepeleri işaretli oylama matrisi resimleri.
    Matris 180 sütunlu ve dar olduğu için yatayda 3 kat büyütülür."""
    ham = (H.astype(np.float64) / H.max() * 255).astype(np.uint8)
    log = np.log1p(H.astype(np.float64))
    log = (log / log.max() * 255).astype(np.uint8)
    renkli = cv2.applyColorMap(log, cv2.COLORMAP_INFERNO)

    olcek = 3
    ham = cv2.resize(ham, None, fx=olcek, fy=1, interpolation=cv2.INTER_NEAREST)
    renkli = cv2.resize(renkli, None, fx=olcek, fy=1, interpolation=cv2.INTER_NEAREST)
    isaretli = renkli.copy()
    for i, (oy, r, t) in enumerate(tepeler):
        merkez = (t * olcek + 1, r)
        cv2.circle(isaretli, merkez, 10, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(isaretli, str(i + 1), (merkez[0] + 11, merkez[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    eksen_ekle = lambda img: eksenler(img, D, olcek)
    return eksen_ekle(ham), eksen_ekle(renkli), eksen_ekle(isaretli)


def eksenler(resim, D, olcek):
    """Oylama matrisi resminin soluna rho, altına theta ölçeği ekler."""
    if resim.ndim == 2:
        resim = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    h, w = resim.shape[:2]
    tuval = np.full((h + 40, w + 75, 3), 255, np.uint8)
    tuval[:h, 60:60 + w] = resim
    for rho in range(-D // 100 * 100, D + 1, 100):
        y = rho + D
        cv2.line(tuval, (54, y), (59, y), (0, 0, 0), 1)
        cv2.putText(tuval, str(rho), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    for t in range(0, 181, 30):
        x = 60 + min(t * olcek, w - 1)
        cv2.line(tuval, (x, h), (x, h + 5), (0, 0, 0), 1)
        cv2.putText(tuval, str(t), (x - 8, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(tuval, "theta (derece)", (60 + w // 2 - 45, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(tuval, "rho", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return tuval


def tek_piksel_oylari(kenar_shape, cos_tablosu, sin_tablosu, noktalar):
    """Verilen her nokta için ayrı bir oylama matrisi doldurup hepsini toplar.
    Tek nokta -> tek sinüs; aynı doğru üzerindeki noktalar -> tek kesişim."""
    H, D = hough_oylama(np.zeros(kenar_shape, np.uint8), cos_tablosu, sin_tablosu)
    for x, y in noktalar:
        for t in range(THETA_SAYISI):
            rho = x * cos_tablosu[t] + y * sin_tablosu[t]
            H[int(round(rho)) + D, t] += 1
    return H, D


def etiket_ekle(resim, yazi):
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
    kenar = kenar_bul(resim)
    print("Boyut:", resim.shape, "| Kenar pikseli:", np.count_nonzero(kenar))

    thetalar, cos_tablosu, sin_tablosu = sin_cos_tablosu()

    baslangic = time.perf_counter()
    H, D = hough_oylama(kenar, cos_tablosu, sin_tablosu)
    print(f"Oylama süresi: {time.perf_counter() - baslangic:.2f} s | "
          f"matris boyutu: {H.shape} (rho: {-D}..{D}, theta: 0..179) | "
          f"toplam oy: {H.sum()} | en büyük oy: {H.max()}")

    baslangic = time.perf_counter()
    tepeler = tepeleri_bul(H)
    print(f"Tepe bulma süresi: {time.perf_counter() - baslangic:.2f} s | eşik: {ESIK} | "
          f"bulunan doğru: {len(tepeler)}")
    for i, (oy, r, t) in enumerate(tepeler):
        print(f"  {i + 1:2d}) rho = {r - D:5d}  theta = {t:3d} derece  oy = {oy}")

    # --- Kontrol: aynı ayarlarla cv2.HoughLines ---
    cv_dogrular = cv2.HoughLines(kenar, 1, np.pi / 180, ESIK)
    cv_dogrular = [] if cv_dogrular is None else cv_dogrular[:, 0]
    print(f"Kontrol: cv2.HoughLines (eşik {ESIK}, yerel maksimum yok) -> {len(cv_dogrular)} doğru")
    eslesen = 0
    for oy, r, t in tepeler:
        if any(abs((r - D) - rho) <= 2 and abs(t - np.degrees(th)) <= 2 for rho, th in cv_dogrular):
            eslesen += 1
    print(f"  Benim bulduğum {len(tepeler)} doğrudan {eslesen} tanesi cv2.HoughLines'ta da var.")

    # --- Sonuçları çiz ---
    sonuc = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for oy, r, t in tepeler:
        dogru_ciz(sonuc, r - D, thetalar[t], KIRMIZI)
    for i, (oy, r, t) in enumerate(tepeler):
        # doğru numarasını, doğrunun orijine en yakın noktasının resim içindeki izdüşümüne yaz
        rho, c, s = r - D, cos_tablosu[t], sin_tablosu[t]
        x, y = rho * c, rho * s
        x, y = min(max(int(x), 5), resim.shape[1] - 20), min(max(int(y), 15), resim.shape[0] - 5)
        cv2.putText(sonuc, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YESIL, 1, cv2.LINE_AA)
    cv_sonuc = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for rho, th in cv_dogrular:
        dogru_ciz(cv_sonuc, rho, th, KIRMIZI)

    ham, renkli, isaretli = oylama_resimleri(H, D, tepeler)
    kaydet(os.path.join(CIKTI_KLASORU, "kenar.png"), kenar)
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_matrisi_ham.png"), ham)
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_matrisi_log.png"), renkli)
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_matrisi_tepeler.png"), isaretli)
    kaydet(os.path.join(CIKTI_KLASORU, "bulunan_dogrular.png"), sonuc)

    # --- Oylamanın kendisi: tek nokta ve aynı doğru üzerindeki 5 nokta ---
    tek, _ = tek_piksel_oylari(kenar.shape, cos_tablosu, sin_tablosu, [(200, 100)])
    # x + y = 300 doğrusu üzerindeki 5 nokta: normali 45 derece, rho = 300 / sqrt(2) = 212.1
    bes = [(x, 300 - x) for x in (20, 80, 150, 220, 290)]
    bes_H, _ = tek_piksel_oylari(kenar.shape, cos_tablosu, sin_tablosu, bes)
    r, t = np.unravel_index(np.argmax(bes_H), bes_H.shape)
    print(f"5 nokta (x + y = 300) eğrileri rho = {r - D}, theta = {t} derecede kesişiyor, "
          f"oy = {bes_H[r, t]} (beklenen: rho = 212, theta = 45)")

    # Tek bir pikselin oylarından birkaç adım
    x, y = 200, 100
    print(f"Oylama örneği, piksel (x={x}, y={y}):")
    for t in (0, 30, 45, 60, 90, 120, 150):
        rho = x * cos_tablosu[t] + y * sin_tablosu[t]
        print(f"   theta = {t:3d}: rho = {x}*{cos_tablosu[t]:+.3f} + {y}*{sin_tablosu[t]:+.3f} "
              f"= {rho:7.1f}  ->  H[{int(round(rho)) + D}, {t}] += 1")
    gosterim = []
    for matris, yazi in ((tek, "(200,100) tek nokta: tek sinus"),
                         (bes_H, "x+y=300 uzerinde 5 nokta: tek kesisim")):
        gorsel = cv2.applyColorMap((matris * (255 // matris.max())).astype(np.uint8), cv2.COLORMAP_INFERNO)
        gorsel = cv2.resize(gorsel, None, fx=3, fy=1, interpolation=cv2.INTER_NEAREST)
        gorsel = cv2.dilate(gorsel, np.ones((3, 3), np.uint8))   # ince eğri görünsün
        gosterim.append(etiket_ekle(eksenler(gorsel, D, 3), yazi))
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_ornegi.png"), np.hstack(gosterim))

    ust = np.hstack([etiket_ekle(resim, "Orijinal"), etiket_ekle(kenar, "Kenar (medyan+Canny)")])
    alt = np.hstack([etiket_ekle(sonuc, f"Elle Hough: {len(tepeler)} dogru"),
                     etiket_ekle(cv_sonuc, f"cv2.HoughLines: {len(cv_dogrular)} dogru")])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "hough_elle_karsilastirma.png"), karsilastirma)

    cv2.namedWindow("Oylama matrisi", cv2.WINDOW_NORMAL)
    cv2.imshow("Oylama matrisi", isaretli)
    cv2.namedWindow("Elle Hough dogru", cv2.WINDOW_NORMAL)
    cv2.imshow("Elle Hough dogru", karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
