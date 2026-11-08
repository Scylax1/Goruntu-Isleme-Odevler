"""
Görüntü İşleme Ödevi - Hough Dönüşümü ile Çember Bulma (tamamen elle)
---------------------------------------------------------------------
Hazır cv2.HoughCircles kullanmadan, 3 boyutlu oylama matrisiyle çember
bulan Hough dönüşümünü baştan yazar.

Çemberin parametrik gösterimi (merkez (a, b), yarıçap r):

        x = a + r * cos(theta)          a = x - r * cos(theta)
        y = b + r * sin(theta)    ->    b = y - r * sin(theta)

Bir kenar pikseli (x, y), r yarıçaplı ve kendisinden geçen bütün çemberlerin
merkezlerine oy verir. Bu merkezler, (x, y) etrafında r yarıçaplı bir
çember çizer. Bütün r'ler üst üste konunca (a, b, r) uzayında bir KONİ
oluşur. Aynı çember üzerindeki kenar piksellerinin konileri, o çemberin
(a, b, r) noktasında kesişir.

Adımlar:

  1) Kenar bulma: 3x3 medyan (tuz-biber için) -> 3x3 Gauss -> Canny.

  2) sin / cos tablosu: theta = 0..359 derece için cos ve sin değerleri bir
     kez hesaplanıp dizilerde tutulur. Bunlardan her yarıçap için
     merkeze göre tamsayı ofsetler (dx, dy) = round(r*cos, r*sin) de bir
     kez hesaplanır. Küçük yarıçaplarda 360 açının birçoğu aynı piksele
     yuvarlandığı için tekrarlar atılır. Böylece bir kenar pikseli aynı
     (a, b, r) hücresine iki kez oy veremez.

  3) 3 boyutlu oylama matrisi H[b, a, r_indeksi], boyutu
     yükseklik x genişlik x yarıçap_sayısı. Oylama:

         her kenar pikseli (x, y) için:
             her yarıçap r için:
                 her (dx, dy) ofseti için:     (r yarıçaplı çemberin noktaları)
                     a = x - dx,  b = y - dy
                     H[b, a, r] += 1           <- oy

     İlk iki döngü açık Python döngüsüdür. En içteki döngü, aynı r için
     bütün ofsetleri tek seferde işleyen numpy dizi işlemiyle yapılır;
     tamamen Python döngüsüyle ~30 milyon işlem dakikalar sürerdi.
     Aynı hücre bir r katmanında yalnızca bir kez geçtiği için
     H[b, a, r] += 1 dizi hâlinde de doğru sayar.

  4) Eşik: r yarıçaplı tam bir çemberde en fazla len(ofset[r]) kenar
     pikseli olabilir (yaklaşık 2*pi*r). Bu yüzden oy, bu sayıya bölünerek
     0..1 arası "çemberin ne kadarı görülüyor" oranına çevrilir. Büyük
     çemberler küçüklere göre kayırılmaz. Oranı ESIK'i geçen hücreler aday
     olur; adaylar oranı büyükten küçüğe sıralanır ve daha önce kabul
     edilmiş bir çembere çok yakın olanlar (merkez ve yarıçap) atılır.

  5) Bulunan çemberler orijinal resmin üzerine çizilir. Oylama matrisinin
     bulunan yarıçaplardaki kesitleri, r ekseni boyunca en büyük değer
     izdüşümü ve tek bir kenar pikselinin oy konisi de resim olarak
     kaydedilir.

Varsayılan girdi 3. görevdeki yapay test resmidir (aynı tohumla yeniden
üretilir). Gerçek çemberler bilindiği için hatalar da yazdırılır.

Kullanım:
    python HoughCemberElle.py [resim_yolu]

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

ACI_SAYISI = 360         # 1 derecelik adım
EN_KUCUK_R = 15
EN_BUYUK_R = 65
ESIK = 0.45              # çemberin en az %45'i kenar pikseli olmalı
MERKEZ_YAKINLIK = 10     # bu kadar yakın merkez ve
YARICAP_YAKINLIK = 10    # bu kadar yakın yarıçap aynı çember sayılır

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
# Test resmi (3. görevdekiyle aynı)
# ---------------------------------------------------------------------------
def test_resmi_olustur():
    rng = np.random.default_rng(1)
    resim = np.full((300, 400), 90, np.uint8)
    cv2.rectangle(resim, (150, 130), (190, 170), 220, -1)
    for x, y, r, gri in GERCEK_CEMBERLER:
        cv2.circle(resim, (x, y), r, gri, -1, cv2.LINE_AA)

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
    """theta = 0..359 derece için cos ve sin, bir kez hesaplanır."""
    thetalar = np.radians(np.arange(ACI_SAYISI))
    return np.cos(thetalar), np.sin(thetalar)


def ofset_tablosu(yaricaplar, cos_tablosu, sin_tablosu):
    """Her yarıçap için çember noktalarının merkeze göre tamsayı ofsetleri.
    Aynı piksele düşen açılar tek ofset olarak tutulur."""
    tablo = []
    for r in yaricaplar:
        dx = np.rint(r * cos_tablosu).astype(np.int32)
        dy = np.rint(r * sin_tablosu).astype(np.int32)
        cift = np.unique(np.stack([dx, dy], axis=1), axis=0)
        tablo.append((cift[:, 0], cift[:, 1]))
    return tablo


def hough_cember_oylama(kenar, yaricaplar, ofsetler):
    """3 boyutlu oylama matrisi H[b, a, r_indeksi] doldurur."""
    yukseklik, genislik = kenar.shape
    H = np.zeros((yukseklik, genislik, len(yaricaplar)), dtype=np.int32)

    ys, xs = np.nonzero(kenar)
    for x, y in zip(xs.tolist(), ys.tolist()):                # her kenar pikseli
        for ri in range(len(yaricaplar)):                      # her yarıçap
            dx, dy = ofsetler[ri]
            a = x - dx                                         # her açı (dizi hâlinde)
            b = y - dy
            gecerli = (a >= 0) & (a < genislik) & (b >= 0) & (b < yukseklik)
            H[b[gecerli], a[gecerli], ri] += 1                 # oy ver
    return H


def cemberleri_sec(H, yaricaplar, ofsetler, esik=ESIK):
    """Oranı eşiği geçen hücrelerden, birbirine yakın olmayanları seçer."""
    nokta_sayisi = np.array([len(dx) for dx, _ in ofsetler], dtype=np.float64)
    oran = H / nokta_sayisi                 # her r katmanı kendi çevresine bölünür

    adaylar = np.argwhere(oran >= esik)
    sirali = sorted(((oran[b, a, ri], a, b, ri) for b, a, ri in adaylar), reverse=True)

    secilen = []
    for o, a, b, ri in sirali:
        r = yaricaplar[ri]
        yakin = any(math.hypot(a - sa, b - sb) < MERKEZ_YAKINLIK and abs(r - sr) < YARICAP_YAKINLIK
                    for _, sa, sb, sr, _ in secilen)
        if not yakin:
            secilen.append((float(o), int(a), int(b), int(r), int(H[b, a, ri])))
    return secilen, oran, len(adaylar)


# ---------------------------------------------------------------------------
# Gösterme
# ---------------------------------------------------------------------------
def renkli_harita(matris):
    norm = matris.astype(np.float64) / max(matris.max(), 1e-9)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


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
    kenar = kenar_bul(resim)
    print("Boyut:", resim.shape, "| Kenar pikseli:", np.count_nonzero(kenar))

    yaricaplar = list(range(EN_KUCUK_R, EN_BUYUK_R + 1))
    cos_tablosu, sin_tablosu = sin_cos_tablosu()
    ofsetler = ofset_tablosu(yaricaplar, cos_tablosu, sin_tablosu)
    print(f"Tablolar: {ACI_SAYISI} açı için cos/sin; {len(yaricaplar)} yarıçap için ofsetler "
          f"(r = {yaricaplar[0]}: {len(ofsetler[0][0])} nokta, "
          f"r = {yaricaplar[-1]}: {len(ofsetler[-1][0])} nokta)")

    # --- Oylamanın kendisi: tek bir pikselin birkaç oyu ---
    ys, xs = np.nonzero(kenar)
    x, y = int(xs[0]), int(ys[0])
    print(f"Oylama örneği, kenar pikseli (x={x}, y={y}), r = {yaricaplar[0]}:")
    for t in (0, 90, 180, 270):
        a = x - round(yaricaplar[0] * cos_tablosu[t])
        b = y - round(yaricaplar[0] * sin_tablosu[t])
        print(f"   theta = {t:3d}: a = {x} - {yaricaplar[0]}*{cos_tablosu[t]:+.2f} = {a:4d}, "
              f"b = {y} - {yaricaplar[0]}*{sin_tablosu[t]:+.2f} = {b:4d}  ->  H[{b}, {a}, r={yaricaplar[0]}] += 1")

    baslangic = time.perf_counter()
    H = hough_cember_oylama(kenar, yaricaplar, ofsetler)
    print(f"Oylama süresi: {time.perf_counter() - baslangic:.2f} s | matris boyutu (b, a, r): {H.shape} | "
          f"toplam oy: {H.sum()} | en büyük oy: {H.max()}")

    cemberler, oran, aday_sayisi = cemberleri_sec(H, yaricaplar, ofsetler)
    print(f"Eşik {ESIK}: {aday_sayisi} aday hücre -> yakınları elenince {len(cemberler)} çember")
    for o, a, b, r, oy in cemberler:
        satir = f"   merkez = ({a:3d}, {b:3d})  r = {r:2d}  oy = {oy:3d}  oran = {o:.2f}"
        if gercek_var:
            gx, gy, gr, _ = min(GERCEK_CEMBERLER, key=lambda c: math.hypot(c[0] - a, c[1] - b))
            satir += f"   | gerçek ({gx}, {gy}) r = {gr}"
        print(satir)

    # Eşiğin etkisi
    for e in (0.25, 0.35, ESIK, 0.6, 0.75):
        print(f"   eşik {e:.2f} -> {len(cemberleri_sec(H, yaricaplar, ofsetler, e)[0])} çember")

    # --- Sonuçları çiz ---
    sonuc = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for o, a, b, r, oy in cemberler:
        cv2.circle(sonuc, (a, b), r, YESIL, 2, cv2.LINE_AA)
        cv2.circle(sonuc, (a, b), 2, KIRMIZI, -1)
        cv2.putText(sonuc, f"{o:.2f}", (a - 14, b - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, KIRMIZI, 1, cv2.LINE_AA)

    # Oylama matrisinin kesitleri: bulunan her yarıçapta H[:, :, r]
    kesitler = []
    for o, a, b, r, oy in sorted(cemberler, key=lambda c: c[3])[:3]:
        kesit = renkli_harita(H[:, :, r - EN_KUCUK_R])
        cv2.circle(kesit, (a, b), 8, YESIL, 1, cv2.LINE_AA)
        kesitler.append(etiket_ekle(kesit, f"H[:, :, r={r}] kesiti"))
    while len(kesitler) < 3:
        kesitler.append(np.zeros_like(sonuc))

    izdusum = renkli_harita(oran.max(axis=2))
    # Tek bir kenar pikselinin oy konisi: r büyüdükçe genişleyen çemberler
    koni = np.zeros(kenar.shape, np.float64)
    tek = np.zeros_like(kenar)
    tek[150, 200] = 255
    tek_H = hough_cember_oylama(tek, yaricaplar, ofsetler)
    for ri in range(0, len(yaricaplar), 10):
        koni = np.maximum(koni, tek_H[:, :, ri] * (ri + 10))
    koni = cv2.dilate(renkli_harita(koni), np.ones((2, 2), np.uint8))

    # Bulunan her merkezde H'nin r ekseni boyunca kesiti: oran(a, b, r) - r grafiği
    grafik = np.zeros_like(sonuc)
    gh, gw = grafik.shape[:2]
    sol, alt_kenar, ust_kenar = 30, gh - 25, 35
    esik_y = int(alt_kenar - ESIK * (alt_kenar - ust_kenar))
    cv2.line(grafik, (sol, esik_y), (gw - 5, esik_y), (90, 90, 90), 1)
    cv2.putText(grafik, f"esik {ESIK}", (gw - 75, esik_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    for r in range(EN_KUCUK_R, EN_BUYUK_R + 1, 10):
        x = sol + (r - EN_KUCUK_R) * (gw - sol - 10) // (EN_BUYUK_R - EN_KUCUK_R)
        cv2.putText(grafik, str(r), (x - 8, gh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for i, (o, a, b, r, oy) in enumerate(cemberler):
        renk = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[40 + 40 * i]]), cv2.COLORMAP_TURBO)[0, 0])
        noktalar = [(sol + ri * (gw - sol - 10) // (len(yaricaplar) - 1),
                     int(alt_kenar - oran[b, a, ri] * (alt_kenar - ust_kenar)))
                    for ri in range(len(yaricaplar))]
        cv2.polylines(grafik, [np.array(noktalar, np.int32)], False, renk, 1, cv2.LINE_AA)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"), etiket_ekle(kenar, "Kenar (medyan+Canny)"),
                     etiket_ekle(sonuc, f"Bulunan: {len(cemberler)} cember (esik {ESIK})")])
    orta = np.hstack(kesitler)
    alt = np.hstack([etiket_ekle(izdusum, "max_r oran(a, b, r)"),
                     etiket_ekle(koni, "Tek pikselin oylari: r=15,25..65"),
                     etiket_ekle(grafik, "Bulunan merkezlerde oran - r")])
    karsilastirma = np.vstack([ust, orta, alt])

    kaydet(os.path.join(CIKTI_KLASORU, "kenar.png"), kenar)
    kaydet(os.path.join(CIKTI_KLASORU, "bulunan_cemberler.png"), sonuc)
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_izdusum.png"), izdusum)
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_koni_tek_piksel.png"), koni)
    kaydet(os.path.join(CIKTI_KLASORU, "hough_cember_elle.png"), karsilastirma)

    pencere = "Elle Hough cember"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
