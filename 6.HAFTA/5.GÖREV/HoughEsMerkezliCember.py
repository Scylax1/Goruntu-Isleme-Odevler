"""
Görüntü İşleme Ödevi - Hough Dönüşümü ile Eş Merkezli Çember Bulma (elle)
-------------------------------------------------------------------------
Aynı merkeze sahip çember gruplarını (hedef tahtası, halkalar) bulan Hough
dönüşümünü baştan yazar.

Neden ayrı bir oylama?
  4. görevdeki gibi her kenar pikseli, r yarıçaplı çemberin BÜTÜN açılarına
  (360 yöne) oy verirse, her halka kendi konisini oluşturur. Halkalar
  birbirinin merkezine katkı vermez. Eş merkezliliği yakalamak için çemberleri
  tek tek bulup sonra merkezleri yakın olanları gruplamak gerekirdi. Bu bir
  SON İŞLEMDİR (post-processing) ve burada istenen o değil.

  Burada oylama, eş merkezliliğin geometrik tanımı üzerine kurulur:

      Bir çemberin her noktasındaki kenar normali (gradyan) merkezden geçer.
      Eş merkezli çemberlerin HEPSİNİN normalleri AYNI noktadan geçer.

  Bu yüzden her kenar pikseli, 360 yöne değil yalnızca kendi gradyan
  doğrultusu üzerindeki merkezlere oy verir (radyal oy). Hangi halkadan
  gelirse gelsin bütün radyal oylar ortak merkezin (a, b) kolonunda
  toplanır. Her halka da bu kolonda kendi yarıçapına (r katmanına) düşer.
  Eş merkezli bir grup, oylama matrisinde "aynı (a, b) kolonunda birden
  fazla r tepesi" olarak doğrudan ortaya çıkar. Tek bir çemberin kolonunda
  ise tek tepe olur. Merkezleri farklı olan iç içe (eksantrik) çemberlerin
  oyları iki ayrı kolona dağılır.

Adımlar:

  1) Kenar ve gradyan: 3x3 medyan -> 3x3 Gauss -> Canny (kenarlar).
     Gradyan YÖNÜ ise daha çok yumuşatılmış (Gauss, sigma = 2.5) resimden
     Sobel Gx, Gy ile bulunur: phi = atan2(Gy, Gx), 1 dereceye yuvarlanır
     (0..359). Radyal oyda yön hatası uzaklıkla büyür: 3 derecelik hata
     r = 75'te oyu 4 piksel kaydırır. 3x3 Gauss'tan alınan gürültülü yönle
     en dış halkanın oranı 0.32'de kalıyordu, sigma = 2.5 ile 0.65'e çıktı.

  2) sin / cos tablosu: 0..359 derecenin cos ve sin değerleri bir kez
     hesaplanıp dizilerde tutulur. Oylamada her piksel için açı
     indeksiyle tablodan okunur, cos / sin bir daha hesaplanmaz.

  3) 3 boyutlu oylama matrisi H[b, a, r_indeksi]. Oylama:

        her kenar pikseli (x, y), açı indeksi k için:
            her yarıçap r için:
                her yön s = +1, -1 için:     (merkez içeride ya da dışarıda
                                              olabilir: açık halka koyu
                                              zeminde ya da tersi)
                    a = x - s * r * COS[k]
                    b = y - s * r * SIN[k]
                    H[b, a, r] += 1          <- oy

     Üç döngü de açık Python döngüsüdür; her piksel r başına yalnızca
     2 oy verdiği için hızlıdır.

  4) Eşik ve okuma:
       - Gradyan açısının 1 dereceye yuvarlanması uzaktaki oyları 1-2
         piksel kaydırır. Bu yüzden her r katmanında 3x3 komşuluk toplanır.
       - Oran = oy / (2*pi*r): halkanın ne kadarının oy verdiği. 3x3
         toplamda komşu katmanlardan taşan oylar da sayıldığı için tam
         ve keskin bir halkada 1'i biraz geçebilir.
       - Bir (a, b) kolonunda oranı HALKA_ESIGI'ni geçen ve r ekseninde
         yerel maksimum olan her r bir halkadır.
       - Kolonunda en az EN_AZ_HALKA halka bulunan merkez, eş merkezli
         grup kabul edilir. Yakın merkezlerden en güçlüsü tutulur.

  5) Bulunan eş merkezli gruplar, her grup kendi renginde olmak üzere
     orijinal resmin üzerine çizilir. Kolonunda tek halka olan merkezler
     (tek çemberler) gri ile gösterilir, eş merkezli sayılmaz.

Karşılaştırma için cv2.HoughCircles da çalıştırılır. Merkez başına tek
yarıçap seçtiği ve minDist ile yakın merkezleri bastırdığı için eş
merkezli halkaların çoğunu kaçırır.

Varsayılan girdi, program içinde çizilen yapay bir test resmidir:
  - 4 halkalı eş merkezli grup
  - 3 halkalı eş merkezli grup, dış halkası bir dikdörtgenle kısmen örtülü
  - tek çemberler (tuzak)
  - merkezleri 14 piksel kaymış iç içe iki çember (eksantrik, tuzak)
  - bir kare (tuzak)
  - Gauss ve tuz-biber gürültüsü

Kullanım:
    python HoughEsMerkezliCember.py [resim_yolu]

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
GRADYAN_SIGMA = 2.5     # gradyan yönü için Gauss yumuşatması

ACI_SAYISI = 360
EN_KUCUK_R = 10
EN_BUYUK_R = 80
HALKA_ESIGI = 0.40      # halkanın en az %40'ı oy vermeli
EN_AZ_HALKA = 2         # eş merkezli sayılmak için gereken halka sayısı
R_KOMSULUK = 3          # r ekseninde yerel maksimum penceresi (+-3)
MERKEZ_YAKINLIK = 10    # bundan yakın merkezler aynı merkez sayılır

# Test resmindeki gerçek gruplar: merkez ve yarıçaplar
GERCEK_GRUPLAR = [((110, 140), [75, 55, 35, 15]),
                  ((300, 200), [60, 40, 20])]

GRUP_RENKLERI = [(0, 255, 0), (255, 160, 0), (0, 200, 255), (255, 0, 255)]
GRI = (60, 60, 60)
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
    rng = np.random.default_rng(2)
    resim = np.full((320, 520), 110, np.uint8)

    # Eş merkezli gruplar: dıştan içe, açık / koyu sırayla dolu daireler
    for (x, y), yaricaplar in GERCEK_GRUPLAR:
        for i, r in enumerate(yaricaplar):
            cv2.circle(resim, (x, y), r, 210 if i % 2 == 0 else 40, -1, cv2.LINE_AA)
    # 2. grubun dış halkasını kısmen örten dikdörtgen
    cv2.rectangle(resim, (345, 150), (400, 275), 160, -1)

    # Tuzaklar
    cv2.circle(resim, (300, 60), 35, 200, -1, cv2.LINE_AA)        # tek çember
    cv2.circle(resim, (450, 260), 28, 30, -1, cv2.LINE_AA)        # tek çember
    cv2.circle(resim, (440, 100), 45, 200, -1, cv2.LINE_AA)       # eksantrik: dış
    cv2.circle(resim, (452, 108), 20, 50, -1, cv2.LINE_AA)        # eksantrik: iç (14 px kayık)
    cv2.rectangle(resim, (200, 250), (240, 290), 220, -1)         # kare

    gurultulu = resim.astype(np.float64) + rng.normal(0, 10, resim.shape)
    resim = np.clip(gurultulu, 0, 255).astype(np.uint8)
    zar = rng.random(resim.shape)
    resim[zar < 0.02] = 0
    resim[zar > 0.98] = 255
    return resim


def kenar_ve_gradyan(resim):
    temiz = cv2.medianBlur(resim, 3)
    yumusak = cv2.GaussianBlur(temiz, (3, 3), 0)
    kenar = cv2.Canny(yumusak, DUSUK_ESIK, YUKSEK_ESIK, apertureSize=3, L2gradient=True)
    yon_icin = cv2.GaussianBlur(temiz, (0, 0), GRADYAN_SIGMA)
    gx = cv2.Sobel(yon_icin, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(yon_icin, cv2.CV_64F, 0, 1, ksize=3)
    aci = np.rint(np.degrees(np.arctan2(gy, gx))).astype(np.int32) % ACI_SAYISI
    return kenar, aci


# ---------------------------------------------------------------------------
# Hough dönüşümü
# ---------------------------------------------------------------------------
def sin_cos_tablosu():
    """0..359 derece için cos ve sin listeleri, bir kez hesaplanır."""
    cos_tablosu = [math.cos(math.radians(k)) for k in range(ACI_SAYISI)]
    sin_tablosu = [math.sin(math.radians(k)) for k in range(ACI_SAYISI)]
    return cos_tablosu, sin_tablosu


def radyal_oylama(kenar, aci, yaricaplar, cos_tablosu, sin_tablosu):
    """Her kenar pikseli yalnızca gradyan doğrultusundaki merkezlere oy verir.
    3 boyutlu oylama matrisi H[b, a, r_indeksi] döner."""
    yukseklik, genislik = kenar.shape
    H = np.zeros((yukseklik, genislik, len(yaricaplar)), dtype=np.int32)

    ys, xs = np.nonzero(kenar)
    for x, y in zip(xs.tolist(), ys.tolist()):           # her kenar pikseli
        k = int(aci[y, x])
        c, s = cos_tablosu[k], sin_tablosu[k]             # tablodan, hesap yok
        for ri, r in enumerate(yaricaplar):               # her yarıçap
            for yon in (1, -1):                           # merkez gradyanın iki yanında olabilir
                a = int(round(x - yon * r * c))
                b = int(round(y - yon * r * s))
                if 0 <= a < genislik and 0 <= b < yukseklik:
                    H[b, a, ri] += 1                      # oy ver
    return H


def oran_hesapla(H, yaricaplar):
    """Her r katmanında 3x3 komşuluk toplamı, 2*pi*r'ye bölünür."""
    toplam = np.zeros(H.shape, np.float64)
    dolgulu = np.pad(H, ((1, 1), (1, 1), (0, 0)))
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            toplam += dolgulu[dy:dy + H.shape[0], dx:dx + H.shape[1], :]
    return toplam / (2 * np.pi * np.array(yaricaplar, np.float64))


def halkalari_bul(oran, yaricaplar, esik=HALKA_ESIGI):
    """Her (a, b) kolonunda eşiği geçen ve r ekseninde yerel maksimum olan katmanlar."""
    tepe = oran >= esik
    for d in range(1, R_KOMSULUK + 1):
        # r - d ve r + d katmanlarından küçük olmamalı (kenarda karşılaştırma yok)
        tepe[:, :, d:] &= oran[:, :, d:] >= oran[:, :, :-d]
        tepe[:, :, :-d] &= oran[:, :, :-d] > oran[:, :, d:]
    return tepe


def merkezleri_sec(oran, tepe, yaricaplar):
    """Kolonundaki halka sayısına göre merkezleri seçer. Döner:
    [(halka_sayısı, toplam_oran, a, b, [(r, oran), ...]), ...]"""
    halka_sayisi = tepe.sum(axis=2)
    guc = (oran * tepe).sum(axis=2)
    adaylar = np.argwhere(halka_sayisi >= 1)
    # önce halka sayısı, sonra toplam güç büyük olan
    sirali = sorted(adaylar.tolist(), key=lambda p: (halka_sayisi[p[0], p[1]], guc[p[0], p[1]]), reverse=True)

    secilen = []
    for b, a in sirali:
        if any(math.hypot(a - sa, b - sb) < MERKEZ_YAKINLIK for _, _, sa, sb, _ in secilen):
            continue
        halkalar = [(yaricaplar[ri], float(oran[b, a, ri])) for ri in np.nonzero(tepe[b, a])[0]]
        secilen.append((int(halka_sayisi[b, a]), float(guc[b, a]), a, b, halkalar))
    return secilen


# ---------------------------------------------------------------------------
# Gösterme
# ---------------------------------------------------------------------------
def renkli_harita(matris):
    norm = matris.astype(np.float64) / max(matris.max(), 1e-9)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def kolon_grafigi(oran, merkezler, yaricaplar, boyut):
    """Seçilen merkezlerin H kolonları: oran - r grafiği."""
    gh, gw = boyut
    grafik = np.zeros((gh, gw, 3), np.uint8)
    sol, alt, ust = 30, gh - 25, 35
    tavan = 1.25   # dikey eksenin üst sınırı (oran 1'i biraz geçebilir)
    esik_y = int(alt - HALKA_ESIGI / tavan * (alt - ust))
    cv2.line(grafik, (sol, esik_y), (gw - 5, esik_y), (90, 90, 90), 1)
    cv2.putText(grafik, f"esik {HALKA_ESIGI}", (gw - 80, esik_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    for r in range(EN_KUCUK_R, EN_BUYUK_R + 1, 10):
        x = sol + (r - EN_KUCUK_R) * (gw - sol - 10) // (EN_BUYUK_R - EN_KUCUK_R)
        cv2.putText(grafik, str(r), (x - 8, gh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for renk, (a, b) in merkezler:
        noktalar = [(sol + ri * (gw - sol - 10) // (len(yaricaplar) - 1),
                     int(alt - min(oran[b, a, ri], tavan) / tavan * (alt - ust)))
                    for ri in range(len(yaricaplar))]
        cv2.polylines(grafik, [np.array(noktalar, np.int32)], False, renk, 1, cv2.LINE_AA)
    return grafik


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
    kenar, aci = kenar_ve_gradyan(resim)
    print("Boyut:", resim.shape, "| Kenar pikseli:", np.count_nonzero(kenar))

    yaricaplar = list(range(EN_KUCUK_R, EN_BUYUK_R + 1))
    cos_tablosu, sin_tablosu = sin_cos_tablosu()

    # --- Oylamanın kendisi: tek bir pikselin birkaç oyu ---
    ys, xs = np.nonzero(kenar)
    x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    k = int(aci[y, x])
    print(f"Oylama örneği, kenar pikseli (x={x}, y={y}), gradyan açısı {k} derece "
          f"(COS[{k}] = {cos_tablosu[k]:+.3f}, SIN[{k}] = {sin_tablosu[k]:+.3f}):")
    for r in (20, 40, 60):
        for yon in (1, -1):
            a = int(round(x - yon * r * cos_tablosu[k]))
            b = int(round(y - yon * r * sin_tablosu[k]))
            print(f"   r = {r}, yön {yon:+d}: a = {a:4d}, b = {b:4d}  ->  H[{b}, {a}, r={r}] += 1")

    baslangic = time.perf_counter()
    H = radyal_oylama(kenar, aci, yaricaplar, cos_tablosu, sin_tablosu)
    print(f"Oylama süresi: {time.perf_counter() - baslangic:.2f} s | matris boyutu (b, a, r): {H.shape} | "
          f"toplam oy: {H.sum()} (= kenar x {len(yaricaplar)} r x 2 yön, sınır dışı hariç)")

    oran = oran_hesapla(H, yaricaplar)
    tepe = halkalari_bul(oran, yaricaplar)
    merkezler = merkezleri_sec(oran, tepe, yaricaplar)
    gruplar = [m for m in merkezler if m[0] >= EN_AZ_HALKA]
    tekler = [m for m in merkezler if m[0] < EN_AZ_HALKA]

    print(f"\nHalka eşiği {HALKA_ESIGI}, en az {EN_AZ_HALKA} halka:")
    print(f"  {len(gruplar)} eş merkezli grup")
    for n, g, a, b, halkalar in gruplar:
        print(f"   merkez ({a:3d}, {b:3d}): {n} halka -> " +
              ", ".join(f"r={r} ({o:.2f})" for r, o in halkalar))
    print(f"  {len(tekler)} tek halkalı merkez (eş merkezli değil)")
    for n, g, a, b, halkalar in tekler:
        print(f"   merkez ({a:3d}, {b:3d}): r={halkalar[0][0]} ({halkalar[0][1]:.2f})")
    if gercek_var:
        print("  Gerçek gruplar:", "; ".join(f"({x}, {y}) r={rs}" for (x, y), rs in GERCEK_GRUPLAR))

    # --- Karşılaştırma: cv2.HoughCircles ---
    cv_cember = cv2.HoughCircles(kenar, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=24,
                                 minRadius=EN_KUCUK_R, maxRadius=EN_BUYUK_R)
    cv_cember = [] if cv_cember is None else cv_cember[0]
    print(f"\nKarşılaştırma: cv2.HoughCircles -> {len(cv_cember)} çember")
    for cx, cy, cr in cv_cember:
        print(f"   ({cx:5.1f}, {cy:5.1f}) r = {cr:4.1f}")

    # --- Çizim ---
    sonuc = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for n, g, a, b, halkalar in tekler:
        cv2.circle(sonuc, (a, b), halkalar[0][0], GRI, 2, cv2.LINE_AA)
    for i, (n, g, a, b, halkalar) in enumerate(gruplar):
        renk = GRUP_RENKLERI[i % len(GRUP_RENKLERI)]
        for r, o in halkalar:
            cv2.circle(sonuc, (a, b), r, renk, 2, cv2.LINE_AA)
        cv2.circle(sonuc, (a, b), 3, KIRMIZI, -1)
        cv2.putText(sonuc, f"{n} halka", (a + 6, b - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, KIRMIZI, 1, cv2.LINE_AA)

    cv_sonuc = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for cx, cy, cr in cv_cember:
        cv2.circle(cv_sonuc, (int(round(cx)), int(round(cy))), int(round(cr)), (0, 255, 0), 2, cv2.LINE_AA)

    # Oylama matrisi görünümleri
    # tek piksellik tepeler görünsün diye genişletilir
    genislet = np.ones((7, 7), np.uint8)
    halka_haritasi = cv2.dilate(renkli_harita(tepe.sum(axis=2).astype(np.float64)), genislet)
    for n, g, a, b, halkalar in merkezler:
        cv2.putText(halka_haritasi, str(n), (a + 6, b - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    toplam_oy = renkli_harita(np.sqrt(H.sum(axis=2)))
    kesit_r = GERCEK_GRUPLAR[0][1][1] if gercek_var else yaricaplar[len(yaricaplar) // 2]
    kesit = cv2.dilate(renkli_harita(H[:, :, kesit_r - EN_KUCUK_R]), np.ones((3, 3), np.uint8))

    # Tek bir pikselin oyları: yalnızca gradyan doğrultusundaki iki ışın
    tek_kenar = np.zeros_like(kenar)
    tek_kenar[y, x] = 255
    tek_H = radyal_oylama(tek_kenar, aci, yaricaplar, cos_tablosu, sin_tablosu)
    isin = cv2.dilate(renkli_harita(np.argmax(tek_H, axis=2) * (tek_H.sum(axis=2) > 0) + (tek_H.sum(axis=2) > 0) * 10),
                      np.ones((3, 3), np.uint8))
    cv2.circle(isin, (x, y), 4, (255, 255, 255), 1)

    secili = [(GRUP_RENKLERI[i % len(GRUP_RENKLERI)], (a, b)) for i, (_, _, a, b, _) in enumerate(gruplar)]
    secili += [((150, 150, 150), (a, b)) for _, _, a, b, _ in tekler[:4]]
    grafik = kolon_grafigi(oran, secili, yaricaplar, resim.shape)

    ust = np.hstack([etiket_ekle(resim, "Orijinal"), etiket_ekle(kenar, "Kenar (medyan+Canny)"),
                     etiket_ekle(sonuc, f"Es merkezli: {len(gruplar)} grup (gri: tek)")])
    orta = np.hstack([etiket_ekle(isin, "Tek pikselin oylari: gradyan isinlari"),
                      etiket_ekle(toplam_oy, "sum_r H(a, b, r)"),
                      etiket_ekle(kesit, f"H[:, :, r={kesit_r}] kesiti")])
    alt = np.hstack([etiket_ekle(halka_haritasi, "Kolon basina halka sayisi (>=2: es merkezli)"),
                     etiket_ekle(grafik, "Merkez kolonlari: oran - r"),
                     etiket_ekle(cv_sonuc, f"cv2.HoughCircles: {len(cv_cember)} cember")])
    karsilastirma = np.vstack([ust, orta, alt])

    kaydet(os.path.join(CIKTI_KLASORU, "kenar.png"), kenar)
    kaydet(os.path.join(CIKTI_KLASORU, "es_merkezli_sonuc.png"), sonuc)
    kaydet(os.path.join(CIKTI_KLASORU, "oylama_toplam.png"), toplam_oy)
    kaydet(os.path.join(CIKTI_KLASORU, "halka_sayisi_haritasi.png"), halka_haritasi)
    kaydet(os.path.join(CIKTI_KLASORU, "opencv_houghcircles.png"), cv_sonuc)
    kaydet(os.path.join(CIKTI_KLASORU, "es_merkezli_karsilastirma.png"), karsilastirma)

    pencere = "Elle Hough - es merkezli cemberler"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
