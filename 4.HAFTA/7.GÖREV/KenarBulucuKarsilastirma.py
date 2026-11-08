"""
Görüntü İşleme Ödevi - Canny ve Diğer Kenar Bulucuların Süre ve Doğruluk Karşılaştırması
---------------------------------------------------------------------------------------
Kıyaslanan beş yöntem (hepsinde önce 3x3 Gauss, LoG'da yumuşatma çekirdeğin içinde):

  Prewitt + Gauss    : |G| > eşik                    (4. hafta 1-2. görev)
  Sobel + Gauss      : |G| > eşik                    (4. hafta 3. görev)
  Laplacian + Gauss  : eşikli sıfır geçişi           (4. hafta 5. görev)
  LoG 7x7            : eşikli sıfır geçişi           (4. hafta 6. görev)
  Canny              : Sobel + inceltme + histerezis (cv2.Canny)

1) ÇALIŞMA SÜRESİ (512x512 cameraman)
   - OpenCV ile (C++): cv2.GaussianBlur, cv2.filter2D, cv2.Sobel, cv2.Laplacian,
     cv2.Canny. Kenar kararı (eşik / sıfır geçişi) numpy ile. 50 ölçümün ortancası.
   - Elle (Python döngüsü): önceki görevlerdeki konvolüsyon fonksiyonu, 1 ölçüm.
     Canny'nin elle yazılmış bir sürümü yok.

2) DOĞRULUK (kenarları bilinen yapay test resmi)
   Cameraman'de hangi pikselin gerçekten kenar olduğu bilinmediği için doğruluk
   orada ölçülemez. Bunun yerine farklı kontrastlı şekillerden (disk, dikdörtgen,
   üçgen, elips, düşük kontrastlı kare) oluşan bir resim çizilir; şekil sınırları
   "gerçek kenar" haritasını verir. Resme sigma = 0, 10, 20 Gauss gürültüsü eklenir.

   Bulunan kenar pikseli, gerçek bir kenara en fazla TOLERANS (1.5) piksel
   uzaktaysa doğru sayılır:
       precision = doğru bulunan / bulunan        (bulunanların ne kadarı gerçek)
       recall    = yakalanan gerçek / tüm gerçek  (gerçeklerin ne kadarı bulundu)
       F1        = 2 PR / (P + R)

   Eşik seçimi sonucu çarpıtmasın diye her yöntemin eşiği taranır ve her gürültü
   düzeyinde en iyi F1 raporlanır. Ayrıca sigma = 10'da seçilen eşik sabit
   tutularak diğer gürültü düzeylerine uygulanır (eşiğe dayanıklılık).

OpenCV ile yapılan filtrelerin önceki görevlerdeki elle yazılmış sürümlerle aynı
sonucu verdiği başta kontrol edilir.

Kullanım:
    python KenarBulucuKarsilastirma.py [resim_yolu]
"""

import importlib.util
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

TOLERANS = 1.5
GURULTU_DUZEYLERI = (0, 10, 20)
SURE_TEKRARI = 50

# 6. görevdeki LoG çekirdeği, sıfır geçişi ve elle konvolüsyon
_spec = importlib.util.spec_from_file_location(
    "LoGFiltresi", os.path.join(KLASOR, "..", "6.GÖREV", "LoGFiltresi.py"))
log_modul = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(log_modul)

konvolusyon = log_modul.konvolusyon
sifir_gecisleri = log_modul.sifir_gecisleri
gri_oku = log_modul.gri_oku
kaydet = log_modul.kaydet
etiket_ekle = log_modul.etiket_ekle
LOG_CEKIRDEGI = log_modul.log_cekirdegi()

GAUSS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float64)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)


# ---------------------------------------------------------------------------
# Yöntemler (OpenCV ile). Her biri: resim -> karar değeri haritası,
# ve (harita, eşik) -> kenar haritası. Böylece eşik taraması için
# filtre bir kez çalıştırılır.
# ---------------------------------------------------------------------------
def gauss(resim):
    return cv2.GaussianBlur(resim, (3, 3), 0)


def prewitt_yanit(resim):
    y = gauss(resim).astype(np.float64)
    gx = cv2.filter2D(y, -1, PREWITT_X) / 6
    gy = cv2.filter2D(y, -1, PREWITT_X.T) / 6
    return np.sqrt(gx ** 2 + gy ** 2)


def sobel_yanit(resim):
    y = gauss(resim)
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3) / 8
    return np.sqrt(gx ** 2 + gy ** 2)


def laplacian_yanit(resim):
    return cv2.Laplacian(gauss(resim), cv2.CV_64F, ksize=1)


def log_yanit(resim):
    return cv2.filter2D(resim.astype(np.float64), -1, LOG_CEKIRDEGI)


def esikle(yanit, esik):
    return yanit > esik


def canny_yanit(resim):
    return gauss(resim)          # Canny kararı kendi içinde verir


def canny_kenar(yumusak, yuksek):
    # CannyKenar.py'deki gibi düşük eşik = yüksek / 3
    return cv2.Canny(yumusak, yuksek / 3, yuksek, apertureSize=3, L2gradient=True) > 0


# (ad, yanıt, karar, taranacak eşikler)
YONTEMLER = (
    ("Prewitt + Gauss", prewitt_yanit, esikle, np.geomspace(1, 80, 60)),
    ("Sobel + Gauss", sobel_yanit, esikle, np.geomspace(1, 80, 60)),
    ("Laplacian + Gauss", laplacian_yanit, sifir_gecisleri, np.geomspace(1, 150, 60)),
    ("LoG 7x7", log_yanit, sifir_gecisleri, np.geomspace(1, 100, 60)),
    ("Canny", canny_yanit, canny_kenar, np.geomspace(3, 1200, 60)),
)


# ---------------------------------------------------------------------------
# Kontrol: OpenCV sürümleri elle yazılanlarla aynı mı?
# ---------------------------------------------------------------------------
def elle_yanitlar(resim):
    """Önceki görevlerdeki gibi tamamen Python döngüsüyle; (ad -> (yanıt, süre))."""
    sonuc = {}

    def olc(ad, f):
        baslangic = time.perf_counter()
        yanit = f()
        sonuc[ad] = (yanit, time.perf_counter() - baslangic)

    def elle_gauss():
        return np.clip(np.floor(konvolusyon(resim, GAUSS) + 0.5), 0, 255).astype(np.uint8)

    def elle_gradyan(cekirdek, bolen):
        y = elle_gauss()
        # konvolüsyon çekirdeği çevirir; |G| işaretten bağımsız olduğu için fark etmez
        return np.sqrt((konvolusyon(y, cekirdek) / bolen) ** 2 + (konvolusyon(y, cekirdek.T) / bolen) ** 2)

    olc("Prewitt + Gauss", lambda: elle_gradyan(PREWITT_X, 6))
    olc("Sobel + Gauss", lambda: elle_gradyan(SOBEL_X, 8))
    olc("Laplacian + Gauss", lambda: konvolusyon(elle_gauss(), LAPLACIAN))
    olc("LoG 7x7", lambda: konvolusyon(resim, LOG_CEKIRDEGI))
    return sonuc


# ---------------------------------------------------------------------------
# Çalışma süresi
# ---------------------------------------------------------------------------
def sure_olc(f, tekrar=SURE_TEKRARI):
    f()                                      # ilk çağrıdaki ısınmayı ölçme
    sureler = []
    for _ in range(tekrar):
        baslangic = time.perf_counter()
        f()
        sureler.append(time.perf_counter() - baslangic)
    return float(np.median(sureler))


# ---------------------------------------------------------------------------
# Yapay test resmi ve gerçek kenar haritası
# ---------------------------------------------------------------------------
def test_resmi(boyut=512, olcek=4):
    """Şekilleri 4 kat büyük bir etiket haritasına çizer, sonra küçültür.

    Kenar pürüzsüz (antialias) olur. Gerçek kenar büyük haritada bulunur:
    sağ ya da alt komşusuyla etiketi farklı olan pikseller. Bunları içeren
    küçük pikseller gerçek kenar sayılır. (Yalnızca "4x4 bloğunda birden çok
    etiket olan piksel" demek, piksel sınırına tam denk gelen kenarları
    kaçırırdı.)
    """
    B = boyut * olcek
    etiket = np.zeros((B, B), np.uint8)

    def p(x, y):
        return int(x * olcek), int(y * olcek)

    cv2.circle(etiket, p(140, 140), 90 * olcek, 1, -1)
    cv2.rectangle(etiket, p(280, 60), p(460, 200), 2, -1)
    cv2.fillPoly(etiket, [np.array([p(70, 470), p(230, 270), p(250, 480)], np.int32)], 3)
    cv2.ellipse(etiket, p(380, 370), (110 * olcek, 60 * olcek), 25, 0, 360, 4, -1)
    cv2.rectangle(etiket, p(340, 330), p(420, 410), 5, -1)       # elipsin içinde
    cv2.circle(etiket, p(140, 140), 35 * olcek, 6, -1)           # diskin içinde

    # etiket -> parlaklık; arka plan yatay bir rampa (düz değil)
    parlaklik = np.array([90, 170, 140, 210, 50, 80, 150], np.float64)
    rampa = np.linspace(-20, 20, B)[None, :]
    buyuk = parlaklik[etiket] + np.where(etiket == 0, rampa, 0)
    resim = cv2.resize(buyuk, (boyut, boyut), interpolation=cv2.INTER_AREA)
    resim = np.clip(np.floor(resim + 0.5), 0, 255).astype(np.uint8)

    sinir = np.zeros((B, B), bool)
    sinir[:, :-1] |= etiket[:, :-1] != etiket[:, 1:]
    sinir[:-1, :] |= etiket[:-1, :] != etiket[1:, :]
    gercek = sinir.reshape(boyut, olcek, boyut, olcek).any(axis=(1, 3))
    return resim, gercek


def gurultu_ekle(resim, sigma, tohum):
    if sigma == 0:
        return resim
    uretec = np.random.default_rng(tohum)
    return np.clip(resim + uretec.normal(0, sigma, resim.shape), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Doğruluk ölçüsü
# ---------------------------------------------------------------------------
def uzaklik_haritasi(kenar):
    """Her pikselin en yakın kenar pikseline uzaklığı."""
    return cv2.distanceTransform(np.where(kenar, 0, 1).astype(np.uint8), cv2.DIST_L2, 5)


def f_olcusu(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def puanla(bulunan, gercek, gercek_uzaklik):
    """(precision, recall, F1, birebir F1) döndürür.

    Toleranslı precision kalın kenarı cezalandırmaz: 3 piksellik bir bandın her
    pikseli gerçek kenara yakın olduğu için doğru sayılır. Birebir ölçüde her
    gerçek kenar pikseli en fazla bir bulunan pikseli doğrulayabilir:
        birebir precision = min(doğru bulunan, yakalanan gerçek) / bulunan
    (Tam eşleştirmenin üst sınırıdır.) Böylece kenarı 2.5 kat kalın çizen bir
    yöntemin precision'ı ~1/2.5'e düşer.
    """
    n = np.count_nonzero(bulunan)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    dogru_bulunan = np.count_nonzero(gercek_uzaklik[bulunan] <= TOLERANS)
    yakalanan = np.count_nonzero(uzaklik_haritasi(bulunan)[gercek] <= TOLERANS)
    precision = dogru_bulunan / n
    recall = yakalanan / np.count_nonzero(gercek)
    birebir_precision = min(dogru_bulunan, yakalanan) / n
    return precision, recall, f_olcusu(precision, recall), f_olcusu(birebir_precision, recall)


# ---------------------------------------------------------------------------
# Görseller
# ---------------------------------------------------------------------------
def dogruluk_goruntu(bulunan, gercek, gercek_uzaklik):
    """yeşil: doğru bulunan, kırmızı: yanlış bulunan, mavi: kaçırılan gerçek kenar."""
    renkli = np.zeros(gercek.shape + (3,), np.uint8)
    kacirilan = gercek & (uzaklik_haritasi(bulunan) > TOLERANS)
    dogru = bulunan & (gercek_uzaklik <= TOLERANS)
    renkli[kacirilan] = (255, 120, 0)
    renkli[bulunan & ~dogru] = (0, 0, 255)
    renkli[dogru] = (0, 220, 0)
    return renkli


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape, "| OpenCV", cv2.__version__,
          "| iş parçacığı:", cv2.getNumThreads())

    # --- Elle yazılan sürümler: süre ve OpenCV ile aynılık kontrolü ---
    print("\nElle yazılmış (Python döngüsü) sürümler çalışıyor...")
    elle = elle_yanitlar(resim)
    for ad, f, _, _ in YONTEMLER[:4]:
        fark = np.abs(elle[ad][0] - f(resim)).max()
        print(f"   Kontrol -> {ad:<18} OpenCV ile en büyük fark: {fark:.1e}")

    # --- 1) Çalışma süresi ---
    print(f"\n1) Çalışma süresi (512x512, OpenCV: {SURE_TEKRARI} ölçümün ortancası)")
    print(f"   {'Yöntem':<18} | {'OpenCV':>9} | {'Elle (Python)':>13} | {'oran':>7}")
    # Eşik değeri süreyi etkilemez; önceki görevlerdekine yakın temsilî değerler
    sure_esikleri = {"Prewitt + Gauss": 12, "Sobel + Gauss": 12, "Laplacian + Gauss": 14,
                     "LoG 7x7": 11, "Canny": 150}
    sureler = {}
    for ad, f, karar, _ in YONTEMLER:
        esik = sure_esikleri[ad]
        sureler[ad] = sure_olc(lambda: karar(f(resim), esik))
        if ad in elle:
            elle_sure = elle[ad][1]
            print(f"   {ad:<18} | {sureler[ad] * 1000:6.2f} ms | {elle_sure:11.2f} s | "
                  f"{elle_sure / sureler[ad]:6.0f}x")
        else:
            print(f"   {ad:<18} | {sureler[ad] * 1000:6.2f} ms | {'-':>13} |")

    # --- 2) Doğruluk ---
    temiz, gercek = test_resmi()
    gercek_uzaklik = uzaklik_haritasi(gercek)
    print(f"\n2) Doğruluk (yapay resim, {np.count_nonzero(gercek)} gerçek kenar pikseli, "
          f"tolerans {TOLERANS} px)")

    tablo = {}       # (ad, sigma) -> (en iyi eşik, P, R, F1, kalınlık, en iyi birebir F1)
    resimler = {s: gurultu_ekle(temiz, s, tohum=s) for s in GURULTU_DUZEYLERI}
    for ad, f, karar, esikler in YONTEMLER:
        for sigma in GURULTU_DUZEYLERI:
            yanit = f(resimler[sigma])
            puanlar = [(esik,) + puanla(karar(yanit, esik), gercek, gercek_uzaklik) for esik in esikler]
            esik, p, r, f1, _ = max(puanlar, key=lambda t: t[3])
            kalinlik = np.count_nonzero(karar(yanit, esik)) / np.count_nonzero(gercek)
            # birebir F1 için eşik ayrıca seçilir (kalın kenarlı yöntemler daha yüksek eşik ister)
            birebir_f1 = max(t[4] for t in puanlar)
            tablo[ad, sigma] = (esik, p, r, f1, kalinlik, birebir_f1)

    for sigma in GURULTU_DUZEYLERI:
        print(f"   Gürültü sigma = {sigma}: (her ölçü kendi en iyi eşiğinde)")
        for ad, *_ in YONTEMLER:
            esik, p, r, f1, kalinlik, birebir_f1 = tablo[ad, sigma]
            print(f"      {ad:<18} eşik {esik:7.2f} | P {p:.3f} | R {r:.3f} | F1 {f1:.3f} | "
                  f"bulunan/gerçek {kalinlik:4.2f} | birebir F1 {birebir_f1:.3f}")

    print("   Eşik sigma = 10'da seçilip sabit tutulunca F1:")
    for ad, f, karar, _ in YONTEMLER:
        sabit = tablo[ad, 10][0]
        satir = []
        for sigma in GURULTU_DUZEYLERI:
            f1 = puanla(karar(f(resimler[sigma]), sabit), gercek, gercek_uzaklik)[2]
            satir.append(f"sigma={sigma}: {f1:.3f}")
        print(f"      {ad:<18} (eşik {sabit:6.2f}) " + " | ".join(satir))

    # --- Görseller ---
    karolar = [etiket_ekle(resimler[10], "Test resmi, s=10"),
               etiket_ekle(np.where(gercek, 255, 0).astype(np.uint8), "Gercek kenar")]
    for ad, f, karar, _ in YONTEMLER:
        esik = tablo[ad, 10][0]
        f1 = tablo[ad, 10][3]
        karolar.append(etiket_ekle(dogruluk_goruntu(karar(f(resimler[10]), esik), gercek, gercek_uzaklik),
                                   f"{ad}  F1={f1:.2f}"))
    karolar.append(etiket_ekle(np.zeros_like(karolar[0]),
                               "yesil dogru, kirmizi yanlis, mavi kacan"))
    kaydet(os.path.join(CIKTI_KLASORU, "dogruluk_sigma10.png"),
           np.vstack([np.hstack(karolar[:4]), np.hstack(karolar[4:])]))

    # Cameraman üzerinde, sigma = 10 test resminde seçilen eşiklerle
    karolar = [etiket_ekle(resim, "Orijinal")]
    for ad, f, karar, _ in YONTEMLER:
        kenar = karar(f(resim), tablo[ad, 10][0])
        karolar.append(etiket_ekle(np.where(kenar, 255, 0).astype(np.uint8), ad))
    kaydet(os.path.join(CIKTI_KLASORU, "cameraman_tum_yontemler.png"),
           np.vstack([np.hstack(karolar[:3]), np.hstack(karolar[3:])]))

    # Gürültüye göre en iyi birebir F1 grafiği (OpenCV ile basit çizim)
    grafik = np.full((360, 660, 3), 255, np.uint8)
    renkler = [(200, 120, 0), (0, 140, 255), (160, 0, 160), (0, 0, 220), (0, 160, 0)]
    def nokta(i, f1):
        return int(80 + i * 200), int(320 - f1 * 280)
    for f1 in (0, 0.25, 0.5, 0.75, 1.0):
        cv2.line(grafik, (70, nokta(0, f1)[1]), (500, nokta(0, f1)[1]), (225, 225, 225), 1)
        cv2.putText(grafik, f"{f1:.2f}", (20, nokta(0, f1)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    for i, sigma in enumerate(GURULTU_DUZEYLERI):
        cv2.putText(grafik, f"sigma={sigma}", (nokta(i, 0)[0] - 25, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    # Prewitt ve Sobel eğrileri neredeyse çakıştığı için etiketleri üst/alt kaydırılır
    etiket_kaydirma = {"Prewitt + Gauss": 14, "Sobel + Gauss": -6}
    for (ad, *_), renk in zip(YONTEMLER, renkler):
        noktalar = [nokta(i, tablo[ad, s][5]) for i, s in enumerate(GURULTU_DUZEYLERI)]
        for a, b in zip(noktalar, noktalar[1:]):
            cv2.line(grafik, a, b, renk, 2, cv2.LINE_AA)
        for n in noktalar:
            cv2.circle(grafik, n, 4, renk, -1, cv2.LINE_AA)
        cv2.putText(grafik, ad, (noktalar[-1][0] + 8, noktalar[-1][1] + 4 + etiket_kaydirma.get(ad, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, renk, 1, cv2.LINE_AA)
    cv2.putText(grafik, "Birebir F1 - gurultu", (70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    kaydet(os.path.join(CIKTI_KLASORU, "f1_gurultu_grafigi.png"), grafik)


if __name__ == "__main__":
    main()
