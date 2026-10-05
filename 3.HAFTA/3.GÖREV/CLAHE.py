"""
Görüntü İşleme Ödevi - CLAHE (Contrast Limited Adaptive Histogram Equalization)
-------------------------------------------------------------------------------
Düşük kontrastlı, tek kanallı (gri) bir görüntüyü OpenCV ile açar ve:

  1) OpenCV'nin hazır CLAHE fonksiyonu (cv2.createCLAHE) ile düzeltir.
  2) Aynı işlemi hazır fonksiyon KULLANMADAN, döngülerle kendisi yapar:
       a) Resmi 8x8 karoya (tile) böler.
       b) Her karonun 256 değerli histogramını çıkarır.
       c) Histogramı kırpar: sınırı aşan sayıları kesip tüm tonlara eşit dağıtır
          (Contrast Limited kısmı; gürültünün aşırı büyümesini engeller).
       d) Kırpılmış histogramın CDF'sinden her karo için bir dönüşüm tablosu (LUT) çıkarır.
       e) Her pikseli, çevresindeki 4 karonun tablosundan bilineer enterpolasyonla
          hesaplar (karo sınırlarında kare kare izler oluşmasın diye).
  3) Kendi sonucunu OpenCV'nin sonucuyla karşılaştırarak farklı resim ve ayarlarla test eder.

Varsayılan girdi, kontrast germe ödevinde (1. görev) üretilen "dusuk_kontrast.png" resmidir.
O resim yoksa cameraman resminin tonları [100, 160] aralığına sıkıştırılarak üretilir.

Kullanım:
    python CLAHE.py [resim_yolu]

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
DUSUK_KONTRAST_RESIM = os.path.join(CIKTI_KLASORU, "dusuk_kontrast.png")

# Düşük kontrastlı resim üretirken tonların sıkıştırılacağı aralık
DAR_ALT, DAR_UST = 100, 160

# CLAHE ayarları
KIRPMA_SINIRI = 2.0        # histogram kırpma sınırı (OpenCV'deki clipLimit)
KARO_IZGARASI = (8, 8)     # (yatay karo sayısı, dikey karo sayısı) (OpenCV'deki tileGridSize)


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
# 1) Düşük kontrastlı resmi hazırlama
# ---------------------------------------------------------------------------
def dusuk_kontrast_resim_olustur():
    """Cameraman resminin 0-255 tonlarını DAR_ALT-DAR_UST aralığına sıkıştırır."""
    resim = gri_oku(KAYNAK_RESIM)
    dar = DAR_ALT + resim.astype(np.float32) * (DAR_UST - DAR_ALT) / 255.0
    kaydet(DUSUK_KONTRAST_RESIM, dar.astype(np.uint8))


# ---------------------------------------------------------------------------
# 2) Hazır fonksiyonla CLAHE
# ---------------------------------------------------------------------------
def clahe_opencv(resim, kirpma_siniri=KIRPMA_SINIRI, karo_izgarasi=KARO_IZGARASI):
    clahe = cv2.createCLAHE(clipLimit=kirpma_siniri, tileGridSize=karo_izgarasi)
    return clahe.apply(resim)


# ---------------------------------------------------------------------------
# 3) Kendi CLAHE'miz (hazır fonksiyon yok)
# ---------------------------------------------------------------------------
def yansit(i, n):
    """Resmin dışına taşan indeksi kenardan geri yansıtır (OpenCV'deki BORDER_REFLECT_101).

    n = 5 için: ..., 3, 4 | 3, 2, ...  (kenar pikselinin kendisi tekrar edilmez)
    """
    if i >= n:
        return 2 * n - 2 - i
    return i


def karo_boyutu_bul(yukseklik, genislik, karo_x, karo_y):
    """Bir karonun (yükseklik, genişlik) değerini döndürür.

    Resim karo sayısına tam bölünmüyorsa OpenCV resmi alttan ve sağdan yansıtarak
    büyütür. Yalnızca bir kenar tam bölünmüyorsa bile İKİ kenar birden büyütülür;
    sonuç aynı çıksın diye bu davranış da aynen uygulanıyor.
    """
    if genislik % karo_x == 0 and yukseklik % karo_y == 0:
        return yukseklik // karo_y, genislik // karo_x
    yeni_yukseklik = yukseklik + karo_y - yukseklik % karo_y
    yeni_genislik = genislik + karo_x - genislik % karo_x
    return yeni_yukseklik // karo_y, yeni_genislik // karo_x


def karo_histogrami(resim, y0, x0, karo_h, karo_w):
    """Bir karodaki her tonun piksel sayısı (256 kutu)."""
    yukseklik, genislik = resim.shape
    histogram = [0] * 256
    for y in range(y0, y0 + karo_h):
        yy = yansit(y, yukseklik)
        for x in range(x0, x0 + karo_w):
            histogram[int(resim[yy, yansit(x, genislik)])] += 1
    return histogram


def histogrami_kirp(histogram, sinir):
    """Sınırı aşan sayıları keser, kesilen toplamı 256 tona eşit dağıtır.

    Tam bölünmeyen artık kısım, 0'dan başlayarak eşit aralıklı tonlara birer birer eklenir.
    Toplam piksel sayısı değişmez, sadece tepeler kısalır.
    """
    kirpilmis = list(histogram)
    kesilen = 0
    for k in range(256):
        if kirpilmis[k] > sinir:
            kesilen += kirpilmis[k] - sinir
            kirpilmis[k] = sinir

    pay = kesilen // 256
    artik = kesilen - pay * 256
    for k in range(256):
        kirpilmis[k] += pay

    if artik != 0:
        adim = max(256 // artik, 1)
        k = 0
        while k < 256 and artik > 0:
            kirpilmis[k] += 1
            k += adim
            artik -= 1
    return kirpilmis


def karo_tablosu(histogram, piksel_sayisi):
    """Kırpılmış histogramın CDF'sini [0, 255] aralığına ölçekleyip LUT yapar.

    tablo[k] = round( cdf[k] * 255 / karodaki_piksel_sayisi )
    """
    olcek = 255.0 / piksel_sayisi
    tablo = [0] * 256
    toplam = 0
    for k in range(256):
        toplam += histogram[k]
        # round(): OpenCV gibi en yakın tam sayıya yuvarlar
        tablo[k] = min(255, round(toplam * olcek))
    return tablo


# Enterpolasyon OpenCV'deki gibi 32 bit float ile yapılır. Python'un kendi float'ı 64 bit
# olduğu için onunla hesaplanınca .5 sınırındaki birkaç piksel 1 ton farklı yuvarlanıyor.
F32 = np.float32


def enterpolasyon_bilgisi(konum, karo_boyu, karo_sayisi):
    """Bir pikselin, solundaki/üstündeki ve sağındaki/altındaki karo numarası ile ağırlığı.

    Karo merkezleri arasındaki konuma göre ağırlık 0 ile 1 arasında değişir;
    resmin kenarlarında karo numarası sınırlanır, böylece tek bir karonun tablosu kullanılır.
    """
    t = F32(konum) * (F32(1) / F32(karo_boyu)) - F32(0.5)
    karo1 = int(t // 1)                 # aşağı yuvarlama (negatifte de doğru çalışır)
    agirlik = t - F32(karo1)            # ikinci karonun ağırlığı
    karo2 = min(karo1 + 1, karo_sayisi - 1)
    karo1 = max(karo1, 0)
    return karo1, karo2, agirlik


def clahe_kendi(resim, kirpma_siniri=KIRPMA_SINIRI, karo_izgarasi=KARO_IZGARASI):
    yukseklik, genislik = resim.shape
    karo_x, karo_y = karo_izgarasi
    karo_h, karo_w = karo_boyutu_bul(yukseklik, genislik, karo_x, karo_y)
    karo_piksel = karo_h * karo_w

    # Kırpma sınırı, ortalama kutu doluluğunun kaç katı olduğu şeklinde verilir
    sinir = 0
    if kirpma_siniri > 0:
        sinir = max(int(kirpma_siniri * karo_piksel / 256), 1)

    # a-d) Her karo için dönüşüm tablosu
    tablolar = []
    for ky in range(karo_y):
        satir = []
        for kx in range(karo_x):
            histogram = karo_histogrami(resim, ky * karo_h, kx * karo_w, karo_h, karo_w)
            if sinir > 0:
                histogram = histogrami_kirp(histogram, sinir)
            satir.append(karo_tablosu(histogram, karo_piksel))
        tablolar.append(satir)

    # e) Her piksel için 4 komşu karonun tablosundan bilineer enterpolasyon
    sutun_bilgisi = [enterpolasyon_bilgisi(x, karo_w, karo_x) for x in range(genislik)]
    sonuc = np.zeros((yukseklik, genislik), dtype=np.uint8)
    for y in range(yukseklik):
        ust, alt, ya = enterpolasyon_bilgisi(y, karo_h, karo_y)
        for x in range(genislik):
            sol, sag, xa = sutun_bilgisi[x]
            ton = int(resim[y, x])
            ust_deger = F32(tablolar[ust][sol][ton]) * (F32(1) - xa) + F32(tablolar[ust][sag][ton]) * xa
            alt_deger = F32(tablolar[alt][sol][ton]) * (F32(1) - xa) + F32(tablolar[alt][sag][ton]) * xa
            sonuc[y, x] = min(255, round(float(ust_deger * (F32(1) - ya) + alt_deger * ya)))
    return sonuc


# ---------------------------------------------------------------------------
# 4) Test: kendi sonucumuz OpenCV ile aynı mı?
# ---------------------------------------------------------------------------
def farki_olc(kendi, hazir):
    """(en büyük piksel farkı, farklı piksel sayısı)"""
    fark = np.abs(kendi.astype(int) - hazir.astype(int))
    return int(fark.max()), int(np.count_nonzero(fark))


def testleri_calistir(resim):
    print("\n--- Test: kendi CLAHE'miz vs cv2.createCLAHE ---")
    rastgele = np.random.default_rng(0).integers(90, 170, size=(150, 200), dtype=np.uint8)
    testler = [
        ("Ana resim, varsayılan ayar", resim, KIRPMA_SINIRI, KARO_IZGARASI),
        ("Ana resim, sınır 4.0, 4x4 karo", resim, 4.0, (4, 4)),
        ("Ana resim, kırpma yok (sınır 0)", resim, 0.0, (8, 8)),
        ("Kesilmiş 301x427 (tam bölünmüyor)", resim[:301, :427], KIRPMA_SINIRI, (8, 8)),
        ("Yalnız genişlik bölünmüyor 256x250", resim[:256, :250], KIRPMA_SINIRI, (8, 8)),
        ("Rastgele gürültü 150x200, 3x5 karo", rastgele, 3.0, (5, 3)),
    ]

    hepsi_ayni = True
    for ad, girdi, sinir, izgara in testler:
        girdi = np.ascontiguousarray(girdi)
        baslangic = time.perf_counter()
        kendi = clahe_kendi(girdi, sinir, izgara)
        sure = time.perf_counter() - baslangic
        en_buyuk, farkli = farki_olc(kendi, clahe_opencv(girdi, sinir, izgara))
        durum = "AYNI" if en_buyuk == 0 else "FARKLI"
        hepsi_ayni = hepsi_ayni and en_buyuk == 0
        print(f"  [{durum:^6}] {ad:<38} | en büyük fark: {en_buyuk} | "
              f"farklı piksel: {farkli:>5} / {girdi.size:<6} | süre: {sure:.2f} s")
    print("Sonuç:", "tüm testlerde OpenCV ile birebir aynı." if hepsi_ayni
          else "bazı testlerde fark var (yukarıya bakın).")


# ---------------------------------------------------------------------------
# 5) Sonuçları gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def histogram_ciz(resim, yukseklik=200):
    """256 sütunluk basit bir histogram görüntüsü çizer (genişlik = resim genişliği)."""
    sayac = np.zeros(256, dtype=np.int64)
    for deger in resim.ravel():
        sayac[deger] += 1

    genislik = resim.shape[1]
    tuval = np.full((yukseklik, genislik, 3), 255, dtype=np.uint8)
    en_cok = sayac.max()
    for ton in range(256):
        boy = int(sayac[ton] / en_cok * (yukseklik - 10))
        x = int(ton * genislik / 256)
        x2 = int((ton + 1) * genislik / 256)
        cv2.rectangle(tuval, (x, yukseklik - boy), (x2 - 1, yukseklik - 1), (60, 60, 60), -1)
    return tuval


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        yol = sys.argv[1]
    else:
        if not os.path.exists(DUSUK_KONTRAST_RESIM):
            dusuk_kontrast_resim_olustur()
        yol = DUSUK_KONTRAST_RESIM

    # 1) Düşük kontrastlı, tek kanallı resmi OpenCV ile aç
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape, "| Kanal sayısı: 1")
    print(f"CLAHE ayarları -> kırpma sınırı: {KIRPMA_SINIRI} | karo ızgarası: "
          f"{KARO_IZGARASI[0]}x{KARO_IZGARASI[1]}")

    # 2) Hazır fonksiyonla düzelt
    hazir = clahe_opencv(resim)

    # 3) Kendi CLAHE'mizle düzelt
    baslangic = time.perf_counter()
    kendi = clahe_kendi(resim)
    print(f"Kendi CLAHE süresi: {time.perf_counter() - baslangic:.2f} s")

    en_buyuk, farkli = farki_olc(kendi, hazir)
    print(f"Kontrol -> cv2.createCLAHE ile en büyük piksel farkı: {en_buyuk} | "
          f"farklı piksel sayısı: {farkli}")

    # 4) Farklı resim ve ayarlarla test
    testleri_calistir(resim)

    kaydet(os.path.join(CIKTI_KLASORU, "clahe_opencv_sonuc.png"), hazir)
    kaydet(os.path.join(CIKTI_KLASORU, "clahe_kendi_sonuc.png"), kendi)

    ust = np.hstack([etiket_ekle(resim, f"Orijinal [{resim.min()}, {resim.max()}]"),
                     etiket_ekle(hazir, "CLAHE - OpenCV"),
                     etiket_ekle(kendi, "CLAHE - kendi")])
    alt = np.hstack([histogram_ciz(resim), histogram_ciz(hazir), histogram_ciz(kendi)])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "clahe_karsilastirma.png"), karsilastirma)

    pencere = "CLAHE - Orijinal / OpenCV / Kendi"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
