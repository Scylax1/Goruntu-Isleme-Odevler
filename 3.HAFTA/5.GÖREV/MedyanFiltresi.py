"""
Görüntü İşleme Ödevi - 5x5 Medyan Filtresi ile Tuz-Biber (Salt-and-Pepper) Gürültüsü Bastırma
---------------------------------------------------------------------------------------------
Tuz-biber gürültülü, tek kanallı (gri) bir görüntüyü OpenCV ile açar ve:

  1) OpenCV'nin hazır medyan filtresi (cv2.medianBlur) ile 5x5 boyutunda düzeltir.
  2) Aynı 5x5 medyan filtresini hazır fonksiyon KULLANMADAN kendisi yapar:
       - Her pikselin 5x5 komşuluğundaki 25 değeri toplar,
       - bunları kendi yazdığı eklemeli sıralama (insertion sort) ile sıralar,
       - ortadaki (13.) değeri yeni piksel değeri yapar.
  3) Kendi sonucunu OpenCV'ninkiyle karşılaştırır, gürültünün ne kadar
     bastırıldığını PSNR ile ölçer.

Tuz-biber gürültüsü: bazı pikseller rastgele tam beyaz (255, tuz) ya da tam siyah
(0, biber) olur. Ortalama filtresi bu uç değerleri komşulara yayar; medyan ise
sıralamada uçlara düştükleri için onları tamamen atar.

Medyan için np.median, np.sort, sorted, list.sort gibi hazır fonksiyonlar
KULLANILMAZ. cv2.medianBlur yalnızca 1. adımda ve sonucu doğrulamak için çağrılır.

Varsayılan girdi "tuz_biber.png" resmidir. O resim yoksa 2. haftadaki (1. görev)
cameraman resmine tuz-biber gürültüsü eklenerek üretilir.

Kullanım:
    python MedyanFiltresi.py [resim_yolu]

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
GURULTULU_RESIM = os.path.join(CIKTI_KLASORU, "tuz_biber.png")

# Gürültülü resim üretirken: piksellerin %5'i tuz (255), %5'i biber (0) olur
TUZ_ORANI = 0.05
BIBER_ORANI = 0.05
TOHUM = 0                  # aynı gürültü her seferinde aynı çıksın diye

FILTRE_BOYUTU = 5          # 5x5 medyan filtresi

# Karşılaştırmada büyütülerek gösterilecek bölge (y1, y2, x1, x2): kameranın olduğu yer
YAKIN_CEKIM = (100, 228, 220, 348)


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
# 1) Tuz-biber gürültülü resmi hazırlama
# ---------------------------------------------------------------------------
def tuz_biber_resim_olustur():
    """Cameraman resminde rastgele seçilen pikselleri 255 (tuz) ya da 0 (biber) yapar."""
    temiz = gri_oku(KAYNAK_RESIM)
    zar = np.random.default_rng(TOHUM).random(temiz.shape)   # her piksel için 0-1 arası sayı
    gurultulu = temiz.copy()
    gurultulu[zar < BIBER_ORANI] = 0
    gurultulu[zar > 1 - TUZ_ORANI] = 255
    kaydet(GURULTULU_RESIM, gurultulu)


# ---------------------------------------------------------------------------
# 2) Hazır fonksiyonla medyan filtresi
# ---------------------------------------------------------------------------
def medyan_opencv(resim, boyut=FILTRE_BOYUTU):
    return cv2.medianBlur(resim, boyut)


# ---------------------------------------------------------------------------
# 3) Kendi medyan filtremiz (hazır fonksiyon yok)
# ---------------------------------------------------------------------------
def sinirla(i, n):
    """Resmin dışına taşan indeksi en yakın kenar pikseline çeker (OpenCV'deki BORDER_REPLICATE).

    -2, -1 -> 0,   n, n+1 -> n-1  (kenar pikseli tekrar edilir)
    """
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


def eklemeli_sirala(degerler):
    """Listeyi küçükten büyüğe sıralar (yerinde, insertion sort).

    Her elemanı alıp, solundaki sıralı kısımda kendinden büyükleri bir sağa
    kaydırarak doğru yerine yerleştirir. Kart oyununda eldeki kartları dizmek gibi.
    """
    for i in range(1, len(degerler)):
        anahtar = degerler[i]
        j = i - 1
        while j >= 0 and degerler[j] > anahtar:
            degerler[j + 1] = degerler[j]
            j -= 1
        degerler[j + 1] = anahtar
    return degerler


def medyan_kendi(resim, boyut=FILTRE_BOYUTU):
    """Her pikseli, boyut x boyut komşuluğundaki değerlerin medyanıyla değiştirir."""
    yukseklik, genislik = resim.shape
    yaricap = boyut // 2                   # 5x5 için 2
    orta = (boyut * boyut) // 2            # 25 değerin ortası: 12. indeks (13. değer)
    sonuc = np.zeros((yukseklik, genislik), dtype=np.uint8)

    for y in range(yukseklik):
        for x in range(genislik):
            komsular = []
            for i in range(-yaricap, yaricap + 1):
                yy = sinirla(y + i, yukseklik)
                for j in range(-yaricap, yaricap + 1):
                    komsular.append(int(resim[yy, sinirla(x + j, genislik)]))
            sonuc[y, x] = eklemeli_sirala(komsular)[orta]
    return sonuc


# ---------------------------------------------------------------------------
# 4) Gürültü ölçümü
# ---------------------------------------------------------------------------
def psnr(temiz, resim):
    """Tepe sinyal-gürültü oranı (dB). Ne kadar yüksekse temiz resme o kadar yakın.

    MSE = ortalama( (temiz - resim)^2 ),   PSNR = 10 * log10( 255^2 / MSE )
    """
    mse = np.mean((temiz.astype(np.float64) - resim.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


# ---------------------------------------------------------------------------
# 5) Sonuçları gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR) if resim.ndim == 2 else resim.copy()
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def yakin_cekim(resim):
    """YAKIN_CEKIM bölgesini keser, gürültü görünsün diye 4 kat büyütür (piksel piksel)."""
    y1, y2, x1, x2 = YAKIN_CEKIM
    return cv2.resize(resim[y1:y2, x1:x2], (resim.shape[1], resim.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def bolgeyi_isaretle(resim):
    """Yakın çekim bölgesini resmin üzerinde kırmızı çerçeveyle gösterir."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    y1, y2, x1, x2 = YAKIN_CEKIM
    cv2.rectangle(kopya, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return kopya


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        yol = sys.argv[1]
    else:
        if not os.path.exists(GURULTULU_RESIM):
            tuz_biber_resim_olustur()
        yol = GURULTULU_RESIM

    # 1) Tuz-biber gürültülü, tek kanallı resmi OpenCV ile aç
    gurultulu = gri_oku(yol)
    if gurultulu is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", gurultulu.shape, "| Kanal sayısı: 1")
    print(f"Tuz (255) piksel: {np.count_nonzero(gurultulu == 255)} | "
          f"biber (0) piksel: {np.count_nonzero(gurultulu == 0)}")

    # 2) OpenCV'nin medyan filtresi
    hazir = medyan_opencv(gurultulu)

    # 3) Kendi medyan filtremiz
    baslangic = time.perf_counter()
    kendi = medyan_kendi(gurultulu)
    print(f"Kendi {FILTRE_BOYUTU}x{FILTRE_BOYUTU} medyan filtresi süresi: "
          f"{time.perf_counter() - baslangic:.2f} s")

    fark = np.abs(kendi.astype(int) - hazir.astype(int))
    print(f"Kontrol -> cv2.medianBlur ile en büyük piksel farkı: {fark.max()} | "
          f"farklı piksel sayısı: {np.count_nonzero(fark)}")

    # 4) Gürültü ne kadar bastırıldı? (temiz resim varsa onunla karşılaştır)
    temiz = gri_oku(KAYNAK_RESIM) if len(sys.argv) == 1 else None
    if temiz is not None:
        once = psnr(temiz, gurultulu)
        print(f"PSNR (temiz resme göre) -> gürültülü: {once:.2f} dB | "
              f"OpenCV medyan: {psnr(temiz, hazir):.2f} dB | kendi medyan: {psnr(temiz, kendi):.2f} dB")
        # Karşılaştırma için: aynı boyutta ortalama filtresi tuz-biberde ne yapardı?
        print(f"Karşılaştırma -> {FILTRE_BOYUTU}x{FILTRE_BOYUTU} ortalama filtresi (cv2.blur): "
              f"{psnr(temiz, cv2.blur(gurultulu, (FILTRE_BOYUTU, FILTRE_BOYUTU))):.2f} dB")
        resimler = [temiz, gurultulu, hazir, kendi]
        yazilar = ["Temiz (referans)", f"Tuz-biber  PSNR {once:.1f} dB",
                   f"OpenCV medyan  {psnr(temiz, hazir):.1f} dB", f"Kendi medyan  {psnr(temiz, kendi):.1f} dB"]
    else:
        resimler = [gurultulu, hazir, kendi]
        yazilar = ["Tuz-biber", "OpenCV medyan 5x5", "Kendi medyan 5x5"]

    kaydet(os.path.join(CIKTI_KLASORU, "medyan_opencv_sonuc.png"), hazir)
    kaydet(os.path.join(CIKTI_KLASORU, "medyan_kendi_sonuc.png"), kendi)

    ust = np.hstack([etiket_ekle(bolgeyi_isaretle(r), yazi) for r, yazi in zip(resimler, yazilar)])
    alt = np.hstack([etiket_ekle(yakin_cekim(r), "Yakin cekim (4x)") for r in resimler])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "medyan_karsilastirma.png"), karsilastirma)

    pencere = "5x5 medyan filtresi - Tuz-biber / OpenCV / Kendi"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
