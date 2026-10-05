"""
Görüntü İşleme Ödevi - 3x3 Ortalama Filtresi ile Gürültü Bastırma (Konvolüsyon)
-------------------------------------------------------------------------------
Gürültülü, tek kanallı (gri) bir görüntüyü OpenCV ile açar ve 3x3 ortalama
filtresini konvolüsyonla resmin üzerinde gezdirir:

            | 1 1 1 |
    K = 1/9 | 1 1 1 |        yeni(y, x) = toplam  K(i, j) * resim(y - i, x - j)
            | 1 1 1 |                    i,j = -1..1

Her piksel, kendisi ve 8 komşusunun ortalamasıyla değiştirilir. Gürültü
rastgele olduğu için komşular arasında ortalaması sıfıra yaklaşır ve bastırılır;
bedeli, kenarların biraz bulanıklaşmasıdır.

Konvolüsyon için cv2.blur, cv2.filter2D, cv2.boxFilter gibi hazır fonksiyonlar
KULLANILMAZ; çekirdek döngüyle piksel piksel gezdirilir. Hazır fonksiyonlar
yalnızca sonucu doğrulamak için çağrılır.

Varsayılan girdi "gurultulu.png" resmidir. O resim yoksa 2. haftadaki (1. görev)
cameraman resmine Gauss gürültüsü eklenerek üretilir.

Kullanım:
    python OrtalamaFiltresi.py [resim_yolu]

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
GURULTULU_RESIM = os.path.join(CIKTI_KLASORU, "gurultulu.png")

# Gürültülü resim üretirken eklenen Gauss gürültüsünün standart sapması
GURULTU_SAPMASI = 20
TOHUM = 0                  # aynı gürültü her seferinde aynı çıksın diye

# 3x3 ortalama çekirdeği: 9 hücrenin hepsi 1/9, toplamı 1 (resmin parlaklığı değişmez)
CEKIRDEK = [[1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]]

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
# 1) Gürültülü resmi hazırlama
# ---------------------------------------------------------------------------
def gurultulu_resim_olustur():
    """Cameraman resminin her pikseline ortalaması 0 olan Gauss gürültüsü ekler."""
    temiz = gri_oku(KAYNAK_RESIM)
    gurultu = np.random.default_rng(TOHUM).normal(0, GURULTU_SAPMASI, temiz.shape)
    gurultulu = np.clip(temiz + gurultu, 0, 255).round().astype(np.uint8)
    kaydet(GURULTULU_RESIM, gurultulu)


# ---------------------------------------------------------------------------
# 2) Konvolüsyon (hazır fonksiyon yok)
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

    Konvolüsyonda çekirdek 180 derece çevrilerek uygulanır: resim(y - i, x - j).
    Ortalama çekirdeği simetrik olduğu için çevirmek sonucu değiştirmez, ama
    tanım gereği öyle yazıldı (çevirmeden yapılan işlemin adı korelasyondur).
    Kenar pikselleri için resmin dışına taşan komşular kenardan yansıtılır.
    """
    yukseklik, genislik = resim.shape
    yaricap = len(cekirdek) // 2           # 3x3 için 1
    sonuc = np.zeros((yukseklik, genislik), dtype=np.uint8)

    for y in range(yukseklik):
        for x in range(genislik):
            toplam = 0.0
            for i in range(-yaricap, yaricap + 1):
                yy = yansit(y - i, yukseklik)
                for j in range(-yaricap, yaricap + 1):
                    xx = yansit(x - j, genislik)
                    toplam += cekirdek[i + yaricap][j + yaricap] * int(resim[yy, xx])
            sonuc[y, x] = min(255, max(0, round(toplam)))   # en yakın tam sayıya yuvarla
    return sonuc


# ---------------------------------------------------------------------------
# 3) Gürültü ölçümü
# ---------------------------------------------------------------------------
def psnr(temiz, resim):
    """Tepe sinyal-gürültü oranı (dB). Ne kadar yüksekse temiz resme o kadar yakın.

    MSE = ortalama( (temiz - resim)^2 ),   PSNR = 10 * log10( 255^2 / MSE )
    """
    mse = np.mean((temiz.astype(np.float64) - resim.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


# ---------------------------------------------------------------------------
# 4) Sonuçları gösterme
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
            gurultulu_resim_olustur()
        yol = GURULTULU_RESIM

    # 1) Gürültülü, tek kanallı resmi OpenCV ile aç
    gurultulu = gri_oku(yol)
    if gurultulu is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", gurultulu.shape, "| Kanal sayısı: 1")

    # 2) 3x3 ortalama filtresini konvolüsyonla gezdir
    baslangic = time.perf_counter()
    filtreli = konvolusyon(gurultulu, CEKIRDEK)
    print(f"Konvolüsyon süresi: {time.perf_counter() - baslangic:.2f} s")

    # Sadece doğrulama amaçlı: OpenCV'nin hazır sonuçlarıyla karşılaştır
    for ad, hazir in (("cv2.blur", cv2.blur(gurultulu, (3, 3))),
                      ("cv2.filter2D", cv2.filter2D(gurultulu, -1, np.array(CEKIRDEK, np.float32)))):
        fark = np.abs(filtreli.astype(int) - hazir.astype(int))
        print(f"Kontrol -> {ad} ile en büyük piksel farkı: {fark.max()} | "
              f"farklı piksel sayısı: {np.count_nonzero(fark)}")

    # 3) Gürültü ne kadar bastırıldı? (temiz resim varsa onunla karşılaştır)
    temiz = gri_oku(KAYNAK_RESIM) if len(sys.argv) == 1 else None
    if temiz is not None:
        once, sonra = psnr(temiz, gurultulu), psnr(temiz, filtreli)
        print(f"PSNR (temiz resme göre) -> gürültülü: {once:.2f} dB | "
              f"filtreli: {sonra:.2f} dB | iyileşme: +{sonra - once:.2f} dB")
        ust_yazilar = ["Temiz (referans)", f"Gurultulu  PSNR {once:.1f} dB", f"3x3 ortalama  PSNR {sonra:.1f} dB"]
        resimler = [temiz, gurultulu, filtreli]
    else:
        ust_yazilar = ["Gurultulu", "3x3 ortalama"]
        resimler = [gurultulu, filtreli]

    kaydet(os.path.join(CIKTI_KLASORU, "ortalama_filtre_sonuc.png"), filtreli)

    ust = np.hstack([etiket_ekle(bolgeyi_isaretle(r), yazi) for r, yazi in zip(resimler, ust_yazilar)])
    alt = np.hstack([etiket_ekle(yakin_cekim(r), "Yakin cekim (4x)") for r in resimler])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "ortalama_filtre_karsilastirma.png"), karsilastirma)

    pencere = "3x3 ortalama filtresi - Gurultulu / Filtreli"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
