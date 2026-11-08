"""
Görüntü İşleme Ödevi - 7x7 Laplacian of Gaussian (LoG) ile Kenar Bulma
----------------------------------------------------------------------
Tek kanallı (gri) bir görüntüyü OpenCV ile açar ve iki adımda işler:

  1) 7x7 LoG çekirdeği ile konvolüsyon.

     LoG, Gauss fonksiyonunun Laplacian'ıdır (Marr-Hildreth kenar bulucu):

        LoG(x, y) = (x^2 + y^2 - 2 sigma^2) / sigma^4 * exp(-(x^2 + y^2) / (2 sigma^2))

     Konvolüsyon birleşmeli olduğu için  Laplacian(Gauss * resim) = LoG * resim.
     Yani "önce Gauss ile yumuşat, sonra Laplacian uygula" işlemi tek bir
     çekirdekle, tek geçişte yapılır. Bu yüzden resim ayrıca Gauss
     filtresinden geçirilmez; yumuşatma çekirdeğin içindedir.

     Çekirdek boyutu 7x7 -> yarıçap 3. Gauss'un ~%99.7'si +-3 sigma içinde
     kaldığı için sigma = 1.0 seçildi. Kesilen çekirdek iki adımda düzeltilir:
       - Ortalaması çıkarılır: toplam tam 0 olur, düz bölgede sonuç 0.
       - f(x, y) = x^2 resmine uygulanınca 2 (gerçek Laplacian) verecek
         şekilde ölçeklenir. Böylece sonuç 3x3 Laplacian ile aynı birimde olur.

  2) Sıfır geçişi (zero crossing) ile kenar bulma:
     Bir piksel ile sağ ya da alt komşusu arasında LoG işaret değiştiriyorsa
     ve bu sıçrama yeterince büyükse (> SIFIR_GECIS_ESIGI) o piksel kenar
     sayılır. Kenarlar 1 piksel inceliğinde çıkar.

Konvolüsyon döngüyle piksel piksel yapılır; cv2.filter2D yalnızca sonucu
doğrulamak için çağrılır.

Varsayılan girdi 2. haftadaki (1. görev) cameraman resmidir.

Kullanım:
    python LoGFiltresi.py [resim_yolu]

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

LOG_BOYUTU = 7
LOG_SIGMA = 1.0

# Sıfır geçişinde iki komşu arasındaki LoG farkı bunu aşmalı.
# 150 gri seviyelik bir kenardaki sıçramanın ~%20'si (bkz. LoGKarsilastirma.py: 10.66)
SIFIR_GECIS_ESIGI = 11


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
# 1) LoG çekirdeği
# ---------------------------------------------------------------------------
def log_cekirdegi(boyut=LOG_BOYUTU, sigma=LOG_SIGMA):
    """boyut x boyut LoG çekirdeği: toplamı 0, x^2 resmine yanıtı 2."""
    r = boyut // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    kare = x ** 2 + y ** 2
    cekirdek = (kare - 2 * sigma ** 2) / sigma ** 4 * np.exp(-kare / (2 * sigma ** 2))

    # Kesilen kuyruklar yüzünden toplam tam 0 değil; düzeltiyoruz
    cekirdek -= cekirdek.mean()
    # x^2 resminde Laplacian = 2 olmalı: toplam K(x, y) * x^2 = 2
    cekirdek *= 2 / (cekirdek * x ** 2).sum()
    return cekirdek


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

        sonuc(y, x) = toplam  K(i, j) * resim(y - i, x - j)      i, j = -r..r

    LoG çekirdeği simetrik olduğundan konvolüsyondaki 180 derece çevirme
    sonucu değiştirmez.
    """
    yukseklik, genislik = resim.shape
    yaricap = len(cekirdek) // 2           # 7x7 için 3
    sonuc = np.zeros((yukseklik, genislik), dtype=np.float64)
    cekirdek = [[float(k) for k in satir] for satir in cekirdek]

    for y in range(yukseklik):
        for x in range(genislik):
            toplam = 0.0
            for i in range(-yaricap, yaricap + 1):
                yy = yansit(y - i, yukseklik)
                satir = cekirdek[i + yaricap]
                for j in range(-yaricap, yaricap + 1):
                    toplam += satir[j + yaricap] * int(resim[yy, yansit(x - j, genislik)])
            sonuc[y, x] = toplam
    return sonuc


def log_uygula(resim):
    return konvolusyon(resim, log_cekirdegi())


# ---------------------------------------------------------------------------
# 3) Sıfır geçişi ile kenar bulma
# ---------------------------------------------------------------------------
def sifir_gecisleri(yanit, esik=SIFIR_GECIS_ESIGI):
    """İkinci türevin işaret değiştirdiği yerleri kenar olarak işaretler.

    Her piksel sağ ve alt komşusuyla kıyaslanır. İşaretler zıtsa ve
    aradaki fark eşikten büyükse, sıfıra (kenara) daha yakın olan piksel
    işaretlenir; böylece kenar tek piksel kalınlığında olur.
    """
    kenar = np.zeros(yanit.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = yanit[:yanit.shape[0] - dy, :yanit.shape[1] - dx]
        b = yanit[dy:, dx:]
        gecis = (a * b < 0) & (np.abs(a - b) > esik)
        a_yakin = gecis & (np.abs(a) <= np.abs(b))
        b_yakin = gecis & (np.abs(a) > np.abs(b))
        kenar[:yanit.shape[0] - dy, :yanit.shape[1] - dx] |= a_yakin
        kenar[dy:, dx:] |= b_yakin
    return kenar


# ---------------------------------------------------------------------------
# 4) Sonuçları gösterme
# ---------------------------------------------------------------------------
def mutlak_goruntu(yanit, kazanc=6):
    """|LoG| değerini kazançla çarpıp 255'te kırparak gösterilebilir resme çevirir."""
    return np.clip(np.abs(yanit) * kazanc, 0, 255).astype(np.uint8)


def isaretli_goruntu(yanit, kazanc=3):
    """İşaretli LoG'u gri zemin üzerinde gösterir: 0 -> 128, negatif koyu, pozitif açık."""
    return np.clip(128 + yanit * kazanc, 0, 255).astype(np.uint8)


def kenar_goruntu(kenar):
    return np.where(kenar, 255, 0).astype(np.uint8)


def cekirdek_goruntu(cekirdek, boyut):
    """Çekirdeği büyütülmüş ısı haritası olarak çizer (pozitif kırmızı, negatif mavi)."""
    olcek = np.abs(cekirdek).max()
    renkli = np.zeros(cekirdek.shape + (3,), np.uint8)
    renkli[..., 2] = np.clip(cekirdek / olcek * 255, 0, 255)
    renkli[..., 0] = np.clip(-cekirdek / olcek * 255, 0, 255)
    renkli = cv2.resize(renkli, (boyut, boyut), interpolation=cv2.INTER_NEAREST)
    hucre = boyut / len(cekirdek)
    for i, satir in enumerate(cekirdek):
        for j, deger in enumerate(satir):
            cv2.putText(renkli, f"{deger:.2f}", (int(j * hucre + 6), int(i * hucre + hucre / 2 + 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return renkli


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

    # 1) Tek kanallı resmi OpenCV ile aç
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape, "| Kanal sayısı: 1")

    cekirdek = log_cekirdegi()
    print(f"{LOG_BOYUTU}x{LOG_BOYUTU} LoG çekirdeği (sigma = {LOG_SIGMA}), toplam = {cekirdek.sum():.1e}:")
    for satir in cekirdek:
        print("   " + " ".join(f"{d:6.3f}" for d in satir))

    # 2) LoG konvolüsyonu (yumuşatma çekirdeğin içinde)
    baslangic = time.perf_counter()
    yanit = log_uygula(resim)
    print(f"LoG konvolüsyonu süresi: {time.perf_counter() - baslangic:.2f} s")

    # Sadece doğrulama amaçlı: aynı çekirdekle cv2.filter2D
    hazir = cv2.filter2D(resim.astype(np.float64), -1, cekirdek)
    print(f"Kontrol -> cv2.filter2D ile en büyük fark: {np.abs(yanit - hazir).max():.1e}")

    # 3) Sıfır geçişleriyle kenarları bul
    kenar = sifir_gecisleri(yanit)
    kenar_esiksiz = sifir_gecisleri(yanit, esik=0)
    print(f"Ortalama |LoG|: {np.abs(yanit).mean():.2f} | en büyük |LoG|: {np.abs(yanit).max():.0f}")
    print(f"Sıfır geçişi kenar pikseli (eşik {SIFIR_GECIS_ESIGI}): {np.count_nonzero(kenar)} | "
          f"eşiksiz: {np.count_nonzero(kenar_esiksiz)}")

    kaydet(os.path.join(CIKTI_KLASORU, "log_sonuc.png"), mutlak_goruntu(yanit))
    kaydet(os.path.join(CIKTI_KLASORU, "log_kenarlar.png"), kenar_goruntu(kenar))

    ust = np.hstack([etiket_ekle(resim, "Orijinal"),
                     etiket_ekle(cekirdek_goruntu(cekirdek, resim.shape[0]), f"LoG {LOG_BOYUTU}x{LOG_BOYUTU}, sigma={LOG_SIGMA}"),
                     etiket_ekle(isaretli_goruntu(yanit), "LoG (isaretli)")])
    alt = np.hstack([etiket_ekle(kenar_goruntu(kenar_esiksiz), "Sifir gecisi (esiksiz)"),
                     etiket_ekle(kenar_goruntu(kenar), f"Sifir gecisi > {SIFIR_GECIS_ESIGI}"),
                     etiket_ekle(np.where(kenar[..., None], [0, 0, 255], cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR))
                                 .astype(np.uint8), "Kenarlar resim uzerinde")])
    karsilastirma = np.vstack([ust, alt])
    kaydet(os.path.join(CIKTI_KLASORU, "log_karsilastirma.png"), karsilastirma)

    pencere = f"{LOG_BOYUTU}x{LOG_BOYUTU} LoG filtresi"
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.imshow(pencere, karsilastirma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
