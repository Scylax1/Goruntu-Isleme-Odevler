"""
Görüntü İşleme Ödevi - Prewitt ve Sobel Operatörlerinin Karşılaştırılması
-------------------------------------------------------------------------
Aynı Gauss ile yumuşatılmış resme 3x3 Prewitt ve 3x3 Sobel filtreleri
(her ikisi de x ve y yönünde) uygulanır ve sonuçlar sayısal olarak kıyaslanır.

Adil kıyas için çekirdekler normalize edilir:
    Prewitt / 6, Sobel / 8
Böylece 0'dan 1'e birim eğimli bir rampada ikisi de türev = 1 verir;
kalan farklar yalnızca çekirdeklerin şeklinden gelir.

Yapılan üç deney:
  1) Cameraman resmi: kenar haritalarının benzerliği ve kenar piksel sayısı
  2) Gürültü testi: resme Gauss gürültüsü (sigma = 10) eklenince kenar
     haritasının ne kadar bozulduğu
  3) Yön testi: yapay bir disk resminin kenarında, operatörün bulduğu
     gradyan yönünün gerçek kenar normalinden kaç derece saptığı

Konvolüsyon, 3. görevdeki elle yazılmış fonksiyonla yapılır.
Sonuçların yorumu README.md dosyasındadır.

Kullanım:
    python PrewittSobelKarsilastirma.py [resim_yolu]
"""

import importlib.util
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
KAYNAK_RESIM = os.path.join(KLASOR, "..", "..", "2.HAFTA", "1.GÖREV", "ciktilar", "cameraman.png")

# 3. görevdeki konvolüsyon ve yardımcı fonksiyonları kullan
# (klasör adında nokta olduğu için normal import ile alınamıyor)
_spec = importlib.util.spec_from_file_location(
    "SobelFiltresi", os.path.join(KLASOR, "..", "3.GÖREV", "SobelFiltresi.py"))
sobel_modul = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sobel_modul)

konvolusyon = sobel_modul.konvolusyon
gauss_yumusat = sobel_modul.gauss_yumusat
gri_oku = sobel_modul.gri_oku
kaydet = sobel_modul.kaydet
etiket_ekle = sobel_modul.etiket_ekle

PREWITT_X = [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
PREWITT_Y = [[-1, -1, -1],
             [ 0,  0,  0],
             [ 1,  1,  1]]
SOBEL_X = sobel_modul.SOBEL_X
SOBEL_Y = sobel_modul.SOBEL_Y

# Normalize edilmiş büyüklükte bu değeri aşan piksel "kenar" sayılır
KENAR_ESIGI = 12
GURULTU_SIGMA = 10


# ---------------------------------------------------------------------------
# Operatörler
# ---------------------------------------------------------------------------
def prewitt_buyukluk(resim):
    gx = konvolusyon(resim, PREWITT_X) / 6
    gy = konvolusyon(resim, PREWITT_Y) / 6
    return np.sqrt(gx ** 2 + gy ** 2)


def sobel_buyukluk(resim):
    gx = konvolusyon(resim, SOBEL_X) / 8
    gy = konvolusyon(resim, SOBEL_Y) / 8
    return np.sqrt(gx ** 2 + gy ** 2)


def gosterilebilir(buyukluk, kazanc=4):
    """Normalize büyüklük küçük olduğu için kazançla çarpıp 255'te kırpar."""
    return np.clip(buyukluk * kazanc, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Deneyler
# ---------------------------------------------------------------------------
def deney_cameraman(yumusak):
    gp = prewitt_buyukluk(yumusak)
    gs = sobel_buyukluk(yumusak)
    kenar_p = gp > KENAR_ESIGI
    kenar_s = gs > KENAR_ESIGI
    ortak = np.count_nonzero(kenar_p & kenar_s)
    birlesim = np.count_nonzero(kenar_p | kenar_s)

    print("1) Cameraman (Gauss sonrası, normalize)")
    print(f"   Prewitt -> ortalama |G|: {gp.mean():6.2f} | en büyük: {gp.max():6.1f} | "
          f"kenar pikseli (> {KENAR_ESIGI}): {np.count_nonzero(kenar_p)}")
    print(f"   Sobel   -> ortalama |G|: {gs.mean():6.2f} | en büyük: {gs.max():6.1f} | "
          f"kenar pikseli (> {KENAR_ESIGI}): {np.count_nonzero(kenar_s)}")
    print(f"   İki büyüklük haritası arası korelasyon: {np.corrcoef(gp.ravel(), gs.ravel())[0, 1]:.4f}")
    print(f"   Kenar haritası örtüşmesi (ortak / birleşim): {ortak / birlesim:.3f}")
    print(f"   Ortalama |Sobel - Prewitt|: {np.abs(gs - gp).mean():.3f} | "
          f"en büyük: {np.abs(gs - gp).max():.1f}")
    return gp, gs


def deney_gurultu(resim, gp_temiz, gs_temiz):
    uretec = np.random.default_rng(0)
    gurultulu = np.clip(resim + uretec.normal(0, GURULTU_SIGMA, resim.shape), 0, 255).astype(np.uint8)
    yumusak = gauss_yumusat(gurultulu)
    gp = prewitt_buyukluk(yumusak)
    gs = sobel_buyukluk(yumusak)

    def rapor(ad, temiz, kirli):
        hata = np.sqrt(np.mean((kirli - temiz) ** 2))
        sahte = np.count_nonzero((kirli > KENAR_ESIGI) & ~(temiz > KENAR_ESIGI))
        print(f"   {ad:<7} -> |G| değişiminin RMS'i: {hata:5.2f} | yeni çıkan sahte kenar pikseli: {sahte}")

    print(f"2) Gürültü testi (Gauss gürültüsü, sigma = {GURULTU_SIGMA})")
    rapor("Prewitt", gp_temiz, gp)
    rapor("Sobel", gs_temiz, gs)
    return yumusak, gp, gs


def deney_yon():
    """Merkezli bir disk çizer; her açıdaki kenar boyunca en büyük |G|'yi ölçer."""
    boyut, yaricap = 301, 100
    merkez = boyut // 2
    # Kenarı pürüzsüz yapmak için 4 kat büyük çizip küçültüyoruz (antialias)
    buyuk = np.full((boyut * 4, boyut * 4), 50, np.uint8)
    cv2.circle(buyuk, (merkez * 4 + 2, merkez * 4 + 2), yaricap * 4, 200, -1)
    disk = cv2.resize(buyuk, (boyut, boyut), interpolation=cv2.INTER_AREA)
    yumusak = gauss_yumusat(disk)
    gp = prewitt_buyukluk(yumusak)
    gs = sobel_buyukluk(yumusak)

    # Diskin her kenar pikselinde gerçek kenar normali merkezden dışarı bakar.
    # Operatörün bulduğu gradyan yönü atan2(gy, gx) bununla kıyaslanır.
    # Gerçek merkez, küçültmedeki yarım piksel kaymasını tahmin etmemek için
    # parlaklığın ağırlık merkezinden bulunur.
    yy, xx = np.mgrid[:boyut, :boyut]
    agirlik = disk.astype(np.float64) - 50
    my, mx = (agirlik * yy).sum() / agirlik.sum(), (agirlik * xx).sum() / agirlik.sum()
    gercek_aci = np.arctan2(yy - my, xx - mx)

    print("3) Yön testi (disk kenarında gradyan yönü hatası)")
    for ad, cx, cy in (("Prewitt", PREWITT_X, PREWITT_Y), ("Sobel", SOBEL_X, SOBEL_Y)):
        # konvolusyon çekirdeği çevirdiği için işaret ters; -1 ile düzeltiyoruz
        gx = -konvolusyon(yumusak, cx)
        gy = -konvolusyon(yumusak, cy)
        g = np.sqrt(gx ** 2 + gy ** 2)
        kenar = g > g.max() / 2
        # disk içi parlak olduğu için gradyan içe bakar; normal ile kıyas için 180° çevir
        hata = np.angle(np.exp(1j * (np.arctan2(-gy, -gx) - gercek_aci)))[kenar]
        hata = np.rad2deg(np.abs(hata))
        print(f"   {ad:<7} -> ortalama açı hatası: {hata.mean():5.2f}° | en büyük: {hata.max():5.2f}° "
              f"| kenar pikseli: {np.count_nonzero(kenar)}")
    return disk, gp, gs


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape)

    yumusak = gauss_yumusat(resim)
    gp, gs = deney_cameraman(yumusak)
    _, gp_g, gs_g = deney_gurultu(resim, gp, gs)
    disk, gp_d, gs_d = deney_yon()

    fark = np.clip(np.abs(gs - gp) * 16, 0, 255).astype(np.uint8)
    ust = np.hstack([etiket_ekle(gosterilebilir(gp), "Prewitt |G|"),
                     etiket_ekle(gosterilebilir(gs), "Sobel |G|"),
                     etiket_ekle(fark, "|Sobel - Prewitt| x16")])
    alt = np.hstack([etiket_ekle(gosterilebilir(gp_g), f"Prewitt, gurultu s={GURULTU_SIGMA}"),
                     etiket_ekle(gosterilebilir(gs_g), f"Sobel, gurultu s={GURULTU_SIGMA}"),
                     etiket_ekle(np.dstack([np.where(gs > KENAR_ESIGI, 255, 0),
                                            np.where(gp > KENAR_ESIGI, 255, 0),
                                            np.where(gs > KENAR_ESIGI, 255, 0)]).astype(np.uint8),
                                 "kenar: beyaz=ikisi, yesil=P, mor=S")])
    kaydet(os.path.join(CIKTI_KLASORU, "prewitt_sobel_karsilastirma.png"), np.vstack([ust, alt]))

    disk_tablo = np.hstack([etiket_ekle(disk, "Disk"),
                            etiket_ekle(gosterilebilir(gp_d, 3), "Prewitt"),
                            etiket_ekle(gosterilebilir(gs_d, 3), "Sobel")])
    kaydet(os.path.join(CIKTI_KLASORU, "yon_testi.png"), disk_tablo)


if __name__ == "__main__":
    main()
