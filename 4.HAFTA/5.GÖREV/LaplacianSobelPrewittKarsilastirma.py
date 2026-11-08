"""
Görüntü İşleme Ödevi - Laplacian, Sobel ve Prewitt Kenar Bulucularının Karşılaştırılması
---------------------------------------------------------------------------------------
Üç operatör de aynı, Gauss ile yumuşatılmış resme uygulanır:

  Prewitt, Sobel : birinci türev -> |G| = sqrt(Gx^2 + Gy^2) eşiklenir
  Laplacian      : ikinci türev  -> sıfır geçişleri (eşikli) kenar sayılır

Yapılan üç deney:
  1) Cameraman resmi: kenar piksel sayısı ve kenar haritalarının örtüşmesi
  2) Konum ve kalınlık: yarıçapı bilinen yapay bir diskte bulunan kenarın
     ne kadar kalın olduğu ve gerçek kenardan ne kadar uzakta olduğu
  3) Gürültü: diske ve cameraman'e Gauss gürültüsü (sigma = 10) eklenince
     sinyal/gürültü oranı ve ortaya çıkan sahte kenar sayısı

Konvolüsyonlar elle yazılmış fonksiyonla yapılır (bu görevdeki
LaplacianFiltresi.py ve 3. görevdeki SobelFiltresi.py'den alınır).
Sonuçların yorumu README.md dosyasındadır.

Kullanım:
    python LaplacianSobelPrewittKarsilastirma.py [resim_yolu]
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


def modul_yukle(ad, yol):
    # klasör adlarında nokta olduğu için normal import ile alınamıyor
    spec = importlib.util.spec_from_file_location(ad, yol)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


lap_modul = modul_yukle("LaplacianFiltresi", os.path.join(KLASOR, "LaplacianFiltresi.py"))
sobel_modul = modul_yukle("SobelFiltresi", os.path.join(KLASOR, "..", "3.GÖREV", "SobelFiltresi.py"))

konvolusyon = lap_modul.konvolusyon
gauss_yumusat = lap_modul.gauss_yumusat
laplacian = lap_modul.laplacian
sifir_gecisleri = lap_modul.sifir_gecisleri
gri_oku = lap_modul.gri_oku
kaydet = lap_modul.kaydet
etiket_ekle = lap_modul.etiket_ekle

PREWITT_X = [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
PREWITT_Y = [[-1, -1, -1],
             [ 0,  0,  0],
             [ 1,  1,  1]]

# Normalize |G| (Prewitt/6, Sobel/8) için kenar eşiği; 4. görevle aynı
GRADYAN_ESIGI = 12
SIFIR_GECIS_ESIGI = lap_modul.SIFIR_GECIS_ESIGI
GURULTU_SIGMA = 10


# ---------------------------------------------------------------------------
# Operatörler: her biri (yanıt, kenar haritası) döndürür
# ---------------------------------------------------------------------------
def prewitt(resim):
    g = np.sqrt((konvolusyon(resim, PREWITT_X) / 6) ** 2 + (konvolusyon(resim, PREWITT_Y) / 6) ** 2)
    return g, g > GRADYAN_ESIGI


def sobel(resim):
    g = np.sqrt((konvolusyon(resim, sobel_modul.SOBEL_X) / 8) ** 2 +
                (konvolusyon(resim, sobel_modul.SOBEL_Y) / 8) ** 2)
    return g, g > GRADYAN_ESIGI


def laplacian_kenar(resim):
    lap = laplacian(resim)
    return lap, sifir_gecisleri(lap)


OPERATORLER = (("Prewitt", prewitt), ("Sobel", sobel), ("Laplacian", laplacian_kenar))


def genislet(kenar):
    """Kenar haritasını 1 piksel kalınlaştırır (3x3 komşulukta herhangi biri kenarsa)."""
    dolgulu = np.pad(kenar, 1)
    sonuc = np.zeros_like(kenar)
    for dy in range(3):
        for dx in range(3):
            sonuc |= dolgulu[dy:dy + kenar.shape[0], dx:dx + kenar.shape[1]]
    return sonuc


# ---------------------------------------------------------------------------
# Deneyler
# ---------------------------------------------------------------------------
def deney_cameraman(yumusak):
    print("1) Cameraman (Gauss sonrası)")
    sonuclar = {ad: f(yumusak) for ad, f in OPERATORLER}
    for ad, (_, kenar) in sonuclar.items():
        print(f"   {ad:<9} -> kenar pikseli: {np.count_nonzero(kenar)}")

    sobel_kenar = sonuclar["Sobel"][1]
    lap_kenar = sonuclar["Laplacian"][1]
    icinde = np.count_nonzero(lap_kenar & genislet(sobel_kenar)) / np.count_nonzero(lap_kenar)
    print(f"   Laplacian kenarlarının Sobel kenarına en fazla 1 piksel uzak olanı: %{icinde * 100:.1f}")
    # Sobel kenar bölgelerinin içinden geçen Laplacian çizgisi var mı?
    kapsanan = np.count_nonzero(sobel_kenar & genislet(genislet(lap_kenar))) / np.count_nonzero(sobel_kenar)
    print(f"   Sobel kenar piksellerinin bir Laplacian kenarına en fazla 2 piksel uzak olanı: %{kapsanan * 100:.1f}")
    return sonuclar


def disk_resmi(boyut=301, yaricap=100):
    # Kenarı pürüzsüz yapmak için 4 kat büyük çizip küçültüyoruz (antialias)
    merkez = boyut // 2
    buyuk = np.full((boyut * 4, boyut * 4), 50, np.uint8)
    cv2.circle(buyuk, (merkez * 4 + 2, merkez * 4 + 2), yaricap * 4, 200, -1)
    disk = cv2.resize(buyuk, (boyut, boyut), interpolation=cv2.INTER_AREA)
    # gerçek merkez, parlaklığın ağırlık merkezinden
    yy, xx = np.mgrid[:boyut, :boyut]
    agirlik = disk.astype(np.float64) - 50
    my, mx = (agirlik * yy).sum() / agirlik.sum(), (agirlik * xx).sum() / agirlik.sum()
    uzaklik = np.sqrt((yy - my) ** 2 + (xx - mx) ** 2)
    return disk, uzaklik, yaricap


def deney_disk():
    disk, uzaklik, yaricap = disk_resmi()
    yumusak = gauss_yumusat(disk)
    cevre = 2 * np.pi * yaricap

    print("2) Konum ve kalınlık (disk, yarıçap 100, kontrast 150)")
    sonuclar = {}
    for ad, f in OPERATORLER:
        yanit, kenar = f(yumusak)
        d = uzaklik[kenar]
        print(f"   {ad:<9} -> kenar kalınlığı (piksel / çevre): {np.count_nonzero(kenar) / cevre:4.2f} px | "
              f"ortalama |r - 100|: {np.abs(d - yaricap).mean():4.2f} px | "
              f"en uzak: {np.abs(d - yaricap).max():4.2f} px")
        sonuclar[ad] = (yanit, kenar)
    return disk, yumusak, uzaklik, yaricap, sonuclar


def deney_gurultu(resim, temiz_cam, disk, uzaklik, yaricap, temiz_disk):
    uretec = np.random.default_rng(0)

    def gurultu_ekle(r):
        return gauss_yumusat(np.clip(r + uretec.normal(0, GURULTU_SIGMA, r.shape), 0, 255).astype(np.uint8))

    gurultulu_disk = gurultu_ekle(disk)
    gurultulu_cam = gurultu_ekle(resim)
    duz_bolge = uzaklik < yaricap - 20        # diskin içi, kenardan uzak

    print(f"3) Gürültü (Gauss gürültüsü, sigma = {GURULTU_SIGMA})")
    sonuclar = {}
    for ad, f in OPERATORLER:
        temiz_yanit = temiz_disk[ad][0]
        kirli_yanit, kirli_kenar = f(gurultulu_disk)
        # sinyal: kenar boyunca yanıtın tepe değeri; gürültü: düz bölgede yanıtın sapması
        halka = np.abs(uzaklik - yaricap) < 3
        sinyal = np.abs(temiz_yanit[halka]).max()
        gurultu = (kirli_yanit - temiz_yanit)[duz_bolge].std()
        sahte_disk = np.count_nonzero(kirli_kenar & duz_bolge)

        _, kirli_cam_kenar = f(gurultulu_cam)
        temiz_cam_kenar = temiz_cam[ad][1]
        sahte_cam = np.count_nonzero(kirli_cam_kenar & ~genislet(temiz_cam_kenar))
        print(f"   {ad:<9} -> disk S/G: {sinyal / gurultu:5.1f} | diskin düz içinde sahte kenar: {sahte_disk:5d} | "
              f"cameraman'de yeni sahte kenar: {sahte_cam}")
        sonuclar[ad] = kirli_cam_kenar
    return sonuclar


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def kenar_goruntu(kenar):
    return np.where(kenar, 255, 0).astype(np.uint8)


def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape)

    yumusak = gauss_yumusat(resim)
    cam = deney_cameraman(yumusak)
    disk, _, uzaklik, yaricap, disk_sonuc = deney_disk()
    gurultulu = deney_gurultu(resim, cam, disk, uzaklik, yaricap, disk_sonuc)

    # Cameraman: üstte temiz, altta gürültülü kenar haritaları
    ust = np.hstack([etiket_ekle(kenar_goruntu(cam[ad][1]), f"{ad}") for ad, _ in OPERATORLER])
    alt = np.hstack([etiket_ekle(kenar_goruntu(gurultulu[ad]), f"{ad}, gurultu s={GURULTU_SIGMA}")
                     for ad, _ in OPERATORLER])
    kaydet(os.path.join(CIKTI_KLASORU, "uc_operator_karsilastirma.png"), np.vstack([ust, alt]))

    # Detay: tripod bölgesinde Sobel kenar bölgesi (gri) ve Laplacian çizgisi (kırmızı)
    y0, y1, x0, x1 = 280, 440, 200, 360
    renkli = cv2.cvtColor(yumusak[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR) // 3
    renkli[cam["Sobel"][1][y0:y1, x0:x1]] = (160, 160, 160)
    renkli[cam["Laplacian"][1][y0:y1, x0:x1]] = (0, 0, 255)
    renkli = cv2.resize(renkli, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    kaydet(os.path.join(CIKTI_KLASORU, "kenar_kalinligi_detay.png"),
           etiket_ekle(renkli, "gri: Sobel, kirmizi: Laplacian"))

    # Disk: üç operatörün bulduğu kenar (yakınlaştırılmış bir dilim)
    dilim = [cv2.resize(kenar_goruntu(disk_sonuc[ad][1])[30:90, 120:180], None, fx=5, fy=5,
                        interpolation=cv2.INTER_NEAREST) for ad, _ in OPERATORLER]
    kaydet(os.path.join(CIKTI_KLASORU, "disk_kenar_detay.png"),
           np.hstack([etiket_ekle(d, ad) for d, (ad, _) in zip(dilim, OPERATORLER)]))


if __name__ == "__main__":
    main()
