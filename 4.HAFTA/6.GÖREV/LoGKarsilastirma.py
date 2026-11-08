"""
Görüntü İşleme Ödevi - 7x7 LoG ile Diğer Kenar Filtrelerinin Karşılaştırılması
------------------------------------------------------------------------------
Yedi kenar bulucu aynı deneylerle kıyaslanır:

  Prewitt, Sobel       : birinci türev -> |G| = sqrt(Gx^2 + Gy^2) eşiklenir
  Laplacian (3x3)      : ikinci türev  -> sıfır geçişleri (eşikli)
  LoG (7x7, sigma = 1) : ikinci türev  -> sıfır geçişleri (eşikli)

İlk üçü hem doğrudan ham resme, hem de 3x3 Gauss ile yumuşatılmış resme
uygulanır. LoG yumuşatmayı kendi içinde yaptığı için yalnızca ham resme
uygulanır.

Adil eşik: Operatörlerin çıktıları farklı birimlerde olduğu için hepsine aynı
sayı verilemez. Her operatörün eşiği, 150 gri seviyelik temiz bir disk
kenarındaki kendi karar büyüklüğünün (gradyanda |G|, sıfır geçişinde iki
komşu arasındaki sıçrama) ESIK_ORANI (%20) katı olarak belirlenir.
Böylece hepsi "bu kontrasttaki bir kenarın %20'sinden zayıf olanı yok say"
kuralıyla çalışır.

Yapılan deneyler:
  1) Cameraman resmi: kenar piksel sayısı
  2) Konum ve kalınlık: yarıçapı bilinen yapay bir diskte kenarın ne kadar
     kalın olduğu ve gerçek kenardan ne kadar uzakta olduğu
  3) Gürültü: diske ve cameraman'e Gauss gürültüsü (sigma = 10) eklenince
     sinyal/gürültü oranı ve ortaya çıkan sahte kenar sayısı

Konvolüsyonlar LoGFiltresi.py'deki elle yazılmış fonksiyonla yapılır.
Sonuçların yorumu README.md dosyasındadır.

Kullanım:
    python LoGKarsilastirma.py [resim_yolu]
"""

import os
import sys

import cv2
import numpy as np

import LoGFiltresi as log_modul

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = log_modul.KLASOR
CIKTI_KLASORU = log_modul.CIKTI_KLASORU
KAYNAK_RESIM = log_modul.KAYNAK_RESIM

konvolusyon = log_modul.konvolusyon
sifir_gecisleri = log_modul.sifir_gecisleri
gri_oku = log_modul.gri_oku
kaydet = log_modul.kaydet
etiket_ekle = log_modul.etiket_ekle
kenar_goruntu = log_modul.kenar_goruntu

GAUSS = [[1 / 16, 2 / 16, 1 / 16],
         [2 / 16, 4 / 16, 2 / 16],
         [1 / 16, 2 / 16, 1 / 16]]
PREWITT_X = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
PREWITT_Y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
LAPLACIAN = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

ESIK_ORANI = 0.20
GURULTU_SIGMA = 10


# ---------------------------------------------------------------------------
# Operatörler: her biri resimden "yanıt" üretir
# ---------------------------------------------------------------------------
def gauss_yumusat(resim):
    return np.clip(np.floor(konvolusyon(resim, GAUSS) + 0.5), 0, 255).astype(np.uint8)


def gradyan(cx, cy, bolen):
    def f(resim):
        return np.sqrt((konvolusyon(resim, cx) / bolen) ** 2 + (konvolusyon(resim, cy) / bolen) ** 2)
    return f


def laplacian(resim):
    return konvolusyon(resim, LAPLACIAN)


def gauss_ile(f):
    return lambda resim: f(gauss_yumusat(resim))


# (ad, yanıt fonksiyonu, tür) — tür: "gradyan" eşiklenir, "ikinci" sıfır geçişi
OPERATORLER = (
    ("Prewitt", gradyan(PREWITT_X, PREWITT_Y, 6), "gradyan"),
    ("Prewitt + Gauss", gauss_ile(gradyan(PREWITT_X, PREWITT_Y, 6)), "gradyan"),
    ("Sobel", gradyan(SOBEL_X, SOBEL_Y, 8), "gradyan"),
    ("Sobel + Gauss", gauss_ile(gradyan(SOBEL_X, SOBEL_Y, 8)), "gradyan"),
    ("Laplacian", laplacian, "ikinci"),
    ("Laplacian + Gauss", gauss_ile(laplacian), "ikinci"),
    ("LoG 7x7", log_modul.log_uygula, "ikinci"),
)


def kenar_bul(yanit, tur, esik):
    if tur == "gradyan":
        return yanit > esik
    return sifir_gecisleri(yanit, esik)


def karar_degerleri(yanit, tur, bolge):
    """Kenar kararında eşikle kıyaslanan büyüklükler (bölge içinde).

    Gradyan operatörlerinde bu |G|'dir. Sıfır geçişinde ise eşik, işaret
    değiştiren iki komşu arasındaki sıçramayla kıyaslanır; o sıçramalar döner.
    """
    if tur == "gradyan":
        return yanit[bolge]
    sicramalar = []
    for dy, dx in ((0, 1), (1, 0)):
        a = yanit[:yanit.shape[0] - dy, :yanit.shape[1] - dx]
        b = yanit[dy:, dx:]
        gecis = (a * b < 0) & bolge[:yanit.shape[0] - dy, :yanit.shape[1] - dx]
        sicramalar.append(np.abs(a - b)[gecis])
    return np.concatenate(sicramalar)


def genislet(kenar):
    """Kenar haritasını 1 piksel kalınlaştırır (3x3 komşulukta herhangi biri kenarsa)."""
    dolgulu = np.pad(kenar, 1)
    sonuc = np.zeros_like(kenar)
    for dy in range(3):
        for dx in range(3):
            sonuc |= dolgulu[dy:dy + kenar.shape[0], dx:dx + kenar.shape[1]]
    return sonuc


# ---------------------------------------------------------------------------
# Test resmi: disk
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else KAYNAK_RESIM
    resim = gri_oku(yol)
    if resim is None:
        sys.exit(f"Resim açılamadı: {yol}")
    print("Resim:", yol, "| Boyut:", resim.shape)

    disk, uzaklik, yaricap = disk_resmi()
    halka = np.abs(uzaklik - yaricap) < 3      # kenarın çevresi
    duz_bolge = uzaklik < yaricap - 20         # diskin içi, kenardan uzak
    cevre = 2 * np.pi * yaricap

    uretec = np.random.default_rng(0)
    gurultulu_disk = np.clip(disk + uretec.normal(0, GURULTU_SIGMA, disk.shape), 0, 255).astype(np.uint8)
    gurultulu_cam = np.clip(resim + uretec.normal(0, GURULTU_SIGMA, resim.shape), 0, 255).astype(np.uint8)

    sonuclar = {}
    print(f"\n{'Operatör':<18} {'eşik':>6} | {'kenar px':>8} | {'kalınlık':>8} | {'|r-100|':>7} | "
          f"{'S/G':>5} | {'disk içi sahte':>14} | {'cameraman yeni sahte':>20}")
    for ad, f, tur in OPERATORLER:
        # Adil eşik: temiz disk kenarındaki karar büyüklüğünün (|G| ya da
        # sıfır geçişindeki sıçrama) tepe değerinin %20'si
        disk_yanit = f(disk)
        referans = np.percentile(karar_degerleri(disk_yanit, tur, halka), 95)
        esik = ESIK_ORANI * referans

        # 1) Cameraman
        cam_kenar = kenar_bul(f(resim), tur, esik)

        # 2) Disk: kalınlık ve konum
        disk_kenar = kenar_bul(disk_yanit, tur, esik)
        d = np.abs(uzaklik[disk_kenar] - yaricap)

        # 3) Gürültü
        kirli_disk_yanit = f(gurultulu_disk)
        sinyal = np.abs(disk_yanit[halka]).max()
        gurultu = (kirli_disk_yanit - disk_yanit)[duz_bolge].std()
        sahte_disk = np.count_nonzero(kenar_bul(kirli_disk_yanit, tur, esik) & duz_bolge)
        kirli_cam_kenar = kenar_bul(f(gurultulu_cam), tur, esik)
        sahte_cam = np.count_nonzero(kirli_cam_kenar & ~genislet(cam_kenar))

        print(f"{ad:<18} {esik:6.2f} | {np.count_nonzero(cam_kenar):8d} | "
              f"{np.count_nonzero(disk_kenar) / cevre:6.2f}px | {d.mean():5.2f}px | "
              f"{sinyal / gurultu:5.1f} | {sahte_disk:14d} | {sahte_cam:20d}")
        sonuclar[ad] = (cam_kenar, kirli_cam_kenar)

    # Görseller: üstte temiz, altta gürültülü kenar haritaları (orijinal + 7 operatör)
    def izgara(indeks, ilk, ilk_ad):
        karolar = [etiket_ekle(ilk, ilk_ad)] + [etiket_ekle(kenar_goruntu(s[indeks]), ad)
                                                for ad, s in sonuclar.items()]
        return np.vstack([np.hstack(karolar[:4]), np.hstack(karolar[4:])])

    kaydet(os.path.join(CIKTI_KLASORU, "tum_operatorler.png"), izgara(0, resim, "Orijinal"))
    kaydet(os.path.join(CIKTI_KLASORU, "tum_operatorler_gurultulu.png"),
           izgara(1, gurultulu_cam, f"Gurultulu (s={GURULTU_SIGMA})"))

    # Detay: tripod ve çimen bölgesi, üç ikinci türev operatörü + Sobel + Gauss
    y0, y1, x0, x1 = 280, 440, 200, 360
    detay = [cv2.resize(kenar_goruntu(sonuclar[ad][0])[y0:y1, x0:x1], None, fx=2, fy=2,
                        interpolation=cv2.INTER_NEAREST)
             for ad in ("Sobel + Gauss", "Laplacian", "Laplacian + Gauss", "LoG 7x7")]
    kaydet(os.path.join(CIKTI_KLASORU, "detay_karsilastirma.png"),
           np.hstack([etiket_ekle(d, ad) for d, ad in
                      zip(detay, ("Sobel+Gauss", "Laplacian", "Laplacian+Gauss", "LoG 7x7"))]))


if __name__ == "__main__":
    main()
