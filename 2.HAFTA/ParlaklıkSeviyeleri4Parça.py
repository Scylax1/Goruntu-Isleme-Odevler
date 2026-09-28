"""
Görüntü İşleme Ödevi - Parlaklık (Gri Seviye) Çözünürlüğü (4 Parça)
--------------------------------------------------------------------
ParlaklikSeviyeleri.py'deki deneylerin aynısı, ama bu sefer fotoğraf 2x2 olarak
4 parçaya bölünür ve her parçaya farklı bir ayar uygulanır. Parçalar tekrar
birleştirildiği için farklar TEK bir fotoğrafın içinde yan yana görünür.

    +-------------+-------------+
    |  Sol üst    |  Sağ üst    |
    +-------------+-------------+
    |  Sol alt    |  Sağ alt    |
    +-------------+-------------+

  1) Nicemleme: her parçada farklı sayıda gri ton (256, 64, 16, 4 / 128, 32, 8, 2)
  2) Maksimum parlaklık: her parçada farklı en parlak değer (255, 128, 64, 32)
  3) Satır satır / sütun sütun gezinme hızı: 4 parçalı nicemleme her iki
     sırayla yapılıp süreleri karşılaştırılır.

Kullanım:
    python ParlaklıkSeviyeleri4Parça.py

Herhangi bir tuşa basınca bir sonraki pencereye geçer.
"""

import os
import time
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
# Görüntü işlemede klasik test resmi "cameraman" (scikit-image deposundan)
RESIM_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/camera.png"

KLASOR = os.path.dirname(os.path.abspath(__file__))
INEN_RESIM = os.path.join(KLASOR, "cameraman.png")
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

# Her liste 4 eleman: [sol üst, sağ üst, sol alt, sağ alt]
SEVIYE_GRUPLARI = [[256, 64, 16, 4], [128, 32, 8, 2]]
MAKS_PARLAKLIKLAR = [255, 128, 64, 32]


# ---------------------------------------------------------------------------
# 1) Fotoğrafı internetten indirme
# ---------------------------------------------------------------------------
def resmi_indir():
    """Resim daha önce indirildiyse tekrar indirmez."""
    if not os.path.exists(INEN_RESIM):
        print("Resim indiriliyor:", RESIM_URL)
        istek = urllib.request.Request(RESIM_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(istek, timeout=30) as cevap:
            veri = cevap.read()
        with open(INEN_RESIM, "wb") as f:
            f.write(veri)

    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(INEN_RESIM, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# 2) Fotoğrafı 4 parçaya bölme
# ---------------------------------------------------------------------------
def parca_sinirlari(resim):
    """4 parçanın (y1, y2, x1, x2) sınırlarını döndürür.

    Sıra: sol üst, sağ üst, sol alt, sağ alt.
    """
    yukseklik, genislik = resim.shape[:2]
    orta_y, orta_x = yukseklik // 2, genislik // 2
    return [
        (0, orta_y, 0, orta_x),
        (0, orta_y, orta_x, genislik),
        (orta_y, yukseklik, 0, orta_x),
        (orta_y, yukseklik, orta_x, genislik),
    ]


def parcalara_uygula(resim, islem, degerler):
    """Her parçaya kendi değeriyle 'islem' fonksiyonunu uygular ve birleştirir."""
    sonuc = np.zeros_like(resim)
    for (y1, y2, x1, x2), deger in zip(parca_sinirlari(resim), degerler):
        sonuc[y1:y2, x1:x2] = islem(resim[y1:y2, x1:x2], deger)
    return sonuc


# ---------------------------------------------------------------------------
# 3) Deneyler
# ---------------------------------------------------------------------------
def nicemle(resim, seviye_sayisi):
    """256 tonu 'seviye_sayisi' kadar tona indirir.

    Örnek: 64 seviyede adım = 4 olur; 200, 201, 202, 203 değerlerinin hepsi 200'e döner.
    """
    adim = 256 // seviye_sayisi
    return (resim // adim) * adim


def maks_parlakligi_degistir(resim, maks):
    """0-255 aralığını 0-maks aralığına sıkıştırır (resim kararır)."""
    return (resim.astype(np.float32) * (maks / 255.0)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Deney 3: Satır satır mı, sütun sütun mu gezmek daha hızlı?
# ---------------------------------------------------------------------------
# NumPy resmi bellekte satır satır (C sırası) tutar: [0,0], [0,1], [0,2], ...
# Satır satır gezersek bellekte yan yana duran piksellere sırayla gideriz,
# sütun sütun gezersek her adımda bir satır boyu (genişlik kadar byte) atlarız.
# Burada her parça kendi seviyesiyle, piksel piksel nicemlenir.
def nicemle_4parca_satir_satir(resim, seviyeler):
    sonuc = np.zeros_like(resim)
    for (y1, y2, x1, x2), seviye in zip(parca_sinirlari(resim), seviyeler):
        adim = 256 // seviye
        for y in range(y1, y2):             # dış döngü: satırlar
            for x in range(x1, x2):         # iç döngü: sütunlar
                sonuc[y, x] = (resim[y, x] // adim) * adim
    return sonuc


def nicemle_4parca_sutun_sutun(resim, seviyeler):
    sonuc = np.zeros_like(resim)
    for (y1, y2, x1, x2), seviye in zip(parca_sinirlari(resim), seviyeler):
        adim = 256 // seviye
        for x in range(x1, x2):             # dış döngü: sütunlar
            for y in range(y1, y2):         # iç döngü: satırlar
                sonuc[y, x] = (resim[y, x] // adim) * adim
    return sonuc


def sure_olc(fonksiyon, *argumanlar, tekrar=3):
    """Fonksiyonu birkaç kez çalıştırıp en kısa süreyi (saniye) döndürür."""
    sureler = []
    for _ in range(tekrar):
        baslangic = time.perf_counter()
        fonksiyon(*argumanlar)
        sureler.append(time.perf_counter() - baslangic)
    return min(sureler)


def gezinme_sirasi_deneyi(resim):
    print("\n--- Deney 3: Satır satır vs sütun sütun gezinme (4 parça) ---")
    seviyeler = SEVIYE_GRUPLARI[0]

    # a) Piksel piksel Python döngüsü (ödevdeki klasik iç içe for yapısı)
    satir_suresi = sure_olc(nicemle_4parca_satir_satir, resim, seviyeler)
    sutun_suresi = sure_olc(nicemle_4parca_sutun_sutun, resim, seviyeler)
    beklenen = parcalara_uygula(resim, nicemle, seviyeler)
    assert np.array_equal(nicemle_4parca_satir_satir(resim, seviyeler), beklenen)
    assert np.array_equal(nicemle_4parca_sutun_sutun(resim, seviyeler), beklenen)
    print(f"Piksel döngüsü  | satır satır: {satir_suresi:.4f} s | sütun sütun: {sutun_suresi:.4f} s"
          f" | oran: {sutun_suresi / satir_suresi:.2f}x")

    # b) Büyük bir resimde her parçanın satırlarını / sütunlarını tek seferde işleme.
    #    Python yükü azaldığı için bellek erişim farkı burada net görünür.
    buyuk = cv2.resize(resim, (4096, 4096), interpolation=cv2.INTER_NEAREST)

    def satirlari_isle(r):
        for (y1, y2, x1, x2), seviye in zip(parca_sinirlari(r), seviyeler):
            adim = 256 // seviye
            for y in range(y1, y2):
                r[y, x1:x2] // adim * adim

    def sutunlari_isle(r):
        for (y1, y2, x1, x2), seviye in zip(parca_sinirlari(r), seviyeler):
            adim = 256 // seviye
            for x in range(x1, x2):
                r[y1:y2, x] // adim * adim

    satir_suresi = sure_olc(satirlari_isle, buyuk)
    sutun_suresi = sure_olc(sutunlari_isle, buyuk)
    print(f"Satır/sütun dilimi (4096x4096) | satır satır: {satir_suresi:.4f} s | sütun sütun: {sutun_suresi:.4f} s"
          f" | oran: {sutun_suresi / satir_suresi:.2f}x")


# ---------------------------------------------------------------------------
# 4) Sonuçları gösterme
# ---------------------------------------------------------------------------
def etiketle_ve_cizgi_ciz(resim, yazilar):
    """Her parçanın sol üstüne etiketini yazar, parçaların arasına çizgi çeker."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    for (y1, y2, x1, x2), yazi in zip(parca_sinirlari(resim), yazilar):
        cv2.rectangle(kopya, (x1, y1), (x2, y1 + 28), (0, 0, 0), -1)
        cv2.putText(kopya, yazi, (x1 + 6, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    yukseklik, genislik = resim.shape
    cv2.line(kopya, (genislik // 2, 0), (genislik // 2, yukseklik), (0, 0, 255), 2)
    cv2.line(kopya, (0, yukseklik // 2), (genislik, yukseklik // 2), (0, 0, 255), 2)
    return kopya


def goster_ve_kaydet(pencere_adi, dosya_adi, resim):
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = os.path.join(CIKTI_KLASORU, dosya_adi)
    # cv2.imwrite de Türkçe yolda sorun çıkardığı için imencode + tofile
    cv2.imencode(".png", resim)[1].tofile(yol)
    print("Kaydedildi:", yol)

    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(pencere_adi, 800, int(800 * resim.shape[0] / resim.shape[1]))
    cv2.imshow(pencere_adi, resim)
    cv2.waitKey(0)
    cv2.destroyWindow(pencere_adi)


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    resim = resmi_indir()
    print("Boyut:", resim.shape, "| Min:", resim.min(), "| Maks:", resim.max())

    # Deney 1: her parçada farklı gri seviye sayısı
    print("\n--- Deney 1: Nicemleme (4 parça) ---")
    for grup_no, seviyeler in enumerate(SEVIYE_GRUPLARI, start=1):
        sonuc = parcalara_uygula(resim, nicemle, seviyeler)
        yazilar = []
        for (y1, y2, x1, x2), seviye in zip(parca_sinirlari(sonuc), seviyeler):
            bit = int(np.log2(seviye))
            print(f"{seviye:>3} seviye ({bit} bit) -> parçada kullanılan farklı ton sayısı: "
                  f"{len(np.unique(sonuc[y1:y2, x1:x2]))}")
            yazilar.append(f"{seviye} seviye ({bit} bit)")
        goster_ve_kaydet(f"Deney 1 - Gri seviye sayisi ({grup_no}/{len(SEVIYE_GRUPLARI)})",
                         f"4parca_1_nicemleme_{grup_no}.png", etiketle_ve_cizgi_ciz(sonuc, yazilar))

    # Deney 2: her parçada farklı maksimum parlaklık
    print("\n--- Deney 2: Maksimum parlaklık (4 parça) ---")
    sonuc = parcalara_uygula(resim, maks_parlakligi_degistir, MAKS_PARLAKLIKLAR)
    for (y1, y2, x1, x2), maks in zip(parca_sinirlari(sonuc), MAKS_PARLAKLIKLAR):
        print(f"Maks {maks:>3} -> parçanın ortalama parlaklığı: {sonuc[y1:y2, x1:x2].mean():.1f}")
    yazilar = [f"Maks parlaklik = {maks}" for maks in MAKS_PARLAKLIKLAR]
    goster_ve_kaydet("Deney 2 - Maksimum parlaklik", "4parca_2_maks_parlaklik.png",
                     etiketle_ve_cizgi_ciz(sonuc, yazilar))

    # Deney 3: satır satır / sütun sütun gezinme hızı
    gezinme_sirasi_deneyi(resim)


if __name__ == "__main__":
    main()
