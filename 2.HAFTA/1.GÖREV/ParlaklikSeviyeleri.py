"""
Görüntü İşleme Ödevi - Parlaklık (Gri Seviye) Çözünürlüğü
---------------------------------------------------------
İnternetten bir fotoğraf indirir, gri tonlamaya çevirir ve iki deney yapar:

  1) Nicemleme (quantization): 256 ton yerine 128, 64, 32, 16, 8, 4, 2 ton
     kullanılırsa fotoğraf nasıl görünür?
  2) Maksimum parlaklık: en parlak değer 255 değil de 128 veya 64 olursa
     fotoğraf nasıl görünür?

Kullanım:
    python ParlaklikSeviyeleri.py

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
INEN_RESIM = os.path.join(KLASOR, "ciktilar", "cameraman.png")
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

SEVIYELER = [256, 128, 64, 32, 16, 8, 4, 2]
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
# 2) Deneyler
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
def nicemle_satir_satir(resim, seviye_sayisi):
    adim = 256 // seviye_sayisi
    yukseklik, genislik = resim.shape
    sonuc = np.zeros_like(resim)
    for y in range(yukseklik):          # dış döngü: satırlar
        for x in range(genislik):       # iç döngü: sütunlar
            sonuc[y, x] = (resim[y, x] // adim) * adim
    return sonuc


def nicemle_sutun_sutun(resim, seviye_sayisi):
    adim = 256 // seviye_sayisi
    yukseklik, genislik = resim.shape
    sonuc = np.zeros_like(resim)
    for x in range(genislik):           # dış döngü: sütunlar
        for y in range(yukseklik):      # iç döngü: satırlar
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
    print("\n--- Deney 3: Satır satır vs sütun sütun gezinme ---")

    # a) Piksel piksel Python döngüsü (ödevdeki klasik iç içe for yapısı)
    satir_suresi = sure_olc(nicemle_satir_satir, resim, 16)
    sutun_suresi = sure_olc(nicemle_sutun_sutun, resim, 16)
    assert np.array_equal(nicemle_satir_satir(resim, 16), nicemle_sutun_sutun(resim, 16))
    print(f"Piksel döngüsü  | satır satır: {satir_suresi:.4f} s | sütun sütun: {sutun_suresi:.4f} s"
          f" | oran: {sutun_suresi / satir_suresi:.2f}x")

    # b) Büyük bir resimde her satırı / her sütunu tek seferde işleme.
    #    Python yükü azaldığı için bellek erişim farkı burada net görünür.
    buyuk = cv2.resize(resim, (4096, 4096), interpolation=cv2.INTER_NEAREST)

    def satirlari_isle(r):
        for y in range(r.shape[0]):
            r[y, :] // 16 * 16

    def sutunlari_isle(r):
        for x in range(r.shape[1]):
            r[:, x] // 16 * 16

    satir_suresi = sure_olc(satirlari_isle, buyuk)
    sutun_suresi = sure_olc(sutunlari_isle, buyuk)
    print(f"Satır/sütun dilimi (4096x4096) | satır satır: {satir_suresi:.4f} s | sütun sütun: {sutun_suresi:.4f} s"
          f" | oran: {sutun_suresi / satir_suresi:.2f}x")


# ---------------------------------------------------------------------------
# 3) Sonuçları yan yana gösterme
# ---------------------------------------------------------------------------
def etiket_ekle(resim, yazi):
    """Resmin üstüne, okunsun diye siyah bir şerit açıp yazıyı yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(kopya, yazi, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return kopya


def izgara_yap(resimler, sutun_sayisi):
    """Resim listesini satır/sütun düzeninde tek bir büyük resme dönüştürür."""
    bos = np.zeros_like(resimler[0])
    while len(resimler) % sutun_sayisi != 0:
        resimler.append(bos)
    satirlar = [np.hstack(resimler[i:i + sutun_sayisi]) for i in range(0, len(resimler), sutun_sayisi)]
    return np.vstack(satirlar)


def goster_ve_kaydet(pencere_adi, dosya_adi, izgara):
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    yol = os.path.join(CIKTI_KLASORU, dosya_adi)
    # cv2.imwrite de Türkçe yolda sorun çıkardığı için imencode + tofile
    cv2.imencode(".png", izgara)[1].tofile(yol)
    print("Kaydedildi:", yol)

    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(pencere_adi, 1400, int(1400 * izgara.shape[0] / izgara.shape[1]))
    cv2.imshow(pencere_adi, izgara)
    cv2.waitKey(0)
    cv2.destroyWindow(pencere_adi)


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    resim = resmi_indir()
    print("Boyut:", resim.shape, "| Min:", resim.min(), "| Maks:", resim.max())

    # Deney 1: gri seviye sayısını azaltma
    print("\n--- Deney 1: Nicemleme ---")
    kareler = []
    for seviye in SEVIYELER:
        sonuc = nicemle(resim, seviye)
        bit = int(np.log2(seviye))
        print(f"{seviye:>3} seviye ({bit} bit) -> resimde kullanılan farklı ton sayısı: {len(np.unique(sonuc))}")
        kareler.append(etiket_ekle(sonuc, f"{seviye} seviye ({bit} bit)"))
    goster_ve_kaydet("Deney 1 - Gri seviye sayisi", "1_nicemleme.png", izgara_yap(kareler, 4))

    # Deney 2: maksimum parlaklığı düşürme
    print("\n--- Deney 2: Maksimum parlaklık ---")
    kareler = []
    for maks in MAKS_PARLAKLIKLAR:
        sonuc = maks_parlakligi_degistir(resim, maks)
        print(f"Maks {maks:>3} -> ortalama parlaklık: {sonuc.mean():.1f}")
        kareler.append(etiket_ekle(sonuc, f"Maks parlaklik = {maks}"))
    goster_ve_kaydet("Deney 2 - Maksimum parlaklik", "2_maks_parlaklik.png", izgara_yap(kareler, 4))

    # Deney 3: satır satır / sütun sütun gezinme hızı
    gezinme_sirasi_deneyi(resim)


if __name__ == "__main__":
    main()
