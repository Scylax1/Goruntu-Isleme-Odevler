"""
Görüntü İşleme Ödevi - Farklı Çözünürlüklerde Görüntü Kalitesi
---------------------------------------------------------------
ciktilar/SiyahBeyazKare.jpeg (dalgalı siyah-beyaz dama tahtası) ekran
çözünürlüğüne göre farklı ölçeklerde yeniden örneklenir:

    x2    -> ekran çözünürlüğünün 2 katı
    x4    -> ekran çözünürlüğünün 4 katı
    x0.5  -> ekran çözünürlüğünün yarısı
    x0.25 -> ekran çözünürlüğünün dörtte biri

Her sonuç ekrana sığacak şekilde (ekran çözünürlüğüne) geri getirilip gösterilir.
Geri getirirken INTER_NEAREST kullanıldığı için düşük çözünürlükte pikseller
kare kare (bloklu) görünür; yani kalite kaybı gözle fark edilir.
Son pencerede aynı bölgenin tüm çözünürlüklerdeki yakın çekimi yan yana gösterilir.

Kullanım:
    python GörüntüKalitesi.py

Herhangi bir tuşa basınca bir sonraki pencereye geçer.
"""

import ctypes
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")
RESIM_YOLU = os.path.join(CIKTI_KLASORU, "SiyahBeyazKare.jpeg")

OLCEKLER = [2, 4, 0.5, 0.25]

# Yakın çekim karşılaştırması için ekranın ortasından alınacak bölge (ekran pikseli)
YAKIN_CEKIM_BOYUTU = (160, 100)   # (genişlik, yükseklik)
YAKIN_CEKIM_BUYUTME = 3           # her parçayı gösterirken kaç kat büyütelim


# ---------------------------------------------------------------------------
# 1) Ekran çözünürlüğünü bulma
# ---------------------------------------------------------------------------
def ekran_cozunurlugu():
    """Gerçek (fiziksel) ekran çözünürlüğünü döndürür.

    Windows'ta ölçekleme (%125, %150 ...) açıksa, SetProcessDPIAware çağrılmadan
    sorulan çözünürlük ölçeklenmiş sahte değer olur (ör. 1920x1200 yerine 1536x960).
    """
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except AttributeError:
        # Windows dışı sistemler: tkinter ile sor
        import tkinter as tk
        kok = tk.Tk()
        boyut = kok.winfo_screenwidth(), kok.winfo_screenheight()
        kok.destroy()
        return boyut


# ---------------------------------------------------------------------------
# 2) Resmi okuma / kaydetme (Türkçe karakterli yollar için imdecode / imencode)
# ---------------------------------------------------------------------------
def resim_oku(yol):
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


def kaydet(dosya_adi, resim):
    yol = os.path.join(CIKTI_KLASORU, dosya_adi)
    cv2.imencode(".png", resim)[1].tofile(yol)
    return yol


# ---------------------------------------------------------------------------
# 3) Yeniden örnekleme
# ---------------------------------------------------------------------------
def olcekle(resim, genislik, yukseklik):
    """Resmi verilen boyuta getirir.

    Küçültürken INTER_AREA: birden çok pikselin ortalamasını alır, dama tahtasında
    oluşabilecek hare (moiré) desenini azaltır.
    Büyütürken INTER_CUBIC: 4x4 komşuluktan hesaplar, kenarlar daha yumuşak olur.
    """
    y, g = resim.shape[:2]
    kucultme = genislik < g or yukseklik < y
    yontem = cv2.INTER_AREA if kucultme else cv2.INTER_CUBIC
    return cv2.resize(resim, (genislik, yukseklik), interpolation=yontem)


def ekrana_getir(resim, ekran):
    """Sonucu ekran boyutunda göstermek için geri ölçekler.

    INTER_NEAREST ile büyütülen düşük çözünürlüklü resimde her piksel kare blok
    olarak görünür; böylece kaybolan detay açıkça görülür.
    """
    y, g = resim.shape[:2]
    if g > ekran[0]:
        return cv2.resize(resim, ekran, interpolation=cv2.INTER_AREA)
    return cv2.resize(resim, ekran, interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# 4) Yakın çekim karşılaştırması
# ---------------------------------------------------------------------------
def yakin_cekim(resim, ekran):
    """Ekranın ortasındaki aynı bölgeyi, resmin kendi çözünürlüğünde keser ve
    tüm parçalar aynı boyutta olsun diye INTER_NEAREST ile büyütür."""
    oran = resim.shape[1] / ekran[0]
    kg, ky = YAKIN_CEKIM_BOYUTU
    x1 = int((ekran[0] - kg) / 2 * oran)
    y1 = int((ekran[1] - ky) / 2 * oran)
    parca = resim[y1:y1 + max(1, int(ky * oran)), x1:x1 + max(1, int(kg * oran))]
    hedef = (kg * YAKIN_CEKIM_BUYUTME, ky * YAKIN_CEKIM_BUYUTME)
    yontem = cv2.INTER_AREA if parca.shape[1] > hedef[0] else cv2.INTER_NEAREST
    return cv2.resize(parca, hedef, interpolation=yontem)


def etiket_yaz(resim, metin):
    """Gri resmi renkliye çevirip üstüne okunaklı bir etiket yazar."""
    kopya = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(kopya, metin, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return kopya


def karsilastirma_olustur(parcalar):
    """Etiketli yakın çekimleri 3 sütunlu bir ızgarada birleştirir."""
    y, g = parcalar[0].shape[:2]
    bos = np.zeros_like(parcalar[0])
    while len(parcalar) % 3:
        parcalar.append(bos)
    satirlar = [np.hstack(parcalar[i:i + 3]) for i in range(0, len(parcalar), 3)]
    izgara = np.vstack(satirlar)
    # parçaların arasına kırmızı çizgi
    for x in range(g, izgara.shape[1], g):
        cv2.line(izgara, (x, 0), (x, izgara.shape[0]), (0, 0, 255), 2)
    for yy in range(y, izgara.shape[0], y):
        cv2.line(izgara, (0, yy), (izgara.shape[1], yy), (0, 0, 255), 2)
    return izgara


# ---------------------------------------------------------------------------
# 5) Gösterme
# ---------------------------------------------------------------------------
def goster(pencere_adi, resim, ekran):
    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    # Pencere çerçevesi ve görev çubuğu için biraz pay bırak
    g = int(ekran[0] * 0.8)
    y = int(g * resim.shape[0] / resim.shape[1])
    cv2.resizeWindow(pencere_adi, g, y)
    cv2.setWindowProperty(pencere_adi, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(pencere_adi, resim)
    cv2.waitKey(0)
    cv2.destroyWindow(pencere_adi)


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main(goruntule=True):
    ekran = ekran_cozunurlugu()
    orijinal = resim_oku(RESIM_YOLU)
    print(f"Ekran çözünürlüğü: {ekran[0]}x{ekran[1]}")
    print(f"Orijinal resim   : {orijinal.shape[1]}x{orijinal.shape[0]}\n")

    # Referans: resmin ekran çözünürlüğüne getirilmiş hali (x1)
    x1 = olcekle(orijinal, *ekran)
    sonuclar = [("x1 (ekran)", x1)]
    for olcek in OLCEKLER:
        g, y = int(ekran[0] * olcek), int(ekran[1] * olcek)
        sonuclar.append((f"x{olcek}", olcekle(x1, g, y)))

    print(f"{'Ölçek':<11}{'Çözünürlük':>12}{'Piksel sayısı':>16}{'PNG boyutu':>13}")
    yakin_cekimler = []
    for ad, resim in sonuclar:
        g, y = resim.shape[1], resim.shape[0]
        yol = kaydet(f"kalite_{ad.split()[0]}_{g}x{y}.png", resim)
        kb = os.path.getsize(yol) / 1024
        print(f"{ad:<11}{f'{g}x{y}':>12}{g * y:>16,}{kb:>10.0f} KB")

        yakin_cekimler.append(etiket_yaz(yakin_cekim(resim, ekran), f"{ad}  {g}x{y}"))
        if goruntule:
            goster(f"{ad}  -  {g}x{y}  (ekranda {ekran[0]}x{ekran[1]} olarak)",
                   etiket_yaz(ekrana_getir(resim, ekran), f"{ad}  {g}x{y}"), ekran)

    karsilastirma = karsilastirma_olustur(yakin_cekimler)
    print("\nKarşılaştırma kaydedildi:", kaydet("kalite_karsilastirma.png", karsilastirma))
    if goruntule:
        goster("Ayni bolgenin yakin cekimi - tum cozunurlukler", karsilastirma, ekran)


if __name__ == "__main__":
    main()
