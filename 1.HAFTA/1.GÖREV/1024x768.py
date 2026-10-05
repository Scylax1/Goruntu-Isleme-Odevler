"""
Görüntü İşleme Ödevi
--------------------
Bir klasördeki tüm resim dosyalarını bulur ve her birini 1024x768
boyutuna getirerek tek tek ekranda gösterir.

Kullanım:
    python goruntu_gosterici.py                -> klasör seçme penceresi açılır
    python goruntu_gosterici.py C:\\resimler    -> doğrudan bu klasörü açar

Tuşlar:
    D  veya  Sağ ok  veya  Boşluk   -> sonraki resim
    A  veya  Sol ok                 -> önceki resim
    Q  veya  ESC  (ya da pencereyi kapatmak) -> çıkış
"""

import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
HEDEF_GENISLIK = 1024
HEDEF_YUKSEKLIK = 768
PENCERE_ADI = "Goruntu Gosterici"

# Desteklenen resim uzantıları (küçük harfle karşılaştırılacak)
UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Windows'ta cv2.waitKeyEx ile gelen ok tuşu kodları
SAG_OK = 2555904
SOL_OK = 2424832

# OpenCV'miz CUDA ile derlendi; ekran kartı görünüyorsa yeniden boyutlandırmayı GPU'da yaparız
GPU_VAR = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0


# ---------------------------------------------------------------------------
# 1) Klasörü seçme
# ---------------------------------------------------------------------------
def klasor_sec():
    """Komut satırından klasör verildiyse onu, verilmediyse seçme penceresi açar."""
    if len(sys.argv) > 1:
        return sys.argv[1]

    # tkinter Python ile birlikte gelir, ekstra kurulum gerekmez
    import tkinter as tk
    from tkinter import filedialog

    kok = tk.Tk()
    kok.withdraw()  # boş ana pencereyi gizle, sadece klasör seçme penceresi görünsün
    kok.attributes("-topmost", True)  # seçme penceresi editörün arkasında kalmasın
    klasor = filedialog.askdirectory(title="Resimlerin bulunduğu klasörü seçin")
    kok.destroy()
    return klasor


# ---------------------------------------------------------------------------
# 2) Klasördeki resim dosyalarını bulma
# ---------------------------------------------------------------------------
def resimleri_bul(klasor):
    """Klasördeki resim dosyalarının tam yollarını alfabetik sırayla döndürür."""
    dosyalar = []
    for ad in sorted(os.listdir(klasor)):
        yol = os.path.join(klasor, ad)
        if os.path.isfile(yol) and ad.lower().endswith(UZANTILAR):
            dosyalar.append(yol)
    return dosyalar


# ---------------------------------------------------------------------------
# 3) Resmi okuma
# ---------------------------------------------------------------------------
def resim_oku(yol):
    """
    cv2.imread, Windows'ta yolda Türkçe karakter (ş, ğ, ı, ö...) varsa resmi okuyamaz.
    Bu yüzden dosyayı önce ham byte dizisi olarak okuyup sonra OpenCV ile çözüyoruz.
    Okunamazsa None döner.
    """
    try:
        veri = np.fromfile(yol, dtype=np.uint8)
        return cv2.imdecode(veri, cv2.IMREAD_COLOR)  # her zaman 3 kanallı (BGR) renkli resim
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4) 1024x768 boyutuna getirme (asıl görüntü işleme adımı)
# ---------------------------------------------------------------------------
def yeniden_boyutlandir(resim):
    """
    Resmi tam olarak 1024x768 yapar.

    Enterpolasyon seçimi:
      - Küçültürken INTER_AREA: birden fazla pikselin ortalamasını alır,
        titreşim/tırtık (aliasing) oluşmaz. Küçültme için en kaliteli yöntem.
      - Büyütürken INTER_LINEAR: yeni pikselleri komşu 4 pikselden
        doğrusal olarak hesaplar, hızlı ve yumuşak sonuç verir.
    Not: OpenCV boyutu (genişlik, yükseklik) sırasıyla ister,
         ama resim.shape (yükseklik, genişlik, kanal) sırasıyla verir.
    """
    yukseklik, genislik = resim.shape[:2]
    hedef = (HEDEF_GENISLIK, HEDEF_YUKSEKLIK)

    kucultme = genislik > HEDEF_GENISLIK or yukseklik > HEDEF_YUKSEKLIK
    yontem = cv2.INTER_AREA if kucultme else cv2.INTER_LINEAR

    if GPU_VAR:
        try:
            gpu_resim = cv2.cuda_GpuMat()
            gpu_resim.upload(resim)                                  # CPU -> GPU
            gpu_sonuc = cv2.cuda.resize(gpu_resim, hedef, interpolation=yontem)
            return gpu_sonuc.download()                              # GPU -> CPU
        except cv2.error:
            pass  # GPU'da bir sorun olursa aşağıda CPU ile devam et

    return cv2.resize(resim, hedef, interpolation=yontem)


# ---------------------------------------------------------------------------
# 5) Resmin üzerine bilgi yazma
# ---------------------------------------------------------------------------
def bilgi_yaz(resim, metin):
    """Siyah kenarlı beyaz yazı: hem açık hem koyu resimlerde okunur."""
    konum = (15, 35)
    yazi_tipi = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resim, metin, konum, yazi_tipi, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(resim, metin, konum, yazi_tipi, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 6) Tuş bekleme
# ---------------------------------------------------------------------------
def tus_bekle():
    """
    Kullanıcı bir tuşa basana veya pencereyi kapatana kadar bekler.
    Pencere X ile kapatılırsa None döner.
    (waitKeyEx(0) kullanılsaydı pencere kapatıldığında program asılı kalabilirdi.)
    """
    while True:
        tus = cv2.waitKeyEx(50)
        if tus != -1:
            return tus
        if cv2.getWindowProperty(PENCERE_ADI, cv2.WND_PROP_VISIBLE) < 1:
            return None


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    klasor = klasor_sec()
    if not klasor or not os.path.isdir(klasor):
        print("Geçerli bir klasör seçilmedi, çıkılıyor.")
        return

    dosyalar = resimleri_bul(klasor)
    if not dosyalar:
        print(f"'{klasor}' içinde resim dosyası bulunamadı.")
        return

    print(f"{len(dosyalar)} resim bulundu. Yeniden boyutlandırma: {'GPU (CUDA)' if GPU_VAR else 'CPU'}")
    print("Tuşlar: D/Sağ ok/Boşluk = sonraki, A/Sol ok = önceki, Q/ESC = çıkış\n")

    # WINDOW_AUTOSIZE: pencere tam olarak resmin boyutunda (1024x768) açılır
    cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_AUTOSIZE)
    # Pencere terminalin/editörün arkasında açılmasın, en önde dursun
    cv2.setWindowProperty(PENCERE_ADI, cv2.WND_PROP_TOPMOST, 1)

    i = 0
    yon = 1  # okunamayan dosyayı atlarken hangi yöne gideceğimiz
    while dosyalar:
        yol = dosyalar[i]
        resim = resim_oku(yol)

        if resim is None:
            print(f"[ATLANDI] Okunamadı: {os.path.basename(yol)}")
            dosyalar.pop(i)
            if not dosyalar:
                break
            i = i % len(dosyalar) if yon == 1 else (i - 1) % len(dosyalar)
            continue

        orj_y, orj_g = resim.shape[:2]
        gosterilecek = yeniden_boyutlandir(resim)
        bilgi_yaz(gosterilecek, f"{i + 1}/{len(dosyalar)}   orijinal: {orj_g}x{orj_y}  ->  1024x768")

        cv2.imshow(PENCERE_ADI, gosterilecek)
        print(f"{i + 1}/{len(dosyalar)}  {os.path.basename(yol)}  ({orj_g}x{orj_y} -> 1024x768)")

        tus = tus_bekle()
        if tus is None or tus in (27, ord("q"), ord("Q")):
            break
        elif tus in (ord("d"), ord("D"), SAG_OK, 32):
            yon = 1
            i = (i + 1) % len(dosyalar)   # sondan sonra başa döner
        elif tus in (ord("a"), ord("A"), SOL_OK):
            yon = -1
            i = (i - 1) % len(dosyalar)   # baştan önce sona döner
        # diğer tuşlarda aynı resimde kalır

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()