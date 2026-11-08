"""
Görüntü İşleme Ödevi - X-ray Bagaj Görüntülerini Gri Tona Çevirip Renklendirme
------------------------------------------------------------------------------
HUMS X-ray veri setinin ThreatsRGB klasöründen seçilen 5 görüntüyü
(bıçak içeren bagajlar) önce gri tonlu resme dönüştürür, sonra güvenlik
operatörlerinin daha rahat yorumlayabilmesi için kendi tasarladığımız bir
renk paletiyle (LUT) yeniden renklendirir ve orijinal renkli resimle
karşılaştırır.

Veri seti: https://www.kaggle.com/datasets/orvile/hums-x-ray-dataset
(CC BY 4.0). Ödevde "ThreadsRGB" diye geçen klasörün asıl adı "ThreatsRGB".

Orijinal resimler nasıl renklendirilmiş?
  Tarayıcı çift enerjilidir: her piksel için yüksek ve düşük enerjide iki
  ayrı soğurma ölçer. İki ölçümün oranı malzemenin etkin atom numarasını
  (Z) verir, toplam soğurma da kalınlık x yoğunluğu. Sektör standardı renk
  kodu:
      turuncu : organik (düşük Z: plastik, kağıt, gıda, PATLAYICI)
      yeşil   : karışık / hafif inorganik (alüminyum, cam)
      mavi    : metal (yüksek Z: çelik, bıçak, silah)
      siyah   : nüfuz edilemeyen (çok kalın / yoğun)
  Renk tonu MALZEMEYİ, parlaklık KALINLIK x YOĞUNLUĞU gösterir.

Bizim renklendirmemiz:
  1) Gri dönüşüm: cv2.cvtColor(BGR2GRAY), Y = 0.299 R + 0.587 G + 0.114 B.
     Gri resim yalnızca parlaklığı tutar; ton bilgisi (malzeme) kaybolur.
  2) Gri değer 0..255 -> renk tablosu (LUT). Tablo, aşağıdaki çapa
     noktaları arasında doğrusal ara değerlemeyle 256 renge açılır:

         gri   0 : siyah        (nüfuz edilemeyen)
         gri  25 : lacivert
         gri  70 : mavi         (yoğun / kalın: metal benzeri)
         gri 115 : açık mavi
         gri 140 : yeşil        (orta)
         gri 170 : sarı-yeşil
         gri 190 : turuncu      (ince / hafif: organik benzeri)
         gri 225 : açık turuncu
         gri 245 : krem
         gri 255 : beyaz        (boş alan)

Neden bu palet?
  - Operatörler zaten turuncu / yeşil / mavi / siyah koduyla eğitiliyor.
    Tanıdık renkler yeni bir şey öğrenmeden okunur, dikkat dağılmaz.
  - Yoğun nesneler (bıçak, silah gibi metaller) en koyu gri değerlerde
    toplanıyor. Onlara açık zemine en zıt düşen doygun maviyi verdik.
    Bıçaklar beyaz / turuncu zeminde hemen göze çarpar.
  - Paletin parlaklığı (luma) gri değerle tekdüze artar: tablodaki her
    rengin luma'sı, karşılığı olan gri değere ±5 içinde eşittir. Böylece
    renkli resim gri resmin bilgisini bozmaz; renk yalnızca kategoriyi
    ekler. Renkli resim tekrar griye çevrilirse aşağı yukarı aynı gri resim
    çıkar. JET gibi gökkuşağı paletleri parlaklıkta iniş-çıkış yapar ve
    olmayan sınırlar uydurur; üstelik boş zemini koyu kırmızı / bordo
    yapar. Bu yüzden onları seçmedik (karşılaştırma resminde gösteriliyor).
  - Boş alan beyaz kalır. Göz, renkli olan yere, yani nesnelere gider.

Fark analizi:
  Orijinal resmin her pikseli tonuna göre sınıflanır (arka plan / turuncu /
  yeşil / mavi), bizim resmimizin her pikseli de LUT'ta düştüğü renge göre.
  İki sınıflamanın karışıklık matrisi ve uyuşma oranı yazdırılır, uyuşmayan
  pikseller de bir haritada gösterilir.

Kullanım:
    python XrayRenklendirme.py
"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
VERI_KLASORU = os.path.join(KLASOR, "veri")
CIKTI_KLASORU = os.path.join(KLASOR, "ciktilar")

# (gri değer, (R, G, B)) çapa noktaları
CAPALAR = [(0, (0, 0, 0)),
           (25, (10, 20, 90)),
           (70, (20, 70, 220)),
           (115, (50, 125, 225)),
           (140, (90, 180, 80)),
           (170, (150, 200, 70)),
           (190, (245, 175, 90)),
           (225, (255, 215, 160)),
           (245, (255, 248, 230)),
           (255, (255, 255, 255))]

# Bizim paletin kategorileri (gri aralığına göre)
SINIFLAR = ["arka plan", "turuncu", "yeşil", "mavi/siyah"]
SINIF_RENKLERI = [(255, 255, 255), (245, 175, 90), (90, 180, 80), (20, 70, 220)]   # RGB


def gri_sinifi(gri):
    """Gri değerden bizim paletteki kategori: 0 arka plan, 1 turuncu, 2 yeşil, 3 mavi/siyah."""
    sinif = np.full(gri.shape, 3, np.uint8)        # < 128: mavi / siyah
    sinif[gri >= 128] = 2                           # 128..179: yeşil
    sinif[gri >= 180] = 1                           # 180..239: turuncu
    sinif[gri >= 240] = 0                           # >= 240: boş alan
    return sinif


def orijinal_sinifi(bgr):
    """Orijinal sahte renkli resmin tonuna göre kategori (aynı numaralar)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sinif = np.zeros(h.shape, np.uint8)                              # doygun değil: arka plan
    renkli = s >= 60
    sinif[renkli & ((h <= 27) | (h >= 160))] = 1                     # turuncu / sarı
    sinif[renkli & (h > 27) & (h <= 85)] = 2                         # yeşil
    sinif[renkli & (h > 85) & (h < 160)] = 3                         # mavi / mor
    sinif[v < 50] = 3                                                # siyah: nüfuz edilemeyen
    return sinif


# ---------------------------------------------------------------------------
# Türkçe karakterli yollar için okuma / yazma yardımcıları
# ---------------------------------------------------------------------------
def renkli_oku(yol):
    # cv2.imread Türkçe karakterli yolları okuyamadığı için imdecode kullanıyoruz
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_COLOR)


def kaydet(yol, resim):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    cv2.imencode(".png", resim)[1].tofile(yol)
    print("Kaydedildi:", yol)


# ---------------------------------------------------------------------------
# Renk tablosu
# ---------------------------------------------------------------------------
def renk_tablosu_olustur():
    """Çapa noktaları arasında doğrusal ara değerleme: 256 x 3 BGR tablo."""
    tablo = np.zeros((256, 3), np.float64)
    for (g0, renk0), (g1, renk1) in zip(CAPALAR[:-1], CAPALAR[1:]):
        for g in range(g0, g1 + 1):
            t = (g - g0) / (g1 - g0)
            tablo[g] = [(1 - t) * a + t * b for a, b in zip(renk0, renk1)]
    rgb = np.rint(tablo).astype(np.uint8)
    return rgb[:, ::-1].copy()          # RGB -> BGR (OpenCV sırası)


def renklendir(gri, tablo):
    """Her gri piksel, tablodaki rengiyle değiştirilir (LUT)."""
    return tablo[gri]


# ---------------------------------------------------------------------------
# Palet çizimi: bizim palet / JET, renk şeridi ve parlaklık eğrisi
# ---------------------------------------------------------------------------
def palet_ciz(tablo):
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET).reshape(256, 3)
    luma = lambda bgr: cv2.cvtColor(bgr.reshape(256, 1, 3), cv2.COLOR_BGR2GRAY).ravel()
    bizim_luma, jet_luma = luma(tablo), luma(jet)
    sapma = np.abs(bizim_luma.astype(int) - np.arange(256))
    print(f"Paletimizin parlaklığı (luma) gri değerden en fazla {sapma.max()} sapıyor; "
          f"tekdüze artıyor mu: {bool(np.all(np.diff(bizim_luma.astype(int)) >= 0))}")
    print(f"JET'in parlaklığı: en düşük {jet_luma.min()}, en yüksek {jet_luma.max()} "
          f"(gri {int(np.argmax(jet_luma))}'de), gri 255'te {jet_luma[255]}; tekdüze değil")

    fig, eksenler = plt.subplots(3, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 1, 3]})
    for ax, t, ad in ((eksenler[0], tablo, "Bizim palet"), (eksenler[1], jet, "JET (seçilmedi)")):
        ax.imshow(np.repeat(t[None, :, ::-1], 20, axis=0), aspect="auto", extent=(0, 255, 0, 1))
        ax.set_yticks([])
        ax.set_ylabel(ad, rotation=0, ha="right", va="center")
    eksenler[0].set_xticks([g for g, _ in CAPALAR])
    eksenler[1].set_xticks([])
    ax = eksenler[2]
    ax.plot(np.arange(256), bizim_luma, color="tab:orange", label="Bizim palet")
    ax.plot(np.arange(256), jet_luma, color="tab:purple", label="JET")
    ax.plot([0, 255], [0, 255], "k--", linewidth=0.8, label="ideal: luma = gri")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Gri değer")
    ax.set_ylabel("Rengin parlaklığı (luma)")
    ax.legend()
    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "palet.png")
    fig.savefig(yol, dpi=100)
    plt.close(fig)
    print("Kaydedildi:", yol)


# ---------------------------------------------------------------------------
# Karşılaştırma
# ---------------------------------------------------------------------------
def sinif_resmi(sinif):
    renkler = np.array([c[::-1] for c in SINIF_RENKLERI], np.uint8)
    return renkler[sinif]


def karisiklik(orijinal, bizim):
    matris = np.zeros((4, 4), np.int64)
    np.add.at(matris, (orijinal.ravel(), bizim.ravel()), 1)
    return matris


def matris_yazdir(matris, baslik):
    print(f"\n{baslik}  (satır: orijinal renk, sütun: bizim renk, satır yüzdesi)")
    print(f"{'':<14}" + "".join(f"{ad:>12}" for ad in SINIFLAR))
    for i, ad in enumerate(SINIFLAR):
        toplam = max(matris[i].sum(), 1)
        print(f"{ad:<14}" + "".join(f"{100 * matris[i, j] / toplam:>11.1f}%" for j in range(4))
              + f"   ({matris[i].sum()} px)")
    print(f"Uyuşma: %{100 * np.trace(matris) / matris.sum():.1f}")


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def main():
    dosyalar = sorted(glob.glob(os.path.join(VERI_KLASORU, "*.jpg")))
    if not dosyalar:
        raise SystemExit(f"{VERI_KLASORU} içinde resim yok")

    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    tablo = renk_tablosu_olustur()
    palet_ciz(tablo)
    jet = lambda gri: cv2.applyColorMap(gri, cv2.COLORMAP_JET)

    toplam_matris = np.zeros((4, 4), np.int64)
    fig, eksenler = plt.subplots(len(dosyalar), 5, figsize=(22, 3.6 * len(dosyalar)))
    for satir, yol in enumerate(dosyalar):
        ad = os.path.splitext(os.path.basename(yol))[0]
        orijinal = renkli_oku(yol)
        gri = cv2.cvtColor(orijinal, cv2.COLOR_BGR2GRAY)
        renkli = renklendir(gri, tablo)

        o_sinif, b_sinif = orijinal_sinifi(orijinal), gri_sinifi(gri)
        matris = karisiklik(o_sinif, b_sinif)
        toplam_matris += matris
        uyusmayan = o_sinif != b_sinif
        print(f"{ad}: {orijinal.shape[1]}x{orijinal.shape[0]} | uyuşma %{100 * np.trace(matris) / matris.sum():.1f}")

        kaydet(os.path.join(CIKTI_KLASORU, f"{ad}_gri.png"), gri)
        kaydet(os.path.join(CIKTI_KLASORU, f"{ad}_renkli.png"), renkli)

        # Uyuşmayan pikseller: orijinaldeki doğru sınıfın rengiyle, gerisi soluk gri
        fark = np.full_like(orijinal, 235)
        fark[uyusmayan] = sinif_resmi(o_sinif)[uyusmayan]

        paneller = [(orijinal, "Orijinal (çift enerji, sahte renk)"),
                    (cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR), "Gri (BGR2GRAY)"),
                    (renkli, "Bizim renklendirme"),
                    (jet(gri), "JET (karşılaştırma)"),
                    (fark, f"Sınıfı farklı pikseller (%{100 * uyusmayan.mean():.1f})")]
        for sutun, (resim, baslik) in enumerate(paneller):
            ax = eksenler[satir, sutun]
            ax.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{ad}: {baslik}" if sutun == 0 else baslik, fontsize=10)
            ax.axis("off")

    fig.tight_layout()
    yol = os.path.join(CIKTI_KLASORU, "renklendirme_karsilastirma.png")
    fig.savefig(yol, dpi=70)
    print("Kaydedildi:", yol)

    matris_yazdir(toplam_matris, "5 resmin toplamı")
    plt.show()


if __name__ == "__main__":
    main()
