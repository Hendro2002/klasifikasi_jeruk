# cek_feature.py
# ----------------------------------------------------------
# Mengekstrak dan menampilkan semua fitur dari satu gambar
# serta menampilkan semua tahapan praproses visual termasuk HSV & CIELAB per kanal
# dan histogram semua fitur
# ----------------------------------------------------------

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from feature_extraction import extract_features, FEATURE_NAMES
from preprocessing import resize_image, segment_image, complement_image, morph_process, crop_to_object

def show_feature_histogram_all(feature_values, feature_names):
    """
    Menampilkan histogram semua fitur berdasarkan nilai absolutnya
    """
    df = pd.DataFrame([feature_values], columns=feature_names)
    scores = df.iloc[0].abs().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=scores.values, y=scores.index, palette="crest")
    plt.title("Histogram Seleksi Fitur MRMR")
    plt.xlabel("Nilai Fitur")
    plt.ylabel("Nama Fitur")
    plt.tight_layout()
    plt.show()

def main():
    # 🔍 Ganti dengan path gambar yang ingin diuji:
    img_path = "dataset/test/matang/matang_2.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Gambar tidak ditemukan:", img_path)
        return

    # 1. Resize
    resized = resize_image(img)

    # 2. Segmentasi dan mask biner
    mask = segment_image(resized)
    mask = complement_image(mask)
    mask = morph_process(mask)

    # 3. Crop hasil akhir (untuk ekstraksi fitur)
    processed = crop_to_object(resized, mask)

    # 4. Ekstraksi fitur
    features = extract_features(processed)

    # 5. Tampilkan tabel fitur
    table = [[i + 1, FEATURE_NAMES[i], f"{features[i]:.4f}"] for i in range(len(FEATURE_NAMES))]
    print("\n📊 Fitur Gambar:", img_path)
    print(tabulate(table, headers=["#", "Nama Fitur", "Nilai"], tablefmt="pretty"))

    # 6. Simpan ke CSV
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    df.insert(0, "Gambar", img_path)
    df.to_csv("fitur_satu_gambar.csv", index=False)
    print("\n✅ Fitur disimpan ke file: fitur_satu_gambar.csv")

    # 7. Tampilkan histogram semua fitur
    show_feature_histogram_all(features, FEATURE_NAMES)

    # 8. Visualisasi HSV, CIELAB, dll
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    mask_vis = (mask * 255).astype(np.uint8)

    h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    s_color = cv2.applyColorMap(s, cv2.COLORMAP_OCEAN)
    v_color = cv2.applyColorMap(v, cv2.COLORMAP_BONE)
    l_color = cv2.applyColorMap(l, cv2.COLORMAP_BONE)
    a_color = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    b_color = cv2.applyColorMap(b, cv2.COLORMAP_JET)

    # 9. Tampilkan semua jendela
    cv2.imshow("1. Gambar Asli", img)
    cv2.imshow("2. Resize", resized)
    cv2.imshow("3. Mask Biner", mask_vis)
    cv2.imshow("4. HSV - H", h_color)
    cv2.imshow("5. HSV - S", s_color)
    cv2.imshow("6. HSV - V", v_color)
    cv2.imshow("7. Grayscale", gray)
    cv2.imshow("8. CIELAB - L*", l_color)
    cv2.imshow("9. CIELAB - a*", a_color)
    cv2.imshow("10. CIELAB - b*", b_color)
    cv2.imshow("11. Crop Akhir", processed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
