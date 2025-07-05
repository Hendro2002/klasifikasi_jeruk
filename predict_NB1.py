# predict_NB1.py
# ----------------------------------------------------------
# Prediksi 1 gambar jeruk menggunakan model NB1
# Menampilkan semua tahap visual seperti cek_feature
# ----------------------------------------------------------

import cv2
import numpy as np
import joblib
import pandas as pd
from tabulate import tabulate
from preprocessing import resize_image, segment_image, complement_image, morph_process, crop_to_object
from feature_extraction import extract_features, FEATURE_NAMES
from naive_bayes_model import predict_model
import os

def main():
    # 🔍 Ganti path gambar & model
    img_path = "dataset/test/matang/matang_2.jpg"
    model_path = "model/NB1.pkl"
    csv_output_path = "csv/testNB1.csv"

    if not os.path.exists(img_path) or not os.path.exists(model_path):
        print("❌ Gambar atau model tidak ditemukan.")
        return

    # Load model & gambar
    model = joblib.load(model_path)
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Gagal membaca gambar:", img_path)
        return

    # === Pra-pemrosesan manual untuk visualisasi penuh ===
    resized = resize_image(img)
    mask = segment_image(resized)
    mask = complement_image(mask)
    mask = morph_process(mask)
    processed = crop_to_object(resized, mask)

    # === Ekstraksi fitur & prediksi ===
    features = extract_features(processed).reshape(1, -1)
    predicted_label = predict_model(model, features)[0]

    # === Simpan CSV hasil prediksi ===
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df.insert(0, "Gambar", os.path.basename(img_path))
    df["Prediksi"] = predicted_label
    df.to_csv(csv_output_path, index=False)
    print(f"\n✅ Fitur & hasil prediksi disimpan di: {csv_output_path}")

    # === Tampilkan hasil prediksi di terminal ===
    print("\n🎯 Hasil Prediksi:")
    print("Gambar:", img_path)
    print("Prediksi Kelas:", predicted_label)

    # === Tampilkan fitur di terminal ===
    print("\n📋 Fitur yang diekstrak:")
    table = [[i + 1, FEATURE_NAMES[i], f"{features[0][i]:.4f}"] for i in range(len(FEATURE_NAMES))]
    print(tabulate(table, headers=["#", "Nama Fitur", "Nilai"], tablefmt="grid"))

    # === Visualisasi lengkap ===
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    s_color = cv2.applyColorMap(s, cv2.COLORMAP_OCEAN)
    v_color = cv2.applyColorMap(v, cv2.COLORMAP_BONE)

    l_color = cv2.applyColorMap(l, cv2.COLORMAP_BONE)
    a_color = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    b_color = cv2.applyColorMap(b, cv2.COLORMAP_JET)

    mask_vis = (mask * 255).astype(np.uint8)

    # Tambahkan label prediksi ke gambar hasil crop
    result_with_text = processed.copy()
    cv2.putText(result_with_text, f"Prediksi: {predicted_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # === Tampilkan semua jendela
    cv2.imshow("1. Gambar Asli", img)
    cv2.imshow("2. Resize", resized)
    cv2.imshow("3. Mask Biner", mask_vis)
    cv2.imshow("4. HSV - Hue (H)", h_color)
    cv2.imshow("5. HSV - Saturation (S)", s_color)
    cv2.imshow("6. HSV - Value (V)", v_color)
    cv2.imshow("7. Grayscale", gray)
    cv2.imshow("8. CIELAB - L*", l_color)
    cv2.imshow("9. CIELAB - a*", a_color)
    cv2.imshow("10. CIELAB - b*", b_color)
    cv2.imshow("11. Crop Akhir + Prediksi", result_with_text)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
