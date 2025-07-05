# predict_one.py
# ----------------------------------------------------------
# Prediksi 1 gambar jeruk (FITUR TERPILIH, path gambar FIX)
# Menampilkan semua tahapan visual dan hasil prediksi
# ----------------------------------------------------------

import cv2
import numpy as np
import joblib
import os
import pandas as pd
from tabulate import tabulate

from preprocessing import resize_image, segment_image, complement_image, morph_process, crop_to_object
from feature_extraction import extract_features, FEATURE_NAMES
from naive_bayes_model import predict_model

# ✅ Path model, fitur terpilih, dan gambar uji (GANTI sesuai kebutuhan)
MODEL_PATH = "model/NB2.pkl"
FEATURE_PATH = "feature/SF1.npy"
IMG_PATH = "dataset/test/matang/matang_2.jpg"

def predict_image(img_path):
    print("🟢 Prediksi Gambar Jeruk (Fitur MRMR, Visualisasi Lengkap)")

    # Pastikan model & fitur tersedia
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
        print("❗ Model atau fitur terpilih tidak ditemukan.")
        return

    # Load model dan indeks fitur terpilih
    model = joblib.load(MODEL_PATH)
    selected_indices = np.load(FEATURE_PATH)

    # Load gambar
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Gambar tidak ditemukan: {img_path}")
        return

    # === Pra-pemrosesan manual (agar dapat visualisasi penuh)
    resized = resize_image(img)
    mask = segment_image(resized)
    mask = complement_image(mask)
    mask = morph_process(mask)
    processed = crop_to_object(resized, mask)

    # === Ekstraksi fitur & prediksi
    features = extract_features(processed)
    selected_features = features[selected_indices].reshape(1, -1)
    prediction = predict_model(model, selected_features)[0]

    # === Tampilkan hasil prediksi di terminal
    print("\n🔮 HASIL PREDIKSI")
    print(f"Gambar               : {img_path}")
    print(f"Prediksi Kematangan : {prediction.upper()}")

    # === Tampilkan fitur terpilih
    sorted_indices = sorted(selected_indices)
    print("\n✅ FITUR TERPILIH (MRMR, urut index fitur)")
    table_sel = [[i+1, FEATURE_NAMES[i], f"{features[i]:.4f}"] for i in sorted_indices]
    print(tabulate(table_sel, headers=["#", "Fitur", "Nilai"], tablefmt="grid"))

    # === Simpan ke CSV
    os.makedirs("csv", exist_ok=True)
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    df.insert(0, "Gambar", os.path.basename(img_path))
    df.to_csv("csv/testNB2.csv", index=False)
    print("\n📁 Fitur disimpan ke: csv/testNB2.csv")

    # === Visualisasi tambahan
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

    # Tambahkan teks prediksi ke hasil crop
    result_with_text = processed.copy()
    cv2.putText(result_with_text, f"Prediksi: {prediction.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # === Tampilkan semua jendela
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
    cv2.imshow("11. Crop Akhir + Prediksi", result_with_text)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_image(IMG_PATH)
