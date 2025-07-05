# main_all_features.py
# ----------------------------------------------------------
# Latih model klasifikasi jeruk dengan semua fitur (tanpa MRMR)
# ----------------------------------------------------------

import os
import cv2
import numpy as np
import joblib

from preprocessing import preprocess_image
from feature_extraction import extract_features
from naive_bayes_model import train_model, predict_model, evaluate_model

# Konfigurasi path dataset dan model
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "model/NB1.pkl"  # Nama model berbeda agar tidak tertimpa

# Fungsi: load dataset dari folder
def load_dataset(folder_path):
    images = []
    labels = []
    for label_name in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label_name)
        if os.path.isdir(label_folder):
            for file in os.listdir(label_folder):
                if file.endswith(('.jpg', '.png')):
                    path = os.path.join(label_folder, file)
                    img = cv2.imread(path)
                    if img is not None:
                        images.append(img)
                        labels.append(label_name)
    return images, labels

def main():
    print("🔄 Memuat dataset...")
    train_imgs, train_labels = load_dataset(TRAIN_DIR)
    test_imgs, test_labels = load_dataset(TEST_DIR)

    print("🧼 Pra-pemrosesan gambar...")
    train_processed = [preprocess_image(img) for img in train_imgs]
    test_processed = [preprocess_image(img) for img in test_imgs]

    print("🧠 Ekstraksi fitur (semua fitur digunakan)...")
    train_features = np.array([extract_features(img) for img in train_processed])
    test_features = np.array([extract_features(img) for img in test_processed])

    # Latih model atau load jika sudah ada
    if os.path.exists(MODEL_PATH):
        print("📂 Model ditemukan, memuat dari file...")
        model = joblib.load(MODEL_PATH)
    else:
        print("📊 Melatih model Naive Bayes...")
        model = train_model(train_features, train_labels)
        joblib.dump(model, MODEL_PATH)
        print("✅ Model disimpan di:", MODEL_PATH)

    # Prediksi dan evaluasi
    print("📈 Menguji model...")
    predictions = predict_model(model, test_features)

    print("\n📊 Hasil Evaluasi:")
    evaluate_model(predictions, test_labels)

if __name__ == "__main__":
    main()
