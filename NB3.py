# main.py

import os
import cv2
import numpy as np
import joblib

from preprocessing import preprocess_image
from feature_extraction import extract_features
from feature_selection_15 import select_features_mrmr
from naive_bayes_model import train_model, predict_model, evaluate_model

# Konfigurasi path dan file model
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "model/NB3.pkl"
FEATURE_PATH = "feature/SF2.npy"

# Load gambar dan label dari folder
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

    print("🧼 Pra-pemrosesan gambar latih...")
    train_processed = [preprocess_image(img) for img in train_imgs]
    print("🧼 Pra-pemrosesan gambar uji...")
    test_processed = [preprocess_image(img) for img in test_imgs]

    print("🧠 Ekstraksi fitur warna dan tekstur...")
    train_features = np.array([extract_features(img) for img in train_processed])
    test_features = np.array([extract_features(img) for img in test_processed])

    # Cek apakah model sudah ada
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        print("📂 Memuat model & fitur terpilih...")
        model = joblib.load(MODEL_PATH)
        selected_indices = np.load(FEATURE_PATH)
    else:
        print("🔍 Seleksi fitur dengan MRMR...")
        selected_indices = select_features_mrmr(train_features, np.array(train_labels), k=15)
        np.save(FEATURE_PATH, selected_indices)

        print("📊 Melatih model Naive Bayes...")
        X_train_selected = train_features[:, selected_indices]
        model = train_model(X_train_selected, train_labels)
        joblib.dump(model, MODEL_PATH)
        print("✅ Model disimpan di:", MODEL_PATH)

    # Prediksi dan evaluasi
    print("📈 Menguji model...")
    X_test_selected = test_features[:, selected_indices]
    predictions = predict_model(model, X_test_selected)

    print("\n📊 Hasil Evaluasi:")
    evaluate_model(predictions, test_labels)

if __name__ == "__main__":
    main()
