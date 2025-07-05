# cek_feature_dataset.py
# Mengekstrak fitur dari seluruh dataset (train & test) ke Excel terpisah & gabungan

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate

from feature_extraction import extract_features, FEATURE_NAMES
from preprocessing import resize_image, segment_image, complement_image, morph_process, crop_to_object

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Gambar tidak ditemukan: {img_path}")
        return None
    resized = resize_image(img)
    mask = segment_image(resized)
    mask = complement_image(mask)
    mask = morph_process(mask)
    processed = crop_to_object(resized, mask)
    features = extract_features(processed)
    return features

def collect_images_by_set(data_root, set_name):
    """
    Mengumpulkan semua gambar di dataset/train dan dataset/test beserta label mapping
    Label mapping: train_mentah, test_matang, dst
    """
    set_dir = os.path.join(data_root, set_name)
    data = []
    for label_folder in sorted(os.listdir(set_dir)):
        full_folder = os.path.join(set_dir, label_folder)
        if not os.path.isdir(full_folder):
            continue
        label = f"{set_name}_{label_folder}"
        img_types = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        img_paths = []
        for ext in img_types:
            img_paths.extend(glob.glob(os.path.join(full_folder, ext)))
        for img_path in sorted(img_paths):
            data.append({'path': img_path, 'label': label})
    return data

def main():
    data_root = "dataset"
    all_data = []
    set_datas = {}

    for set_name in ["train", "test"]:
        print(f"\n🔎 Mengumpulkan gambar dari: {set_name}")
        images = collect_images_by_set(data_root, set_name)
        print(f"   - {len(images)} gambar ditemukan.")
        set_datas[set_name] = images
        all_data.extend(images)

    # Proses ekstraksi fitur
    results = []
    failed = []

    print("\n🚀 Mengekstrak fitur dari semua gambar ...")
    for idx, item in enumerate(all_data, 1):
        img_path = item['path']
        label = item['label']
        print(f"[{idx}/{len(all_data)}] {img_path} ({label})")
        features = process_image(img_path)
        if features is not None:
            results.append([img_path, label] + list(features))
        else:
            failed.append(img_path)

    columns = ["Gambar", "Label"] + FEATURE_NAMES
    df_all = pd.DataFrame(results, columns=columns)
    df_train = df_all[df_all["Label"].str.startswith("train_")].reset_index(drop=True)
    df_test  = df_all[df_all["Label"].str.startswith("test_")].reset_index(drop=True)

    # Simpan ke Excel
    df_train.to_excel("export/fitur_train.xlsx", index=False)
    df_test.to_excel("export/fitur_test.xlsx", index=False)
    df_all.to_excel("export/fitur_dataset_gabungan.xlsx", index=False)

    print("\n✅ Fitur train disimpan ke: fitur_train.xlsx")
    print("✅ Fitur test disimpan ke: fitur_test.xlsx")
    print("✅ Semua fitur gabungan disimpan ke: fitur_dataset_gabungan.xlsx")

    if failed:
        print("\n❌ Gagal memproses beberapa gambar:")
        for f in failed:
            print("-", f)

    # Tampilkan tabel ringkasan
    print("\n📊 Contoh hasil ekstraksi fitur (5 baris pertama):")
    print(tabulate(df_all.head(), headers="keys", tablefmt="pretty"))

if __name__ == "__main__":
    main()