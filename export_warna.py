import os
import cv2
import numpy as np
import pandas as pd
from webcolors import CSS3_NAMES_TO_HEX, name_to_rgb

# Fungsi RGB ke HEX
def rgb_to_hex(r, g, b):
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)

# Fungsi cari nama warna terdekat
def closest_color(rgb_tuple):
    min_colors = {}
    for name in CSS3_NAMES_TO_HEX.keys():
        r_c, g_c, b_c = name_to_rgb(name)
        rd = (r_c - rgb_tuple[0]) ** 2
        gd = (g_c - rgb_tuple[1]) ** 2
        bd = (b_c - rgb_tuple[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# Path folder gambar
FOLDER_PATH = 'dataset2/train'  # Ganti dengan nama folder kamu

data = []

for filename in os.listdir(FOLDER_PATH):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    file_path = os.path.join(FOLDER_PATH, filename)
    image = cv2.imread(file_path)

    if image is None:
        print(f"❌ Tidak bisa membaca gambar: {file_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (100, 100))

    R = int(np.mean(resized[:, :, 0]))
    G = int(np.mean(resized[:, :, 1]))
    B = int(np.mean(resized[:, :, 2]))

    hex_code = rgb_to_hex(R, G, B)
    color_name = closest_color((R, G, B))

    data.append({
        'Nama Objek': filename,
        'Nama Warna': color_name,
        'Klasifikasi Kode Warna': hex_code,
        'R': R,
        'G': G,
        'B': B,
        
        
    })

# Simpan ke Excel
df = pd.DataFrame(data)
df.to_excel('export/dataset_jeruk_rgb_saja_train.xlsx', index=False, engine='openpyxl')

print("✅ File Excel selesai: dataset_jeruk_rgb_saja_train.xlsx")