# cek_preprocessing.py
# ----------------------------------------------------------
# Menampilkan tiap tahap pra-pemrosesan di window terpisah
# ----------------------------------------------------------

import cv2
from preprocessing import (
    resize_image,
    denoise_filter,
    segment_image,
    complement_image,
    morph_process,
    crop_to_object
)

def main():
    # Ganti path sesuai gambar jeruk yang ingin diuji
    path = "dataset/test/setengah_matang/setengah_matang_76.jpg"
    img = cv2.imread(path)

    if img is None:
        print("❌ Gambar tidak ditemukan:", path)
        return

    # Tahap 1: Resize
    resized = resize_image(img)
    cv2.imshow("1. Resize", resized)

    # Tahap 2: Noise Filter (Median + Bilateral)
    filtered = denoise_filter(resized)
    cv2.imshow("2. Noise Removal", filtered)

    # Tahap 3: Segmentasi dari S-channel
    mask_seg = segment_image(filtered)
    cv2.imshow("3. Segmentasi", mask_seg * 255)

    # Tahap 4: Komplement
    mask_comp = complement_image(mask_seg)
    cv2.imshow("4. Komplement", mask_comp * 255)

    # Tahap 5: Morfologi dan noise biner
    mask_morph = morph_process(mask_comp)
    cv2.imshow("5. Morfologi", mask_morph * 255)

    # Tahap 6: Crop hasil akhir
    cropped = crop_to_object(filtered, mask_morph)
    cv2.imshow("6. Crop Final", cropped)

    # Tahap 7: Crop hasil morfologi (biner)
    # Crop mask_morph sesuai bounding box jeruk (sama dengan crop_to_object)
    # Perlu ambil ulang bbox yang dipakai untuk crop agar hasil biner pas dengan crop warna

    # --- Ambil ulang bbox dari mask_morph ---
    import numpy as np
    from skimage.measure import label, regionprops

    labeled = label(mask_morph)
    props = regionprops(labeled)

    if props:
        largest = max(props, key=lambda p: p.area)
        minr, minc, maxr, maxc = largest.bbox
        padding = 5
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(mask_morph.shape[0], maxr + padding)
        maxc = min(mask_morph.shape[1], maxc + padding)
        cropped_binary_mask = mask_morph[minr:maxr, minc:maxc]
        cropped_binary_mask = cv2.resize(cropped_binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    else:
        cropped_binary_mask = cv2.resize(mask_morph, (256, 256), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("7. Crop Final (Biner)", cropped_binary_mask * 255)


    print("✅ Tekan tombol apa pun untuk menutup semua jendela...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
