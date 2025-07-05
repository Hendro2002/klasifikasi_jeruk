# preprocessing.py
# --------------------------------------------------------
# Modul ini memproses gambar jeruk untuk klasifikasi:
# Resize, segmentasi, filtering, denoise, morfologi, crop
# --------------------------------------------------------

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk

# --- 1. Resize gambar ---
def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

# --- 2. Filter awal untuk reduksi noise (non-biner) ---
def denoise_filter(img):
    """
    Mengurangi noise sambil mempertahankan tepi.
    Kombinasi median dan bilateral filter.
    """
    median = cv2.medianBlur(img, 5)
    filtered = cv2.bilateralFilter(median, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered

# --- 3. Segmentasi berdasarkan kanal saturasi HSV ---
def segment_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    _, thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (s_channel > thresh).astype(np.uint8)
    return mask

# --- 4. Komplement: balik foreground ↔ background ---
def complement_image(mask):
    return 1 - mask

# --- 5. Hapus noise kecil dari mask biner ---
def remove_noise(mask):
    mask = remove_small_objects(mask.astype(bool), min_size=500)
    mask = remove_small_holes(mask, area_threshold=500)
    return mask.astype(np.uint8)

# --- 6. Morfologi untuk menutup objek ---
def morph_process(mask):
    closed = binary_closing(mask, disk(3))
    denoised = remove_noise(closed)
    return denoised

# --- 7. Crop objek utama jeruk (area terbesar) ---
def crop_to_object(img, mask):
    labeled = label(mask)
    props = regionprops(labeled)

    if not props:
        return cv2.resize(img, (256, 256))  # fallback jika gagal segmentasi

    # Bounding box objek terbesar
    largest = max(props, key=lambda p: p.area)
    minr, minc, maxr, maxc = largest.bbox

    # Tambahkan padding kecil agar tidak terlalu mepet
    padding = 5
    minr = max(0, minr - padding)
    minc = max(0, minc - padding)
    maxr = min(img.shape[0], maxr + padding)
    maxc = min(img.shape[1], maxc + padding)

    cropped = img[minr:maxr, minc:maxc]
    return cv2.resize(cropped, (256, 256))

# --- 8. Pipeline utama ---
def preprocess_image(img):
    resized = resize_image(img)
    filtered = denoise_filter(resized)
    mask = segment_image(filtered)
    mask = complement_image(mask)
    mask = morph_process(mask)
    result = crop_to_object(filtered, mask)
    return result
