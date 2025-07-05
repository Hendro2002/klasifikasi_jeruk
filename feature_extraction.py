# feature_extraction.py

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew, entropy

FEATURE_NAMES = [
    # RGB
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",

    # CIELAB
    "mean_l", "mean_a", "mean_b_lab",
    "std_l", "std_a", "std_b_lab",
    "var_l", "var_a", "var_b",

    # HSV
    "mean_h", "mean_s", "mean_v",
    "std_h", "std_s", "std_v",

    # Grayscale/GLCM
    "contrast", "correlation", "energy", "homogeneity",
    "entropy", "mean_gray", "std_gray", "var_gray", "skewness", "kurtosis"
]

def extract_color_features(img):
    img = cv2.resize(img, (256, 256))

    # RGB
    rgb = img
    mean_r = np.mean(rgb[:, :, 2])
    mean_g = np.mean(rgb[:, :, 1])
    mean_b = np.mean(rgb[:, :, 0])
    std_r = np.std(rgb[:, :, 2])
    std_g = np.std(rgb[:, :, 1])
    std_b = np.std(rgb[:, :, 0])

    # CIELAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b_lab = lab[:, :, 2]
    mean_l = np.mean(l)
    mean_a = np.mean(a)
    mean_b = np.mean(b_lab)
    std_l = np.std(l)
    std_a = np.std(a)
    std_b_l = np.std(b_lab)
    var_l = np.var(l)
    var_a = np.var(a)
    var_b = np.var(b_lab)

    # HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    return [
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        mean_l, mean_a, mean_b,
        std_l, std_a, std_b_l,
        var_l, var_a, var_b,
        mean_h, mean_s, mean_v,
        std_h, std_s, std_v
    ]

def extract_texture_features(gray_img):
    gray = cv2.resize(gray_img, (256, 256)).astype(np.uint8)

    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    gray_flat = gray.flatten()
    hist = np.histogram(gray_flat, bins=256)[0] + 1e-5  # avoid log(0)
    entropy_val = entropy(hist)
    mean_val    = np.mean(gray_flat)
    std_val     = np.std(gray_flat)
    var_val     = np.var(gray_flat)
    skew_val    = skew(gray_flat)
    kurt_val    = kurtosis(gray_flat)

    return [
        contrast, correlation, energy, homogeneity,
        entropy_val, mean_val, std_val, var_val,
        skew_val, kurt_val
    ]

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_features = extract_color_features(img)
    texture_features = extract_texture_features(gray)
    return np.array(color_features + texture_features)
