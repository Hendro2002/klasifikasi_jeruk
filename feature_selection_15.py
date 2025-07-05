# feature_selection.py

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

def pearson_corr(x, y):
    """Hitung korelasi Pearson antara dua vektor"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    return np.corrcoef(x, y)[0, 1]

def select_features_mrmr(X, y, k=15):
    """
    MRMR berdasarkan FCD (F-test Correlation Difference)
    X: matriks fitur (n_samples x n_features)
    y: label klasifikasi (kategori/kelas)
    k: jumlah fitur yang ingin dipilih
    return: indeks fitur terpilih (list[int])
    """
    n_features = X.shape[1]

    # Encode label (misal dari string ke angka)
    if y.dtype.kind in {'U', 'S', 'O'}:
        y = LabelEncoder().fit_transform(y)

    # 1. Hitung relevansi: F-score antara fitur dan label
    f_scores, _ = f_classif(X, y)

    # 2. Hitung korelasi antar fitur
    corr_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                corr_matrix[i, j] = abs(pearson_corr(X[:, i], X[:, j]))

    # 3. Hitung FCD: relevansi - rata-rata korelasi
    fcd_scores = np.zeros(n_features)
    for i in range(n_features):
        redundancy = np.mean(corr_matrix[i])
        fcd_scores[i] = f_scores[i] - redundancy

    # 4. Ambil fitur dengan skor FCD tertinggi
    selected_indices = np.argsort(fcd_scores)[-k:][::-1]
    return selected_indices.tolist()
