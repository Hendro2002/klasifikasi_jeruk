# naive_bayes_model.py

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_pred, y_true):
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    acc = accuracy_score(y_true, y_pred)

    print("=== Confusion Matrix ===")
    print(cm)

    print("\n=== Evaluasi Per Kelas ===")
    for i, label in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        sensitivity = TP / (TP + FN) * 100 if (TP + FN) else 0
        specificity = TN / (TN + FP) * 100 if (TN + FP) else 0
        precision   = TP / (TP + FP) * 100 if (TP + FP) else 0

        print(f"\nKelas: {label}")
        print(f"  Sensitivitas (Recall): {sensitivity:.1f}%")
        print(f"  Spesifisitas         : {specificity:.1f}%")
        print(f"  Presisi              : {precision:.1f}%")

    print(f"\n=== Akurasi Total ===\nAkurasi: {acc * 100:.2f}%")
