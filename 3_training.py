import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import xgboost as xgb

# ── BACA DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv('dataset_preprocessed.csv')

X = df.drop(columns=['Tingkat Risiko'])
y = df['Tingkat Risiko']

# ── SPLIT DATA TRAIN & TEST (80:20) ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data training : {len(X_train)} baris")
print(f"Data testing  : {len(X_test)} baris")

# ── RANDOM FOREST ─────────────────────────────────────────────────────────────
print("\n=== RANDOM FOREST ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf  = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf  = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf   = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Accuracy  : {acc_rf*100:.2f}%")
print(f"Precision : {prec_rf*100:.2f}%")
print(f"Recall    : {rec_rf*100:.2f}%")
print(f"F1-Score  : {f1_rf*100:.2f}%")
print(f"\nClassification Report Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=['Rendah','Sedang','Tinggi']))

# ── XGBOOST ───────────────────────────────────────────────────────────────────
print("\n=== XGBOOST ===")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

acc_xgb  = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
rec_xgb  = recall_score(y_test, y_pred_xgb, average='weighted')
f1_xgb   = f1_score(y_test, y_pred_xgb, average='weighted')

print(f"Accuracy  : {acc_xgb*100:.2f}%")
print(f"Precision : {prec_xgb*100:.2f}%")
print(f"Recall    : {rec_xgb*100:.2f}%")
print(f"F1-Score  : {f1_xgb*100:.2f}%")
print(f"\nClassification Report XGBoost:")
print(classification_report(y_test, y_pred_xgb, target_names=['Rendah','Sedang','Tinggi']))

# ── RINGKASAN PERBANDINGAN ─────────────────────────────────────────────────────
print("\n=== RINGKASAN PERBANDINGAN ===")
print(f"{'Metrik':<12} {'Random Forest':>15} {'XGBoost':>12}")
print("-" * 42)
print(f"{'Accuracy':<12} {acc_rf*100:>14.2f}% {acc_xgb*100:>11.2f}%")
print(f"{'Precision':<12} {prec_rf*100:>14.2f}% {prec_xgb*100:>11.2f}%")
print(f"{'Recall':<12} {rec_rf*100:>14.2f}% {rec_xgb*100:>11.2f}%")
print(f"{'F1-Score':<12} {f1_rf*100:>14.2f}% {f1_xgb*100:>11.2f}%")

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
label_names = ['Rendah', 'Sedang', 'Tinggi']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, title in zip(axes,
                               [y_pred_rf, y_pred_xgb],
                               ['Random Forest', 'XGBoost']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f'Confusion Matrix - {title}')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("\nGrafik confusion matrix disimpan: confusion_matrix.png")

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
feature_names = X.columns.tolist()
importances_rf  = rf.feature_importances_
importances_xgb = xgb_model.feature_importances_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, importances, title in zip(axes,
                                   [importances_rf, importances_xgb],
                                   ['Random Forest', 'XGBoost']):
    indices = np.argsort(importances)[::-1]
    ax.bar(range(len(feature_names)), importances[indices], color='steelblue')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=30, ha='right')
    ax.set_title(f'Feature Importance - {title}')
    ax.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("Grafik feature importance disimpan: feature_importance.png")

# ── SIMPAN MODEL ──────────────────────────────────────────────────────────────
joblib.dump(rf, 'model_random_forest.pkl')
joblib.dump(xgb_model, 'model_xgboost.pkl')
print("\nModel disimpan: model_random_forest.pkl dan model_xgboost.pkl")
print("\nSelesai.")