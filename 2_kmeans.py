import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── BACA DATA PREPROCESSED ───────────────────────────────────────────────────
df = pd.read_csv('dataset_preprocessed.csv')

# ── PISAHKAN FITUR DAN LABEL ──────────────────────────────────────────────────
X = df.drop(columns=['Tingkat Risiko'])
y_asli = df['Tingkat Risiko']

# ── NORMALISASI DATA ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── ELBOW METHOD (cari K optimal) ────────────────────────────────────────────
inertia = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method - Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.close()
print("Grafik elbow disimpan: elbow_method.png")

# ── K-MEANS DENGAN K=3 ────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ── MAPPING CLUSTER KE LABEL RISIKO ──────────────────────────────────────────
# Cek rata-rata Ketinggian m per cluster untuk mapping yang logis
cluster_summary = df.groupby('Cluster')['Ketinggian m'].mean().sort_values()
print(f"\nRata-rata Ketinggian per Cluster:\n{cluster_summary}")

mapping = {}
cluster_sorted = cluster_summary.index.tolist()
mapping[cluster_sorted[0]] = 0  # Rendah
mapping[cluster_sorted[1]] = 1  # Sedang
mapping[cluster_sorted[2]] = 2  # Tinggi

df['Label KMeans'] = df['Cluster'].map(mapping)

# ── PERBANDINGAN LABEL ASLI VS K-MEANS ───────────────────────────────────────
print("\n=== DISTRIBUSI LABEL K-MEANS ===")
label_names = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
print(df['Label KMeans'].map(label_names).value_counts())

print("\n=== PERBANDINGAN LABEL ASLI VS K-MEANS ===")
df['Label Asli'] = y_asli
confusion = pd.crosstab(
    df['Label Asli'].map(label_names),
    df['Label KMeans'].map(label_names),
    rownames=['Label Asli'],
    colnames=['Label KMeans']
)
print(confusion)

# ── VISUALISASI HASIL K-MEANS ─────────────────────────────────────────────────
print("\nMembuat grafik visualisasi KMeans...")

# Scatter plot hasil klaster dalam 2 komponen PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='PC1',
    y='PC2',
    hue=df['Label KMeans'].map(label_names),
    palette='Set1',
    s=70,
    edgecolor='w'
)
plt.title('Visualisasi Cluster KMeans di Ruang PCA 2D')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.legend(title='Label KMeans')
plt.tight_layout()
plt.savefig('kmeans_pca_scatter.png', dpi=150)
plt.close()
print("Grafik scatter KMeans disimpan: kmeans_pca_scatter.png")

# Grafik distribusi label KMeans
plt.figure(figsize=(6, 4))
label_counts = df['Label KMeans'].map(label_names).value_counts().reindex(['Rendah', 'Sedang', 'Tinggi'])
label_counts.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52'])
plt.title('Distribusi Label KMeans')
plt.xlabel('Label KMeans')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('kmeans_label_distribution.png', dpi=150)
plt.close()
print("Grafik distribusi label KMeans disimpan: kmeans_label_distribution.png")

# Heatmap perbandingan label asli vs label KMeans
plt.figure(figsize=(6, 5))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Perbandingan Label Asli vs Label KMeans')
plt.tight_layout()
plt.savefig('kmeans_confusion_heatmap.png', dpi=150)
plt.close()
print("Grafik heatmap perbandingan disimpan: kmeans_confusion_heatmap.png")

# ── SIMPAN HASIL ──────────────────────────────────────────────────────────────
df.to_csv('dataset_dengan_kmeans.csv', index=False)
print("\nFile dataset_dengan_kmeans.csv berhasil disimpan.")