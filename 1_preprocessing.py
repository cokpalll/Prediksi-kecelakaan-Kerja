import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ── BACA DATASET ──────────────────────────────────────────────────────────────
df = pd.read_excel('dataset_training_K3_FINAL.xlsx', sheet_name='Dataset Gabungan')

print("=== INFO DATASET ===")
print(f"Jumlah baris  : {len(df)}")
print(f"Jumlah kolom  : {len(df.columns)}")
print(f"\nKolom : {list(df.columns)}")
print(f"\nCek missing value:\n{df.isnull().sum()}")

# ── HAPUS KOLOM TIDAK DIPAKAI UNTUK ML ───────────────────────────────────────
df = df.drop(columns=['Nama Proyek', 'Lokasi', 'Tahun', 'Keterangan', 'Sumber'])

# ── ENCODING KOLOM KATEGORIKAL ────────────────────────────────────────────────
kolom_kategori = ['Pekerjaan', 'Waktu Shift', 'Cuaca', 'APD', 'Alat Berat']

le = LabelEncoder()
for kolom in kolom_kategori:
    df[kolom] = le.fit_transform(df[kolom])

# Encoding target label
label_map = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
df['Tingkat Risiko'] = df['Tingkat Risiko'].map(label_map)

print("\n=== DATASET SETELAH PREPROCESSING ===")
print(df.head(10))
print(f"\nDistribusi Tingkat Risiko:\n{df['Tingkat Risiko'].value_counts()}")

# ── GENERATE GRAFIK DISTRIBUSI SEKALIAN ───────────────────────────────────────
print("\nMembuat grafik distribusi...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Grafik distribusi target
risk_counts = df['Tingkat Risiko'].value_counts().sort_index()
risk_counts.plot(kind='bar', ax=axes[0], color='tab:blue')
axes[0].set_title('Distribusi Tingkat Risiko')
axes[0].set_xlabel('Tingkat Risiko (0=Rendah, 1=Sedang, 2=Tinggi)')
axes[0].set_ylabel('Jumlah')

# Grafik distribusi beberapa fitur kategorikal hasil encoding
for idx, kolom in enumerate(['Pekerjaan', 'Waktu Shift', 'Cuaca']):
    if idx + 1 < len(axes):
        values = df[kolom].value_counts().sort_index()
        values.plot(kind='bar', ax=axes[idx + 1], color='tab:green')
        axes[idx + 1].set_title(f'Distribusi {kolom}')
        axes[idx + 1].set_xlabel('Nilai Terkodefikasi')
        axes[idx + 1].set_ylabel('Jumlah')

# Hapus sumbu kosong jika ada
for ax in axes:
    if not ax.has_data():
        fig.delaxes(ax)

fig.tight_layout()
plt.savefig('preprocessing_distributions.png', dpi=150)
plt.show()

# ── SIMPAN HASIL PREPROCESSING ────────────────────────────────────────────────
df.to_csv('dataset_preprocessed.csv', index=False)
print("\nFile dataset_preprocessed.csv berhasil disimpan.")