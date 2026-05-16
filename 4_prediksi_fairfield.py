import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
rf = joblib.load('model_random_forest.pkl')
print("Model Random Forest berhasil diload.")

# ── BACA DATA FAIRFIELD ───────────────────────────────────────────────────────
df = pd.read_excel('dataset_testing_Fairfield.xlsx', sheet_name='Dataset Testing Fairfield')
print(f"\nTotal data Fairfield: {len(df)} baris")

# ── SIMPAN KOLOM INFO (tidak dipakai ML) ─────────────────────────────────────
df_info = df[['Nama Proyek', 'Lokasi', 'Ketinggian m', 'Pekerjaan',
              'Waktu Shift', 'Cuaca', 'APD', 'Pengalaman Tahun', 'Alat Berat']].copy()

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
df_ml = df[['Ketinggian m', 'Pekerjaan', 'Waktu Shift', 'Cuaca',
            'APD', 'Pengalaman Tahun', 'Alat Berat']].copy()

kolom_kategori = ['Pekerjaan', 'Waktu Shift', 'Cuaca', 'APD', 'Alat Berat']

# Encoding harus sama persis dengan training
encoding_map = {
    'Pekerjaan':        {'Bekisting': 0, 'Facade': 1, 'MEP': 2, 'Pembesian': 3, 'Pengecoran': 4},
    'Waktu Shift':      {'Lembur Malam': 0, 'Pagi': 1, 'Siang': 2},
    'Cuaca':            {'Angin Kencang': 0, 'Berawan': 1, 'Cerah': 2, 'Hujan': 3},
    'APD':              {'Lengkap': 0, 'Rusak': 1, 'Tidak Lengkap': 2},
    'Alat Berat':       {'Concrete Pump': 0, 'Mobile Crane': 1, 'Passenger Hoist': 2, 'Tower Crane': 3},
}

for kolom, mapping in encoding_map.items():
    df_ml[kolom] = df_ml[kolom].map(mapping)

print(f"\nCek missing value setelah encoding:\n{df_ml.isnull().sum()}")

# ── PREDIKSI ──────────────────────────────────────────────────────────────────
y_pred = rf.predict(df_ml)
y_prob  = rf.predict_proba(df_ml)

label_map    = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
df_info      = df_info.copy()
df_info['Tingkat Risiko']      = [label_map[p] for p in y_pred]
df_info['Probabilitas Rendah'] = (y_prob[:, 0] * 100).round(1)
df_info['Probabilitas Sedang'] = (y_prob[:, 1] * 100).round(1)
df_info['Probabilitas Tinggi'] = (y_prob[:, 2] * 100).round(1)

# ── TAMPILKAN HASIL ───────────────────────────────────────────────────────────
print("\n=== HASIL PREDIKSI HOTEL FAIRFIELD JAKARTA ===")
print(f"\nDistribusi Tingkat Risiko:")
dist = df_info['Tingkat Risiko'].value_counts()
for label in ['Rendah', 'Sedang', 'Tinggi']:
    count = dist.get(label, 0)
    pct   = count / len(df_info) * 100
    print(f"  {label:8s}: {count:3d} data ({pct:.1f}%)")

print(f"\nSample 10 hasil prediksi:")
print(df_info[['Pekerjaan','Waktu Shift','Cuaca','APD',
               'Ketinggian m','Tingkat Risiko',
               'Probabilitas Rendah','Probabilitas Sedang',
               'Probabilitas Tinggi']].head(10).to_string())

# ── VISUALISASI DISTRIBUSI PREDIKSI ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
colors = ['#C6EFCE', '#FFEB9C', '#FFC7CE']
counts = [dist.get('Rendah', 0), dist.get('Sedang', 0), dist.get('Tinggi', 0)]
axes[0].pie(counts, labels=['Rendah', 'Sedang', 'Tinggi'],
            colors=colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11})
axes[0].set_title('Distribusi Tingkat Risiko\nHotel Fairfield Jakarta',
                  fontsize=12, fontweight='bold')

# Bar chart per pekerjaan
pivot = df_info.groupby(['Pekerjaan', 'Tingkat Risiko']).size().unstack(fill_value=0)
pivot = pivot.reindex(columns=['Rendah', 'Sedang', 'Tinggi'], fill_value=0)
pivot.plot(kind='bar', ax=axes[1],
           color=['#C6EFCE', '#FFEB9C', '#FFC7CE'],
           edgecolor='black', linewidth=0.5)
axes[1].set_title('Distribusi Risiko per Jenis Pekerjaan\nHotel Fairfield Jakarta',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Jenis Pekerjaan', fontsize=10)
axes[1].set_ylabel('Jumlah Data', fontsize=10)
axes[1].legend(title='Tingkat Risiko')
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('prediksi_fairfield.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrafik disimpan: prediksi_fairfield.png")

# ── SIMPAN HASIL KE EXCEL ─────────────────────────────────────────────────────
df_info.to_excel('hasil_prediksi_fairfield.xlsx', index=False)
print("Hasil disimpan: hasil_prediksi_fairfield.xlsx")
print("\nSelesai.")