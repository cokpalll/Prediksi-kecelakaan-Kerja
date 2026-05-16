import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model_random_forest.pkl')

rf = load_model()

# ── KONFIGURASI HALAMAN ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Risiko K3 Konstruksi",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Prediksi Risiko K3 Konstruksi")
st.subheader("Analisis Risiko Keselamatan Kerja pada Struktur Atas Gedung Bertingkat")
st.markdown("**Metode:** Random Forest | **Studi Kasus:** Hotel Fairfield Jakarta")
st.divider()

# ── FORM INPUT ────────────────────────────────────────────────────────────────
st.header("📋 Input Kondisi Kerja")

col1, col2 = st.columns(2)

with col1:
    ketinggian = st.number_input(
        "Ketinggian Pekerjaan (meter)",
        min_value=3.0, max_value=300.0, value=15.0, step=0.5
    )
    pekerjaan = st.selectbox(
        "Jenis Pekerjaan",
        ['Bekisting', 'Facade', 'MEP', 'Pembesian', 'Pengecoran']
    )
    waktu_shift = st.selectbox(
        "Waktu Shift",
        ['Pagi', 'Siang', 'Lembur Malam']
    )
    cuaca = st.selectbox(
        "Kondisi Cuaca",
        ['Cerah', 'Berawan', 'Hujan', 'Angin Kencang']
    )

with col2:
    apd = st.selectbox(
        "Kelengkapan APD",
        ['Lengkap', 'Tidak Lengkap', 'Rusak']
    )
    pengalaman = st.number_input(
        "Pengalaman Kerja (tahun)",
        min_value=0.5, max_value=30.0, value=5.0, step=0.5
    )
    alat_berat = st.selectbox(
        "Alat Berat yang Digunakan",
        ['Concrete Pump', 'Mobile Crane', 'Passenger Hoist', 'Tower Crane']
    )

st.divider()

# ── PREDIKSI ──────────────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Tingkat Risiko", use_container_width=True, type="primary"):

    encoding_map = {
        'Pekerjaan':   {'Bekisting': 0, 'Facade': 1, 'MEP': 2, 'Pembesian': 3, 'Pengecoran': 4},
        'Waktu Shift': {'Lembur Malam': 0, 'Pagi': 1, 'Siang': 2},
        'Cuaca':       {'Angin Kencang': 0, 'Berawan': 1, 'Cerah': 2, 'Hujan': 3},
        'APD':         {'Lengkap': 0, 'Rusak': 1, 'Tidak Lengkap': 2},
        'Alat Berat':  {'Concrete Pump': 0, 'Mobile Crane': 1, 'Passenger Hoist': 2, 'Tower Crane': 3},
    }

    input_data = pd.DataFrame([{
        'Ketinggian m':     ketinggian,
        'Pekerjaan':        encoding_map['Pekerjaan'][pekerjaan],
        'Waktu Shift':      encoding_map['Waktu Shift'][waktu_shift],
        'Cuaca':            encoding_map['Cuaca'][cuaca],
        'APD':              encoding_map['APD'][apd],
        'Pengalaman Tahun': pengalaman,
        'Alat Berat':       encoding_map['Alat Berat'][alat_berat],
    }])

    prediksi    = rf.predict(input_data)[0]
    probabilitas = rf.predict_proba(input_data)[0]
    label_map   = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    hasil       = label_map[prediksi]

    st.header("📊 Hasil Prediksi")

    if hasil == 'Rendah':
        st.success(f"## ✅ Tingkat Risiko: RENDAH")
        st.info("Kondisi kerja relatif aman. Tetap pertahankan standar K3 yang ada.")
    elif hasil == 'Sedang':
        st.warning(f"## ⚠️ Tingkat Risiko: SEDANG")
        st.info("Perlu peningkatan pengawasan. Pastikan semua APD digunakan dengan benar.")
    else:
        st.error(f"## 🚨 Tingkat Risiko: TINGGI")
        st.info("Kondisi berbahaya! Segera lakukan tindakan pencegahan sebelum pekerjaan dilanjutkan.")

    st.divider()

    # Probabilitas
    st.subheader("📈 Probabilitas Tiap Kelas")
    col_r, col_s, col_t = st.columns(3)
    col_r.metric("Rendah",  f"{probabilitas[0]*100:.1f}%")
    col_s.metric("Sedang",  f"{probabilitas[1]*100:.1f}%")
    col_t.metric("Tinggi",  f"{probabilitas[2]*100:.1f}%")

    # Grafik probabilitas
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#4CAF50', '#FF9800', '#F44336']
    bars = ax.bar(['Rendah', 'Sedang', 'Tinggi'],
                  [p*100 for p in probabilitas],
                  color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, probabilitas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probabilitas (%)')
    ax.set_title('Probabilitas Tingkat Risiko')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("📝 Ringkasan Input")
    st.table(pd.DataFrame({
        'Variabel': ['Ketinggian', 'Pekerjaan', 'Waktu Shift', 'Cuaca',
                     'APD', 'Pengalaman', 'Alat Berat'],
        'Nilai': [f'{ketinggian} m', pekerjaan, waktu_shift, cuaca,
                  apd, f'{pengalaman} tahun', alat_berat]
    }))

st.divider()
st.caption("Dibuat oleh Nauval Rizky | Skripsi Teknik Sipil | Analisis Risiko K3 Menggunakan Machine Learning")