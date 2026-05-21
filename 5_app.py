import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import os

# ── KONFIGURASI HALAMAN ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="K3 Risk Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f2027 100%);
        min-height: 100vh;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }

    .main-header h1 {
        color: #e2f0ff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #7fb3d3;
        font-size: 0.95rem;
        margin: 6px 0 0 0;
        font-weight: 400;
    }

    .badge {
        display: inline-block;
        background: rgba(99,179,237,0.15);
        border: 1px solid rgba(99,179,237,0.3);
        color: #63b3ed;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        letter-spacing: 0.5px;
    }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }

    .card-title {
        color: #a0c4de;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .result-rendah {
        background: linear-gradient(135deg, rgba(72,187,120,0.15) 0%, rgba(56,161,105,0.08) 100%);
        border: 1px solid rgba(72,187,120,0.4);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }

    .result-sedang {
        background: linear-gradient(135deg, rgba(237,137,54,0.15) 0%, rgba(221,107,32,0.08) 100%);
        border: 1px solid rgba(237,137,54,0.4);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }

    .result-tinggi {
        background: linear-gradient(135deg, rgba(245,101,101,0.15) 0%, rgba(229,62,62,0.08) 100%);
        border: 1px solid rgba(245,101,101,0.4);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }

    .result-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .metric-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .metric-label {
        color: #7fb3d3;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #e2f0ff;
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2f0ff !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2f0ff !important;
    }

    label { color: #a0c4de !important; font-weight: 500 !important; font-size: 0.88rem !important; }

    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #1a4a7a 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 12px 24px !important;
        transition: all 0.2s !important;
        letter-spacing: 0.3px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(49,130,206,0.4) !important;
    }

    .sidebar-section {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .history-row {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f2d 0%, #142030 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #7fb3d3;
        font-weight: 600;
        font-size: 0.88rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(49,130,206,0.2) !important;
        color: #63b3ed !important;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    .warning-box {
        background: rgba(237,137,54,0.1);
        border: 1px solid rgba(237,137,54,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fbd38d;
        font-size: 0.88rem;
        margin-top: 8px;
    }

    .success-box {
        background: rgba(72,187,120,0.1);
        border: 1px solid rgba(72,187,120,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #9ae6b4;
        font-size: 0.88rem;
        margin-top: 8px;
    }

    .danger-box {
        background: rgba(245,101,101,0.1);
        border: 1px solid rgba(245,101,101,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #feb2b2;
        font-size: 0.88rem;
        margin-top: 8px;
    }

    h2, h3 { color: #e2f0ff !important; font-weight: 700 !important; }
    p, li { color: #a0c4de !important; }

    .footer {
        text-align: center;
        color: #4a6fa5;
        font-size: 0.78rem;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('k3_database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_proyek TEXT,
            nama_pekerja TEXT,
            tanggal TEXT,
            ketinggian REAL,
            pekerjaan TEXT,
            waktu_shift TEXT,
            cuaca TEXT,
            apd TEXT,
            pengalaman REAL,
            alat_berat TEXT,
            tingkat_risiko TEXT,
            prob_rendah REAL,
            prob_sedang REAL,
            prob_tinggi REAL
        )
    ''')
    conn.commit()
    conn.close()

def simpan_prediksi(data):
    conn = sqlite3.connect('k3_database.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO prediksi (nama_proyek, nama_pekerja, tanggal, ketinggian,
        pekerjaan, waktu_shift, cuaca, apd, pengalaman, alat_berat,
        tingkat_risiko, prob_rendah, prob_sedang, prob_tinggi)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', data)
    conn.commit()
    conn.close()

def load_semua_data():
    conn = sqlite3.connect('k3_database.db')
    df = pd.read_sql_query("SELECT * FROM prediksi ORDER BY id DESC", conn)
    conn.close()
    return df

def hapus_semua_data():
    conn = sqlite3.connect('k3_database.db')
    c = conn.cursor()
    c.execute("DELETE FROM prediksi")
    conn.commit()
    conn.close()

init_db()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model_random_forest.pkl')

rf = load_model()

# ── ENCODING ──────────────────────────────────────────────────────────────────
encoding_map = {
    'Pekerjaan':   {'Bekisting': 0, 'Facade': 1, 'MEP': 2, 'Pembesian': 3, 'Pengecoran': 4},
    'Waktu Shift': {'Lembur Malam': 0, 'Pagi': 1, 'Siang': 2},
    'Cuaca':       {'Angin Kencang': 0, 'Berawan': 1, 'Cerah': 2, 'Hujan': 3},
    'APD':         {'Lengkap': 0, 'Rusak': 1, 'Tidak Lengkap': 2},
    'Alat Berat':  {'Concrete Pump': 0, 'Mobile Crane': 1, 'Passenger Hoist': 2, 'Tower Crane': 3},
}

label_map   = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
risiko_icon = {'Rendah': '✅', 'Sedang': '⚠️', 'Tinggi': '🚨'}
risiko_color= {'Rendah': '#48bb78', 'Sedang': '#ed8936', 'Tinggi': '#f56565'}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.5rem;'>🏗️</div>
        <div style='color:#e2f0ff; font-weight:800; font-size:1.1rem; margin-top:4px;'>K3 Risk Predictor</div>
        <div style='color:#4a6fa5; font-size:0.78rem; margin-top:2px;'>Machine Learning Based</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    nama_proyek = st.text_input("🏢 Nama Proyek", value="Hotel Fairfield Jakarta",
                                 placeholder="Masukkan nama proyek...")

    st.divider()

    menu = st.radio("📌 Navigasi", ["Prediksi Risiko", "Dashboard & Analisis", "Riwayat Data"],
                    label_visibility="collapsed")

    st.divider()

    df_all = load_semua_data()
    total  = len(df_all)
    tinggi = len(df_all[df_all['tingkat_risiko'] == 'Tinggi']) if total > 0 else 0

    st.markdown(f"""
    <div class='sidebar-section'>
        <div style='color:#7fb3d3; font-size:0.72rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;'>📊 Statistik Proyek</div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='color:#a0c4de; font-size:0.85rem;'>Total Data</span>
            <span style='color:#e2f0ff; font-weight:700; font-family:monospace;'>{total}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='color:#a0c4de; font-size:0.85rem;'>Risiko Tinggi</span>
            <span style='color:#f56565; font-weight:700; font-family:monospace;'>{tinggi}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Hapus Semua Data", use_container_width=True):
        hapus_semua_data()
        st.rerun()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='main-header'>
    <div style='margin-bottom:10px;'>
        <span class='badge'>Random Forest</span>
        <span class='badge'>Accuracy 80.23%</span>
        <span class='badge'>K3 Konstruksi</span>
    </div>
    <h1>🏗️ Sistem Prediksi Risiko K3</h1>
    <p>Analisis Risiko Keselamatan Kerja pada Struktur Atas Gedung Bertingkat — {nama_proyek}</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 1: PREDIKSI RISIKO
# ════════════════════════════════════════════════════════════════════════════
if menu == "Prediksi Risiko":

    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown("<div class='card-title'>📋 INPUT KONDISI KERJA</div>", unsafe_allow_html=True)

        nama_pekerja = st.text_input("👷 Nama Pekerja", placeholder="Contoh: Agus Setiawan")
        tanggal      = st.date_input("📅 Tanggal Pekerjaan", value=datetime.today())

        c1, c2 = st.columns(2)
        with c1:
            ketinggian  = st.number_input("📐 Ketinggian (m)", min_value=3.0, max_value=300.0, value=15.0, step=0.5)
            pekerjaan   = st.selectbox("🔨 Jenis Pekerjaan", ['Bekisting', 'Facade', 'MEP', 'Pembesian', 'Pengecoran'])
            waktu_shift = st.selectbox("🕐 Waktu Shift", ['Pagi', 'Siang', 'Lembur Malam'])
            cuaca       = st.selectbox("🌤️ Kondisi Cuaca", ['Cerah', 'Berawan', 'Hujan', 'Angin Kencang'])
        with c2:
            apd         = st.selectbox("🦺 Kelengkapan APD", ['Lengkap', 'Tidak Lengkap', 'Rusak'])
            pengalaman  = st.number_input("📆 Pengalaman Kerja (tahun)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
            alat_berat  = st.selectbox("🏗️ Alat Berat", ['Concrete Pump', 'Mobile Crane', 'Passenger Hoist', 'Tower Crane'])

        st.markdown("<br>", unsafe_allow_html=True)
        prediksi_btn = st.button("🔍 Prediksi Tingkat Risiko", use_container_width=True, type="primary")

    with col_result:
        st.markdown("<div class='card-title'>📊 HASIL PREDIKSI</div>", unsafe_allow_html=True)

        if prediksi_btn:
            input_data = pd.DataFrame([{
                'Ketinggian m':     ketinggian,
                'Pekerjaan':        encoding_map['Pekerjaan'][pekerjaan],
                'Waktu Shift':      encoding_map['Waktu Shift'][waktu_shift],
                'Cuaca':            encoding_map['Cuaca'][cuaca],
                'APD':              encoding_map['APD'][apd],
                'Pengalaman Tahun': pengalaman,
                'Alat Berat':       encoding_map['Alat Berat'][alat_berat],
            }])

            pred  = rf.predict(input_data)[0]
            proba = rf.predict_proba(input_data)[0]
            hasil = label_map[pred]
            icon  = risiko_icon[hasil]
            color = risiko_color[hasil]
            cls   = f"result-{hasil.lower()}"

            st.markdown(f"""
            <div class='{cls}'>
                <div class='result-label' style='color:{color};'>{icon} TINGKAT RISIKO</div>
                <div class='result-value' style='color:{color};'>{hasil.upper()}</div>
                <div style='color:rgba(255,255,255,0.5); font-size:0.82rem; margin-top:8px;'>
                    {nama_pekerja if nama_pekerja else 'Pekerja'} — {pekerjaan} — {tanggal.strftime('%d %b %Y')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c_r, c_s, c_t = st.columns(3)
            with c_r:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-label' style='color:#48bb78;'>Rendah</div>
                    <div class='metric-value' style='color:#48bb78;'>{proba[0]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with c_s:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-label' style='color:#ed8936;'>Sedang</div>
                    <div class='metric-value' style='color:#ed8936;'>{proba[1]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with c_t:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-label' style='color:#f56565;'>Tinggi</div>
                    <div class='metric-value' style='color:#f56565;'>{proba[2]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Rekomendasi
            st.markdown("<br>", unsafe_allow_html=True)
            if hasil == 'Rendah':
                st.markdown("""<div class='success-box'>✅ <b>Kondisi Aman</b> — Pertahankan standar K3 yang ada. Tetap gunakan APD lengkap dan patuhi prosedur keselamatan.</div>""", unsafe_allow_html=True)
            elif hasil == 'Sedang':
                st.markdown("""<div class='warning-box'>⚠️ <b>Perlu Perhatian</b> — Tingkatkan pengawasan. Pastikan APD lengkap, cek kondisi cuaca secara berkala, dan kurangi jam lembur bila memungkinkan.</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='danger-box'>🚨 <b>BAHAYA!</b> — Lakukan tindakan pencegahan segera. Hentikan pekerjaan berisiko, periksa kelengkapan APD, dan pastikan pengawas K3 hadir di lokasi.</div>""", unsafe_allow_html=True)

            # Simpan ke database
            simpan_prediksi((
                nama_proyek, nama_pekerja if nama_pekerja else 'Tidak Diketahui',
                tanggal.strftime('%Y-%m-%d'), ketinggian, pekerjaan,
                waktu_shift, cuaca, apd, pengalaman, alat_berat,
                hasil, round(proba[0]*100,1), round(proba[1]*100,1), round(proba[2]*100,1)
            ))
            st.success("✅ Data berhasil disimpan ke database!")

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#4a6fa5;'>
                <div style='font-size:4rem; margin-bottom:16px;'>🔍</div>
                <div style='font-size:1rem; font-weight:600; color:#7fb3d3;'>Isi form dan klik Prediksi</div>
                <div style='font-size:0.85rem; margin-top:8px;'>Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 2: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Dashboard & Analisis":

    df_all = load_semua_data()

    if len(df_all) == 0:
        st.markdown("""
        <div style='text-align:center; padding:80px; color:#4a6fa5;'>
            <div style='font-size:4rem;'>📊</div>
            <div style='color:#7fb3d3; font-size:1.1rem; font-weight:600; margin-top:16px;'>Belum ada data</div>
            <div style='font-size:0.88rem; margin-top:8px;'>Lakukan prediksi terlebih dahulu untuk melihat dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Metrik utama
        total   = len(df_all)
        rendah  = len(df_all[df_all['tingkat_risiko'] == 'Rendah'])
        sedang  = len(df_all[df_all['tingkat_risiko'] == 'Sedang'])
        tinggi  = len(df_all[df_all['tingkat_risiko'] == 'Tinggi'])

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in zip(
            [c1, c2, c3, c4],
            ['Total Data', 'Risiko Rendah', 'Risiko Sedang', 'Risiko Tinggi'],
            [total, rendah, sedang, tinggi],
            ['#63b3ed', '#48bb78', '#ed8936', '#f56565']
        ):
            with col:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-label' style='color:{color};'>{label}</div>
                    <div class='metric-value' style='color:{color};'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_pie, col_bar = st.columns(2, gap="large")

        # Pie chart distribusi
        with col_pie:
            st.markdown("<div class='card-title'>🥧 DISTRIBUSI RISIKO KESELURUHAN</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
            sizes  = [rendah, sedang, tinggi]
            colors = ['#48bb78', '#ed8936', '#f56565']
            labels = [f'Rendah\n{rendah}', f'Sedang\n{sedang}', f'Tinggi\n{tinggi}']
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': '#1a2d3d', 'linewidth': 2},
                textprops={'color': '#e2f0ff', 'fontsize': 9, 'fontweight': '600'}
            )
            for at in autotexts:
                at.set_color('#0d1f2d')
                at.set_fontweight('800')
                at.set_fontsize(9)
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            st.pyplot(fig)
            plt.close()

        # Bar chart per pekerjaan
        with col_bar:
            st.markdown("<div class='card-title'>📊 RISIKO PER JENIS PEKERJAAN</div>", unsafe_allow_html=True)
            pivot = df_all.groupby(['pekerjaan', 'tingkat_risiko']).size().unstack(fill_value=0)
            for col_name in ['Rendah', 'Sedang', 'Tinggi']:
                if col_name not in pivot.columns:
                    pivot[col_name] = 0
            pivot = pivot[['Rendah', 'Sedang', 'Tinggi']]

            fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
            x     = range(len(pivot))
            width = 0.25
            ax.bar([i - width for i in x], pivot['Rendah'],   width, color='#48bb78', label='Rendah', alpha=0.9)
            ax.bar([i         for i in x], pivot['Sedang'],   width, color='#ed8936', label='Sedang',  alpha=0.9)
            ax.bar([i + width for i in x], pivot['Tinggi'],   width, color='#f56565', label='Tinggi',  alpha=0.9)
            ax.set_xticks(list(x))
            ax.set_xticklabels(pivot.index, rotation=15, ha='right', color='#a0c4de', fontsize=9)
            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2d4a6a')
            ax.spines['bottom'].set_color('#2d4a6a')
            ax.tick_params(colors='#a0c4de')
            ax.yaxis.label.set_color('#a0c4de')
            ax.legend(fontsize=9, labelcolor='#e2f0ff', facecolor='#1a2d3d', edgecolor='#2d4a6a')
            ax.grid(axis='y', color='#2d4a6a', alpha=0.4, linestyle='--')
            fig.patch.set_alpha(0)
            st.pyplot(fig)
            plt.close()

        # Tabel peringkat pekerjaan
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🏆 PERINGKAT PEKERJAAN BERDASARKAN RISIKO TINGGI</div>", unsafe_allow_html=True)

        rank = df_all.groupby('pekerjaan')['tingkat_risiko'].apply(
            lambda x: (x == 'Tinggi').sum()
        ).sort_values(ascending=False).reset_index()
        rank.columns = ['Jenis Pekerjaan', 'Jumlah Risiko Tinggi']
        rank['Total Data'] = df_all.groupby('pekerjaan').size().reindex(rank['Jenis Pekerjaan']).values
        rank['% Risiko Tinggi'] = (rank['Jumlah Risiko Tinggi'] / rank['Total Data'] * 100).round(1).astype(str) + '%'
        rank.index = range(1, len(rank)+1)
        st.dataframe(rank, use_container_width=True)

        # Kesimpulan otomatis
        st.markdown("<br>", unsafe_allow_html=True)
        if len(rank) > 0:
            paling_berisiko = rank.iloc[0]['Jenis Pekerjaan']
            jml_tinggi_max  = rank.iloc[0]['Jumlah Risiko Tinggi']
            pct_tinggi      = f"{tinggi/total*100:.1f}%"
            st.markdown(f"""
            <div class='warning-box' style='padding:16px 20px;'>
                <b>📋 Kesimpulan Otomatis:</b><br>
                Dari <b>{total}</b> data kondisi kerja yang telah dianalisis pada proyek <b>{nama_proyek}</b>,
                sebanyak <b>{tinggi} kondisi ({pct_tinggi})</b> diprediksi berisiko tinggi.
                Jenis pekerjaan <b>{paling_berisiko}</b> menunjukkan jumlah risiko tinggi terbanyak
                yaitu sebanyak <b>{jml_tinggi_max} kasus</b>, sehingga memerlukan perhatian dan
                pengawasan K3 yang lebih ketat.
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 3: RIWAYAT DATA
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Riwayat Data":

    df_all = load_semua_data()

    if len(df_all) == 0:
        st.markdown("""
        <div style='text-align:center; padding:80px; color:#4a6fa5;'>
            <div style='font-size:4rem;'>📂</div>
            <div style='color:#7fb3d3; font-size:1.1rem; font-weight:600; margin-top:16px;'>Belum ada riwayat data</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-title'>📂 RIWAYAT SELURUH PREDIKSI</div>", unsafe_allow_html=True)

        # Filter
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_risiko = st.multiselect("Filter Tingkat Risiko",
                                           ['Rendah', 'Sedang', 'Tinggi'],
                                           default=['Rendah', 'Sedang', 'Tinggi'])
        with col_f2:
            filter_pekerjaan = st.multiselect("Filter Pekerjaan",
                                              df_all['pekerjaan'].unique().tolist(),
                                              default=df_all['pekerjaan'].unique().tolist())

        df_filtered = df_all[
            (df_all['tingkat_risiko'].isin(filter_risiko)) &
            (df_all['pekerjaan'].isin(filter_pekerjaan))
        ]

        # Rename kolom untuk tampilan
        df_show = df_filtered[[
            'id', 'nama_proyek', 'nama_pekerja', 'tanggal', 'pekerjaan',
            'waktu_shift', 'cuaca', 'apd', 'ketinggian', 'pengalaman',
            'alat_berat', 'tingkat_risiko', 'prob_rendah', 'prob_sedang', 'prob_tinggi'
        ]].copy()
        df_show.columns = [
            'ID', 'Proyek', 'Pekerja', 'Tanggal', 'Pekerjaan',
            'Shift', 'Cuaca', 'APD', 'Ketinggian (m)', 'Pengalaman (thn)',
            'Alat Berat', 'Tingkat Risiko', 'P.Rendah (%)', 'P.Sedang (%)', 'P.Tinggi (%)'
        ]

        st.dataframe(df_show, use_container_width=True, height=400)
        st.markdown(f"<p style='font-size:0.82rem; color:#4a6fa5;'>Menampilkan {len(df_filtered)} dari {len(df_all)} data</p>", unsafe_allow_html=True)

        # Download
        csv = df_show.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Data (CSV)",
            data=csv,
            file_name=f"riwayat_prediksi_{nama_proyek.replace(' ','_')}.csv",
            mime='text/csv'
        )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🏗️ K3 Risk Predictor — Dibuat oleh <b>Nauval Rizky</b> | Skripsi Teknik Sipil<br>
    Analisis Risiko K3 pada Struktur Atas Gedung Bertingkat Menggunakan Machine Learning
</div>
""", unsafe_allow_html=True)
