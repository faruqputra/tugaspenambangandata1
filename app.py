import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model_dt.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Risiko Osteoporosis")

# ========== INPUT ==========
usia = st.slider("Usia (tahun)", 18, 100, 30)

jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
perubahan_hormon = st.selectbox("Perubahan Hormon", ["Normal", "Pasca menopause"])
riwayat_keluarga = st.selectbox("Riwayat Keluarga Osteoporosis", ["Tidak", "Ya"])
ras = st.selectbox("Ras / Etnis", ["Asia", "Kaukasia", "Afrika-Amerika"])
berat_badan = st.selectbox("Berat Badan", ["Normal", "Kurus"])
kalsium = st.selectbox("Asupan Kalsium", ["Cukup", "Rendah"])
vitamin_d = st.selectbox("Asupan Vitamin D", ["Cukup", "Tidak cukup"])
aktivitas = st.selectbox("Aktivitas Fisik", ["Aktif", "Kurang aktif"])
merokok = st.selectbox("Merokok", ["Tidak", "Ya"])
alkohol = st.selectbox("Konsumsi Alkohol", ["Tidak", "Sedang"])
kondisi_medis = st.selectbox("Kondisi Medis", ["Gangguan tiroid", "Tidak ada", "Radang sendi"])
obat = st.selectbox("Obat-obatan", ["Kortikosteroid", "Tidak ada"])
patah = st.selectbox("Riwayat Patah Tulang", ["Tidak", "Ya"])

# ========== MAPPING ==========
map_jk = {"Perempuan":0,"Laki-laki":1}
map_hormon = {"Normal":0,"Pasca menopause":1}
map_keluarga = {"Tidak":0,"Ya":1}
map_ras = {"Afrika-Amerika":0,"Asia":1,"Kaukasia":2}
map_berat = {"Normal":0,"Kurus":1}
map_kalsium = {"Cukup":0,"Rendah":1}
map_vitd = {"Tidak cukup":0,"Cukup":1}
map_aktivitas = {"Aktif":0,"Kurang aktif":1}
map_rokok = {"Tidak":0,"Ya":1}
map_alkohol = {"Tidak":0,"Sedang":1}
map_medis = {"Gangguan tiroid":0,"Tidak ada":1,"Radang sendi":2}
map_obat = {"Kortikosteroid":0,"Tidak ada":1}
map_patah = {"Tidak":0,"Ya":1}

# ========== DATA ==========
data = np.array([[
    usia,
    map_jk[jenis_kelamin],
    map_hormon[perubahan_hormon],
    map_keluarga[riwayat_keluarga],
    map_ras[ras],
    map_berat[berat_badan],
    map_kalsium[kalsium],
    map_vitd[vitamin_d],
    map_aktivitas[aktivitas],
    map_rokok[merokok],
    map_alkohol[alkohol],
    map_medis[kondisi_medis],
    map_obat[obat],
    map_patah[patah]
]])

data_scaled = scaler.transform(data)

# ========== PREDIKSI ==========
if st.button("Prediksi"):
    hasil = model.predict(data_scaled)
    if hasil[0] == 1:
        st.error("⚠️ Pasien Berisiko Mengalami Osteoporosis")
    else:
        st.success("✅ Pasien Tidak Berisiko Osteoporosis")
