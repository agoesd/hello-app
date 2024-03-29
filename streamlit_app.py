import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set page title and layout
st.set_page_config(
    page_title="Prediksi Lama Waktu Tender/Seleksi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set sidebar title and description
st.sidebar.title("Estimasi Waktu Tender/Seleksi")
st.sidebar.markdown("Masukan Data Tender/Seleksi:")

# Create input fields for features
bulan_options = {
    "Januari": 0,
    "Februari": 1,
    "Maret": 2,
    "April": 3,
    "Mei": 4,
    "Juni": 5,
    "Juli": 6,
    "Agustus": 7,
    "September": 8,
    "Oktober": 9,
    "November": 10,
    "Desember": 11
}
TglPengumuman_Bln = st.sidebar.selectbox("Bulan Pengumuman", list(bulan_options.keys()), index=0)
TglPengumuman_Bln_value = bulan_options[TglPengumuman_Bln]

lokasi_options = {
    "Aceh": 23,
    "Bali": 13,
    "Banten": 5,
    "Bengkulu": 25,
    "DI Yogyakarta": 9,
    "DKI Jakarta": 0,
    "Jambi": 7,
    "Jawa Barat": 8,
    "Jawa Tengah": 11,
    "Jawa Timur": 1,
    "Kalimantan Selatan": 21,
    "Kalimantan Tengah": 16,
    "Kalimantan Timur": 2,
    "Kalimantan Utara": 19,
    "Kepulauan Riau": 4,
    "Lampung": 17,
    "Maluku": 27,
    "Nusa Tenggara Barat": 26,
    "Nusa Tenggara Timur": 15,
    "Papua": 24,
    "Riau": 12,
    "Sulawesi Selatan": 3,
    "Sulawesi Tengah": 6,
    "Sulawesi Tenggara": 20,
    "Sulawesi Utara": 22,
    "Sumatera Barat": 18,
    "Sumatera Selatan": 14,
    "Sumatera Utara": 10
}
Lokasi_enc = st.sidebar.selectbox("Lokasi Pekerjaan", list(lokasi_options.keys()), index=0)
Lokasi_enc_value = lokasi_options[Lokasi_enc]

klasifikasi_options = {
    "IT": 0,
    "Kendaraan": 11,
    "Konsultansi": 7,
    "Konsultansi Konstruksi": 4,
    "Operasional/TUSI": 1,
    "Pekerjaan Konstruksi": 2,
    "Pemeliharaan Bangunan Gedung": 12,
    "Pemeliharaan Peralatan/Mesin": 3,
    "Penambah Nilai Gedung": 13,
    "Pengadaan Peralatan Mesin": 6,
    "Peralatan dan Perlengkapan Perkantoran": 5,
    "Sewa Fotokopi": 8,
    "Sewa Kendaraan": 10,
    "Sewa operasional lain": 9
}
Klasifikasi_enc = st.sidebar.selectbox("Klasifikasi Barang/Jasa", list(klasifikasi_options.keys()), index=0)
Klasifikasi_enc_value = klasifikasi_options[Klasifikasi_enc]

JenisBelanja_enc = st.sidebar.selectbox("Jenis Belanja", ["Barang", "Modal"], index=0)
JenisBelanja_enc_value = 1 if JenisBelanja_enc == "Modal" else 0

MetodePengadaan_enc = st.sidebar.selectbox("Metode Pengadaan", ["Tender", "Seleksi"], index=0)
MetodePengadaan_enc_value = 1 if MetodePengadaan_enc == "Tender" else 0

# Create input field for Jenis Pengadaan
jenis_pengadaan_options = {
    "Jasa Konsultasi": "KlasBJ_JasaKonsultasi",
    "Jasa Lainnya": "KlasBJ_JasaLainnya",
    "Pekerjaan Konstruksi": "KlasBJ_PekerjaanKonstruksi",
    "Pengadaan Barang": "KlasBJ_PengadaanBarang"
}
jenis_pengadaan = st.sidebar.selectbox("Jenis Pengadaan", list(jenis_pengadaan_options.keys()), index=0)

# Set the value of the selected jenis pengadaan
KlasBJ_JasaKonsultasi = 1 if jenis_pengadaan_options[jenis_pengadaan] == "KlasBJ_JasaKonsultasi" else 0
KlasBJ_JasaLainnya = 1 if jenis_pengadaan_options[jenis_pengadaan] == "KlasBJ_JasaLainnya" else 0
KlasBJ_PekerjaanKonstruksi = 1 if jenis_pengadaan_options[jenis_pengadaan] == "KlasBJ_PekerjaanKonstruksi" else 0
KlasBJ_PengadaanBarang = 1 if jenis_pengadaan_options[jenis_pengadaan] == "KlasBJ_PengadaanBarang" else 0

def format_thousands_separator(value):
    return "{:,.0f}".format(value)

Pagu2_raw = st.sidebar.number_input("Nilai Pagu", value=0, step=1000)
Pagu2 = Pagu2_raw / 1000
formatted_Pagu2 = format_thousands_separator(Pagu2_raw)
st.sidebar.write(f"**{formatted_Pagu2.replace(',', '.')}**")

HPS2_raw = st.sidebar.number_input("Nilai HPS", value=0, step=1000)
HPS2 = HPS2_raw / 1000
formatted_HPS2 = format_thousands_separator(HPS2_raw)
st.sidebar.write(f"**{formatted_HPS2.replace(',', '.')}**")

# Model prediksi waktu tender/seleksi
time_model = joblib.load("dtr_lamatender.joblib")

# Model Efisiensi
efisiensi_model = joblib.load("dtr_efisiensi.joblib")

# Judul
st.title("Prediksi Lama Waktu Tender/Seleksi")
st.markdown("---")

# Tombol aksi
if st.button("Kalkulasi"):
    # Olah nilai input Pagu dan HPS, konversi dengan StandarScaler sesuai format model
    # Generate 1000 random data points
    random_values = np.random.randint(low=HPS2, high=Pagu2, size=(1000, 2))
    
    # Create a sample DataFrame
    data = pd.DataFrame({
        'Pagu2': [Pagu2] + list(random_values[:, 0]),
        'HPS2': [HPS2] + list(random_values[:, 1])
    })
    
    # Create a StandardScaler object
    scaler = StandardScaler()
    
    # Perform scaling
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    # Fetch the first row of scaled data from each column
    first_row_scaled = scaled_data.iloc[0]
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "TglPengumuman_Bln": [TglPengumuman_Bln_value],
        "Lokasi_enc": [Lokasi_enc_value],
        "Klasifikasi_enc": [Klasifikasi_enc_value],
        "JenisBelanja_enc": [JenisBelanja_enc_value],
        "MetodePengadaan_enc": [MetodePengadaan_enc_value],
        "KlasBJ_JasaKonsultasi": [KlasBJ_JasaKonsultasi],
        "KlasBJ_JasaLainnya": [KlasBJ_JasaLainnya],
        "KlasBJ_PekerjaanKonstruksi": [KlasBJ_PekerjaanKonstruksi],
        "KlasBJ_PengadaanBarang": [KlasBJ_PengadaanBarang],
        "Pagu2": [float(f"{first_row_scaled['Pagu2']:.6f}")],
        "HPS2": [float(f"{first_row_scaled['HPS2']:.6f}")]
    })

    # Prediksi lama tender/seleksi
    time_prediction = int(time_model.predict(input_data)[0])

    # Prediksi Efisiensi
    efisiensi_prediction = efisiensi_model.predict(input_data)[0] * 100

    # Display predictions
    st.success(f"Estimasi Waktu Tender/Seleksi: {time_prediction} hari kerja")
    st.success(f"Estimasi Efisiensi: {efisiensi_prediction:.2f}%")
