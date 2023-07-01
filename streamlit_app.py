import streamlit as st
import joblib
import pandas as pd

# Set page title and layout
st.set_page_config(
    page_title="Procurement Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set sidebar title and description
st.sidebar.title("Procurement Prediction")
st.sidebar.markdown("Enter the input features below:")

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

Lokasi_enc = st.sidebar.selectbox("Lokasi Pekerjaan", range(0, 37), index=0)
Klasifikasi_enc = st.sidebar.selectbox("Klasifikasi", [0, 1, 3], index=0)
JenisBelanja_enc = st.sidebar.selectbox("Jenis Belanja", [0, 1], index=0)
MetodePengadaan_enc = st.sidebar.selectbox("Metode Pengadaan", [0, 1], index=0)

jenis_pengadaan = st.sidebar.selectbox("Jenis Pengadaan", ["Jasa Konsultasi", "Jasa Lainnya", "Pekerjaan Konstruksi", "Pengadaan Barang"], index=0)

if jenis_pengadaan == "Jasa Konsultasi":
    KlasBJ_JasaKonsultasi = 1
    KlasBJ_JasaLainnya = 0
    KlasBJ_PekerjaanKonstruksi = 0
    KlasBJ_PengadaanBarang = 0
elif jenis_pengadaan == "Jasa Lainnya":
    KlasBJ_JasaKonsultasi = 0
    KlasBJ_JasaLainnya = 1
    KlasBJ_PekerjaanKonstruksi = 0
    KlasBJ_PengadaanBarang = 0
elif jenis_pengadaan == "Pekerjaan Konstruksi":
    KlasBJ_JasaKonsultasi = 0
    KlasBJ_JasaLainnya = 0
    KlasBJ_PekerjaanKonstruksi = 1
    KlasBJ_PengadaanBarang = 0
else:
    KlasBJ_JasaKonsultasi = 0
    KlasBJ_JasaLainnya = 0
    KlasBJ_PekerjaanKonstruksi = 0
    KlasBJ_PengadaanBarang = 1

Pagu2 = st.sidebar.number_input("Nilai Pagu")
HPS2 = st.sidebar.number_input("Nilai HPS")

# Load the Procurement Time model
time_model = joblib.load("dtr_lamatender.joblib")

# Load the Efisiensi model
efisiensi_model = joblib.load("dtr_efisiensi.joblib")

# Procurement time prediction
st.title("Procurement Prediction")
st.markdown("---")

# Make predictions button
if st.button("Make Predictions"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "TglPengumuman_Bln": [TglPengumuman_Bln_value],
        "Lokasi_enc": [Lokasi_enc],
        "Klasifikasi_enc": [Klasifikasi_enc],
        "JenisBelanja_enc": [JenisBelanja_enc],
        "MetodePengadaan_enc": [MetodePengadaan_enc],
        "KlasBJ_JasaKonsultasi": [KlasBJ_JasaKonsultasi],
        "KlasBJ_JasaLainnya": [KlasBJ_JasaLainnya],
        "KlasBJ_PekerjaanKonstruksi": [KlasBJ_PekerjaanKonstruksi],
        "KlasBJ_PengadaanBarang": [KlasBJ_PengadaanBarang],
        "Pagu2": [Pagu2],
        "HPS2": [HPS2]
    })

    # Make Procurement Time prediction
    time_prediction = time_model.predict(input_data)[0]

    # Make Efisiensi prediction
    efisiensi_prediction = efisiensi_model.predict(input_data)[0]

    # Display predictions
    st.success(f"Predicted Procurement Time: {time_prediction}")
    st.success(f"Predicted Efisiensi: {efisiensi_prediction}")
