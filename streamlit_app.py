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
    "Papua Barat": 29,
    "Riau": 12,
    "Sulawesi Barat": 28,
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

Klasifikasi_enc = st.sidebar.selectbox("Klasifikasi", [0, 1, 3], index=0)
JenisBelanja_enc = st.sidebar.selectbox("Jenis Belanja", [0, 1], index=0)
MetodePengadaan_enc = st.sidebar.selectbox("Metode Pengadaan", [0, 1], index=0)
KlasBJ_JasaKonsultasi = st.sidebar.selectbox("Jasa Konsultasi?", [0, 1], index=0)
KlasBJ_JasaLainnya = st.sidebar.selectbox("Jasa Lainnya?", [0, 1], index=0)
KlasBJ_PekerjaanKonstruksi = st.sidebar.selectbox("Pekerjaan Konstruksi?", [0, 1], index=0)
KlasBJ_PengadaanBarang = st.sidebar.selectbox("Pengadaan Barang?", [0, 1], index=0)
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
        "Lokasi_enc": [Lokasi_enc_value],
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
