import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
TglPengumuman_Bln = st.sidebar.selectbox("Bulan Pengumuman", range(1, 13), index=0)
Lokasi_enc = st.sidebar.selectbox("Lokasi Pekerjaan", range(0, 37), index=0)
Klasifikasi_enc = st.sidebar.selectbox("Klasifikasi Barang/Jasa", ["Jasa Konsultansi", "Jasa Lainnya", "Pekerjaan Konstruksi", "Pengadaan Barang"], index=0)

# Map Klasifikasi_enc to corresponding feature values
KlasBJ_JasaKonsultasi = 1 if Klasifikasi_enc == "Jasa Konsultansi" else 0
KlasBJ_JasaLainnya = 1 if Klasifikasi_enc == "Jasa Lainnya" else 0
KlasBJ_PekerjaanKonstruksi = 1 if Klasifikasi_enc == "Pekerjaan Konstruksi" else 0
KlasBJ_PengadaanBarang = 1 if Klasifikasi_enc == "Pengadaan Barang" else 0

JenisBelanja_enc = st.sidebar.selectbox("Jenis Belanja", [0, 1], index=0)
MetodePengadaan_enc = st.sidebar.selectbox("Metode Pengadaan", [0, 1], index=0)
Pagu2 = st.sidebar.number_input("Nilai Pagu")
HPS2 = st.sidebar.number_input("Nilai HPS")

# Load the Procurement Time model
time_model = joblib.load("modeldecisiontree_lamatender.joblib")

# Load the Efisiensi model
efisiensi_model = joblib.load("modeldecisiontree_efisiensi.joblib")

# Procurement time prediction
st.title("Procurement Prediction")
st.markdown("---")

# Make predictions button
if st.button("Make Predictions"):
    # Prepare input data for prediction
    input_data = np.array([[TglPengumuman_Bln, Lokasi_enc, KlasBJ_JasaKonsultasi, KlasBJ_JasaLainnya, KlasBJ_PekerjaanKonstruksi, KlasBJ_PengadaanBarang, JenisBelanja_enc, MetodePengadaan_enc, Pagu2, HPS2]])

    # Reshape input data
    input_data = np.reshape(input_data, (1, -1))

    # Ensure the input data is of the correct data type
    input_data = input_data.astype(np.float64)

    # Make Procurement Time prediction
    time_prediction = time_model.predict(input_data)[0]

    # Make Efisiensi prediction
    efisiensi_prediction = efisiensi_model.predict(input_data)[0]

    # Display predictions
    st.success(f"Predicted Procurement Time: {time_prediction}")
    st.success(f"Predicted Efisiensi: {efisiensi_prediction}")
