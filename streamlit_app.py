import streamlit as st
import pandas as pd
import joblib

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
TglPengumuman_Bln = st.sidebar.number_input("Bulan Pengumuman")
Lokasi_enc = st.sidebar.number_input("Lokasi Pekerjaan")
Klasifikasi_enc = st.sidebar.number_input("Klasifikasi")
JenisBelanja_enc = st.sidebar.number_input("Jenis Belanja")
MetodePengadaan_enc = st.sidebar.number_input("Metode Pengadaan")
KlasBJ_JasaKonsultasi = st.sidebar.number_input("Jasa Konsultasi?")
KlasBJ_JasaLainnya = st.sidebar.number_input("Jasa Lainnya?")
KlasBJ_PekerjaanKonstruksi = st.sidebar.number_input("Pekerjaan Konstruksi?")
KlasBJ_PengadaanBarang = st.sidebar.number_input("Pengadaan Barang?")
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
    # Make Procurement Time prediction
    time_prediction = time_model.predict([[TglPengumuman_Bln, Lokasi_enc, Klasifikasi_enc, JenisBelanja_enc, MetodePengadaan_enc, KlasBJ_JasaKonsultasi, KlasBJ_JasaLainnya, KlasBJ_PekerjaanKonstruksi, KlasBJ_PengadaanBarang, Pagu2, HPS2]])[0]

    # Make Efisiensi prediction
    efisiensi_prediction = efisiensi_model.predict([[TglPengumuman_Bln, Lokasi_enc, Klasifikasi_enc, JenisBelanja_enc, MetodePengadaan_enc, KlasBJ_JasaKonsultasi, KlasBJ_JasaLainnya, KlasBJ_PekerjaanKonstruksi, KlasBJ_PengadaanBarang, Pagu2, HPS2]])[0]

    # Display predictions
    st.success(f"Predicted Procurement Time: {time_prediction}")
    st.success(f"Predicted Efisiensi: {efisiensi_prediction}")
