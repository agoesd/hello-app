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
TglPengumuman_Bln = st.sidebar.selectbox("Bulan Pengumuman", range(1, 13), index=0)
Lokasi_enc = st.sidebar.selectbox("Lokasi Pekerjaan", range(0, 37), index=0)
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
