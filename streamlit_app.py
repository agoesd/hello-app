import streamlit as st
import pandas as pd
import joblib

# Set page title and layout
st.set_page_config(
    page_title="Procurement Time Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set sidebar title and description
st.sidebar.title("Procurement Time Prediction")
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

# Load the joblib model
model = joblib.load("modeldecisiontree_lamatender.joblib")

# Procurement time prediction
st.title("Procurement Time Prediction")
st.markdown("---")

# Make prediction button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict([[TglPengumuman_Bln, Lokasi_enc, Klasifikasi_enc, JenisBelanja_enc, MetodePengadaan_enc, KlasBJ_JasaKonsultasi, KlasBJ_JasaLainnya, KlasBJ_PekerjaanKonstruksi, KlasBJ_PengadaanBarang, Pagu2, HPS2]])[0]

    # Display prediction
    st.success(f"Predicted Procurement Time: {prediction}")
