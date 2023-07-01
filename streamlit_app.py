import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Set page title
st.set_page_config(page_title="Procurement Time Prediction App", layout="wide")

# Set sidebar
st.sidebar.title("Procurement Time Prediction")

# Create input fields for features
TglPengumuman_Bln = st.sidebar.number_input("Bulan Pengumuman")
Lokasi_enc = st.sidebar.number_input("Lokasi Pekerjaan")
Klasifikasi_enc = st.sidebar.number_input("Klasifikasi")
JenisBelanja_enc = st.sidebar.number_input("Jenis Belanja")
MetodePengadaan_enc = st.sidebar.number_input("Metode Pengadaan")
KlasBJ_JasaKonsultasi = st.sidebar.number_input("Jasa Konsultasi?")
KlasBJ_JasaLainnya  = st.sidebar.number_input("Jasa Lainnya?")
KlasBJ_PekerjaanKonstruksi = st.sidebar.number_input("Pekerjaan Konstruksi?")
KlasBJ_PengadaanBarang = st.sidebar.number_input("Pengadaan Barang?")
Pagu2 = st.sidebar.number_input("Nilai Pagu")
HPS2 = st.sidebar.number_input("Nilai HPS")
# Add more feature inputs if needed

# Load the pickle model
with open("modeldecisiontree_lamatender.joblib", "rb") as file:
    model = pickle.load(file)

# Procurement time prediction
st.subheader("Procurement Time Prediction")

# Make prediction
prediction = model.predict([[feature1, feature2]])[0]

# Display prediction
st.write("Predicted Procurement Time:", prediction)
