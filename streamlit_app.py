import streamlit as st
import joblib
import pandas as pd
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
Klasifikasi_enc = st.sidebar.selectbox("Klasifikasi", list(klasifikasi_options.keys()), index=0)
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

Pagu2 = st.sidebar.number_input("Nilai Pagu (dalam ribuan)", value=0, format="%d", step=1000)
HPS2 = st.sidebar.number_input("Nilai HPS (dalam ribuan)", value=0, format="%d", step=1000)

# Create a StandardScaler object
scaler = StandardScaler()

# Create a DataFrame for scaling
data_scaling = pd.DataFrame({
    'Pagu2': [Pagu2],
    'HPS2': [HPS2]
})

# Fit and transform the data using the scaler
#scaled_data = scaler.fit_transform(data_scaling)

# Create a new DataFrame with scaled values
#datascaling = pd.DataFrame(scaled_data, columns=['Pagu2', 'HPS2'])

#datascaling = pd.DataFrame(scaler.fit_transform(data_scaling[['Pagu2','HPS2']]),columns=['Pagu2','HPS2'])
datascaling = pd.DataFrame(scaler.fit_transform(data_scaling), columns=data_scaling.columns)

# Create a sample DataFrame
data = pd.DataFrame({
    'Pagu2': [10000000, 5000000, 3000000],
    'HPS2': [9999000, 4999000, 2999000]
})

# Create a StandardScaler object
scaler = StandardScaler()

# Perform scaling
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Display the original and scaled data
st.write("Original Data:")
st.write(data)
st.write("\nScaled Data:")
st.write(scaled_data)


# Model prediksi waktu tender/seleksi
time_model = joblib.load("dtr_lamatender.joblib")

# Model Efisiensi
efisiensi_model = joblib.load("dtr_efisiensi.joblib")

# Judul
st.title("Prediksi Lama Waktu Tender/Seleksi")
st.markdown("---")

# Tombol aksi
if st.button("Kalkulasi"):
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
        "Pagu2": [float(f"{datascaling['Pagu2'][0]:.6f}")],
        "HPS2": [float(f"{datascaling['HPS2'][0]:.6f}")]
    })

    # Prediksi lama tender/seleksi
    time_prediction = int(time_model.predict(input_data)[0])

    # Prediksi Efisiensi
    efisiensi_prediction = efisiensi_model.predict(input_data)[0] * 100

    # Display predictions
    st.success(f"Estimasi Waktu Tender/Seleksi: {time_prediction} hari kerja")
    st.success(f"Estimasi Efisiensi: {efisiensi_prediction:.2f}%")

    st.write(input_data)
    st.write(data_scaling)
    #st.write(scaled_data)
    st.write(datascaling)
