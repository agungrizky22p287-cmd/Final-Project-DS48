import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import gdown # Library untuk mengunduh dari Google Drive

# --- 1. Memuat Model dan Scaler ---
# Model akan diunduh dari Google Drive saat aplikasi dimulai
model_url = "https://drive.google.com/uc?id=1KGeTRFZdpZNFrMh7wEwGkjvzaKFe2dyS&export=download" # Link Model dari Google Drive Anda
scaler_url = "https://drive.google.com/uc?id=1cviaP89l1BFvFe50JH9tYT4DhEITcBeQ&export=download" # Link Scaler dari Google Drive Anda

# Nama file lokal setelah diunduh (ini akan disimpan di direktori yang sama dengan app.py)
model_filename = 'random_forest_model.joblib'
scaler_filename = 'standard_scaler.joblib'

try:
    st.write("Mengunduh model dan scaler dari Google Drive...")
    gdown.download(model_url, model_filename, quiet=False, fuzzy=True) # fuzzy=True untuk handle perubahan nama file di Drive
    gdown.download(scaler_url, scaler_filename, quiet=False, fuzzy=True)

    # Muat model dan scaler setelah diunduh
    rf_model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    st.success("Model dan Scaler berhasil dimuat!")
except Exception as e:
    st.error(f"Error memuat model atau scaler: {e}. Pastikan link Drive benar dan file dapat diakses publik.")
    st.stop() # Menghentikan aplikasi jika ada masalah


# --- 2. Judul Aplikasi ---
st.title("Prediksi Hujan Besok di Australia")
st.write("Masukkan parameter cuaca untuk memprediksi apakah besok akan hujan.")

# --- 3. Input Pengguna ---
st.header("Input Data Cuaca Hari Ini")

input_data = {}

today = datetime.date.today()
input_date = st.date_input("Tanggal Hari Ini", value=today)
input_data['Year'] = input_date.year
input_data['Month'] = input_date.month
input_data['Day'] = input_date.day
input_data['Weekday'] = input_date.weekday()

rain_today_map = {"Tidak": 0, "Ya": 1}
selected_rain_today = st.selectbox("Apakah hari ini hujan?", ["Tidak", "Ya"])
input_data['RainToday'] = rain_today_map[selected_rain_today]


input_data['MinTemp'] = st.number_input("Suhu Minimum (°C)", min_value=-10.0, max_value=50.0, value=10.0)
input_data['MaxTemp'] = st.number_input("Suhu Maksimum (°C)", min_value=-10.0, max_value=50.0, value=20.0)
input_data['Rainfall'] = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=400.0, value=0.0)
input_data['WindGustSpeed'] = st.number_input("Kecepatan Hembusan Angin (km/h)", min_value=0.0, max_value=150.0, value=40.0)
input_data['WindSpeed9am'] = st.number_input("Kecepatan Angin Jam 9 Pagi (km/h)", min_value=0.0, max_value=100.0, value=20.0)
input_data['WindSpeed3pm'] = st.number_input("Kecepatan Angin Jam 3 Sore (km/h)", min_value=0.0, max_value=100.0, value=20.0)
input_data['Humidity9am'] = st.number_input("Kelembaban Jam 9 Pagi (%)", min_value=0, max_value=100, value=70)
input_data['Humidity3pm'] = st.number_input("Kelembaban Jam 3 Sore (%)", min_value=0, max_value=100, value=50)
input_data['Pressure9am'] = st.number_input("Tekanan Udara Jam 9 Pagi (hPa)", min_value=980.0, max_value=1050.0, value=1010.0)
input_data['Pressure3pm'] = st.number_input("Tekanan Udara Jam 3 Sore (hPa)", min_value=970.0, max_value=1040.0, value=1005.0)
input_data['Temp9am'] = st.number_input("Suhu Jam 9 Pagi (°C)", min_value=-10.0, max_value=50.0, value=15.0)
input_data['Temp3pm'] = st.number_input("Suhu Jam 3 Sore (°C)", min_value=-10.0, max_value=50.0, value=25.0)


# Input untuk Kolom Kategorikal (One-Hot Encoded) - Location, WindGustDir, WindDir9am, WindDir3pm
# Daftar lengkap kategori unik yang kamu berikan:
all_locations = [
    'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
    'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
    'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong',
    'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo',
    'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil',
    'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
    'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
    'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
    'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
    'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
]

all_wind_dirs = [
    'W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
    'S', 'NW', 'SE', 'ESE', 'E', 'SSW'
]

all_wind_dirs_9am = [
    'W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'SSW', 'N',
    'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'
]

all_wind_dirs_3pm = [
    'WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
    'SW', 'SE', 'N', 'S', 'NNE', 'NE'
]


selected_location = st.selectbox("Lokasi", sorted(all_locations))
selected_wind_gust_dir = st.selectbox("Arah Hembusan Angin (WindGustDir)", sorted(all_wind_dirs))
selected_wind_dir_9am = st.selectbox("Arah Angin Jam 9 Pagi (WindDir9am)", sorted(all_wind_dirs_9am))
selected_wind_dir_3pm = st.selectbox("Arah Angin Jam 3 Sore (WindDir3pm)", sorted(all_wind_dirs_3pm))

# --- 4. Pre-processing Input Pengguna (Penting!) ---

# Urutan Kolom Input Model (X_train_cols_order)
X_train_cols_order = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
    'RainToday', 'Year', 'Month', 'Day', 'Weekday',
    'Location_Albany', 'Location_Albury', 'Location_AliceSprings', 'Location_BadgerysCreek', 'Location_Ballarat', 'Location_Bendigo', 'Location_Brisbane', 'Location_Cairns', 'Location_Canberra', 'Location_Cobar', 'Location_CoffsHarbour', 'Location_Dartmoor', 'Location_Darwin', 'Location_GoldCoast', 'Location_Hobart', 'Location_Katherine', 'Location_Launceston', 'Location_Melbourne', 'Location_MelbourneAirport', 'Location_Mildura', 'Location_Moree', 'Location_MountGambier', 'Location_MountGinini', 'Location_Newcastle', 'Location_Nhil', 'Location_NorahHead', 'Location_NorfolkIsland', 'Location_Nuriootpa', 'Location_PearceRAAF', 'Location_Penrith', 'Location_Perth', 'Location_PerthAirport', 'Location_Portland', 'Location_Richmond', 'Location_Sale', 'Location_SalmonGums', 'Location_Sydney', 'Location_SydneyAirport', 'Location_Townsville', 'Location_Tuggeranong', 'Location_Uluru', 'Location_WaggaWagga', 'Location_Walpole', 'Location_Watsonia', 'Location_Williamtown', 'Location_Witchcliffe', 'Location_Wollongong', 'Location_Woomera',
    'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N', 'WindGustDir_NE', 'WindGustDir_NNE', 'WindGustDir_NNW', 'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE', 'WindGustDir_SSE', 'WindGustDir_SSW', 'WindGustDir_SW', 'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW',
    'WindDir9am_ENE', 'WindDir9am_ESE', 'WindDir9am_N', 'WindDir9am_NE', 'WindDir9am_NNE', 'WindDir9am_NNW', 'WindDir9am_NW', 'WindDir9am_S', 'WindDir9am_SE', 'WindDir9am_SSE', 'WindDir9am_SSW', 'WindDir9am_SW', 'WindDir9am_W', 'WindDir9am_WNW', 'WindDir9am_WSW',
    'WindDir3pm_ENE', 'WindDir3pm_ESE', 'WindDir3pm_N', 'WindDir3pm_NE', 'WindDir3pm_NNE', 'WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_S', 'WindDir3pm_SE', 'WindDir3pm_SSE', 'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W', 'WindDir3pm_WNW', 'WindDir3pm_WSW'
]

# Kolom Numerik yang Diskalakan (numeric_cols_to_scale_for_app)
numeric_cols_to_scale_for_app = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
    'RainToday', 'Year', 'Month', 'Day', 'Weekday'
]

# Inisialisasi DataFrame input untuk prediksi dengan semua kolom yang diharapkan, diisi nol
processed_input = pd.DataFrame(0, index=[0], columns=X_train_cols_order)

# Isi nilai numerik langsung dari input_data
for col in input_data:
    if col in processed_input.columns:
        processed_input[col] = input_data[col]

# Isi nilai one-hot encoded (set 1 untuk kategori yang dipilih)
if f'Location_{selected_location}' in processed_input.columns:
    processed_input[f'Location_{selected_location}'] = 1
if f'WindGustDir_{selected_wind_gust_dir}' in processed_input.columns:
    processed_input[f'WindGustDir_{selected_wind_gust_dir}'] = 1
if f'WindDir9am_{selected_wind_dir_9am}' in processed_input.columns:
    processed_input[f'WindDir9am_{selected_wind_dir_9am}'] = 1
if f'WindDir3pm_{selected_wind_dir_3pm}' in processed_input.columns:
    processed_input[f'WindDir3pm_{selected_wind_dir_3pm}'] = 1

# Lakukan penskalaan pada fitur numerik di processed_input
processed_input[numeric_cols_to_scale_for_app] = scaler.transform(processed_input[numeric_cols_to_scale_for_app])


# --- 5. Tombol Prediksi ---
if st.button("Prediksi Hujan Besok"):
    prediction = rf_model.predict(processed_input)
    prediction_proba = rf_model.predict_proba(processed_input)

    st.header("Hasil Prediksi")
    if prediction[0] == 1:
        st.write(f"Berdasarkan data yang dimasukkan, **BESOK KEMUNGKINAN AKAN HUJAN** 🌧️")
        st.write(f"Probabilitas hujan: **{prediction_proba[0, 1]*100:.2f}%**")
    else:
        st.write(f"Berdasarkan data yang dimasukkan, **BESOK KEMUNGKINAN TIDAK AKAN HUJAN** ☀️")
        st.write(f"Probabilitas tidak hujan: **{prediction_proba[0, 0]*100:.2f}%**")

    st.write("---")
    st.write("Catatan: Prediksi ini berdasarkan model yang dilatih dengan data historis.")