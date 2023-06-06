import streamlit as st
import joblib


modelPerceptron = joblib.load('model/modelPerceptron.pkl')
modelPerceptron_best = joblib.load('model/modelPerceptron_best.pkl')
scaler = joblib.load('model/scaler.pkl')

# Fungsi untuk model 1


def model1(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    # Masukkan logika atau pemrosesan yang sesuai dengan model 1
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    x = scaler.fit_transform(data)
    prediction = modelPerceptron.predict(x)
    return prediction

# Fungsi untuk model 2


def model2_best(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    # Masukkan logika atau pemrosesan yang sesuai dengan model 2
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    x = scaler.fit_transform(data)
    prediction = modelPerceptron.predict(x)
    return prediction


# Judul aplikasi
st.title("Klasifikasi Data Garam Menggunakan Metode Perceptron")

# Input Kadar Air
kadar_air = st.number_input('Kadar Air')

# Input Tak Larut
tak_larut = st.number_input('Tak Larut')

# Input Kalsium
kalsium = st.number_input('Kalsium')

# Input Magnesium
magnesium = st.number_input('Magnesium')

# Input Sulfat
sulfat = st.number_input('Sulfat')

# Input NaCl (wb)
nacl_wb = st.number_input('NaCl (wb)')

# Input NaCl (db)
nacl_db = st.number_input('NaCl (db)')

# Pilihan model
model_choice = st.sidebar.selectbox(
    'Pilih Model', ('Model 1 (Perceptron)', 'Model 2 (With GridSearch)'))

# Prediksi berdasarkan model yang dipilih
if model_choice == 'Model 1 (Perceptron)':
    prediction = model1(
        kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)
else:
    prediction = model2_best(
        kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)

# Tampilkan hasil prediksi
st.write('Hasil Prediksi:', prediction)
