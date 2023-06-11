import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd


load_dt_model = joblib.load('data_resources/dt_model.pkl')
load_knn_model = joblib.load('data_resources/knn_model.pkl')
load_nb_model = joblib.load('data_resources/nb_model.pkl')
load_perceptron_model = joblib.load('data_resources/perceptron_model.pkl')
load_svm_model = joblib.load('data_resources/svm_model.pkl')


def dt_model(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    prediction = load_dt_model.predict(data)
    return prediction


def knn_model(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    prediction = load_knn_model.predict(data)
    return prediction


def nb_model(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    prediction = load_nb_model.predict(data)
    return prediction


def perceptron_model(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    prediction = load_perceptron_model.predict(data)
    return prediction


def svm_model(kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db):
    data = [[kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db]]
    prediction = load_svm_model.predict(data)
    return prediction


selected = option_menu(
    menu_title="Klasifikasi Data Garam Menggunakan Metode Perceptron",
    options=["Dataset", "Prediksi Data Baru"],
    # icons=["data", "Process", "model", "implemen", "Test"],
    orientation="horizontal",
)

if selected == 'Dataset':
    df = pd.read_csv("data_resources/data_garam.csv")
    df = df.drop(["Data"], axis=1)
    df
else:
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

    knn_tabs, dt_tabs, nb_tabs, pt_tabs, svm_tabs = st.tabs(
        ['K-Nerest Neighbor', 'Decission Tree', 'Naive Bayes', 'Perceptron', 'SVM'])

    with knn_tabs:
        knn_prediction = dt_model(
            kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)
        st.write('Hasil Prediksi:', knn_prediction[0])
    with dt_tabs:
        dt_prediction = dt_model(
            kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)
        st.write('Hasil Prediksi:', dt_prediction[0])
    with nb_tabs:
        nb_prediction = dt_model(
            kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)
        st.write('Hasil Prediksi:', nb_prediction[0])
    with pt_tabs:
        pt_prediction = dt_model(
            kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)
        st.write('Hasil Prediksi:', pt_prediction[0])
    with svm_tabs:
        svm_prediction = dt_model(
            kadar_air, tak_larut, kalsium, magnesium, sulfat, nacl_wb, nacl_db)
        st.write('Hasil Prediksi:', svm_prediction[0])
