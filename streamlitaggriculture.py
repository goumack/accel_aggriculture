# app.py

import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import load_data, standardize_data
from prediction import predict_new_observation


st.image("photo.png", width=100)
# Titre de l'application
st.title("Aide a la prise de decision des cultures Agricoles par ACCEL TECH")

# Charger le modèle
model = joblib.load("best_model.pkl")

# Entrée de l'utilisateur
st.sidebar.header("Paramètres d'entrée")
N = st.sidebar.number_input("Nitrogen  (kg/ha)", min_value=0.0, max_value=200.0, value=67.68)
P = st.sidebar.number_input("Phosphorous  (kg/ha)", min_value=0.0, max_value=200.0, value=39.07)
K = st.sidebar.number_input("Potassium  (kg/ha)", min_value=0.0, max_value=200.0, value=36.99)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=26.82)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=79.63)
ph = st.sidebar.number_input("pH du sol", min_value=0.0, max_value=14.0, value=6.387)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=105.27)

# Créer un bouton pour prédire
if st.sidebar.button("Prédire"):
    # Préparer l'observation
    observation = [N, P, K, temperature, humidity, ph, rainfall]
    
    # Standardiser l'observation
    train_mean = pd.Series([67.68, 39.07, 36.99, 26.82, 79.63, 6.38, 105.27])  # Moyenne d'entraînement
    train_std = pd.Series([34.81, 23.10, 15.28, 5.62, 14.69, 0.59, 64.57])  # Écart type d'entraînement
    y_pred = predict_new_observation(model, observation, train_mean, train_std)
    
    # Afficher le résultat
    st.write(f"La culture prédite est : {y_pred[0]}")

    st.image("banner.png", width=200)


