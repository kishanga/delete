import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('app.sav')

st.header('Property Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Age of property at transaction')
    age = st.slider('Age', 5, 90, 1)
with col2:
    st.text('Distance from Ion Orchard in metre(hint: use Google Maps)')
    distance_from_town = st.slider('Distance from town', 1400, 3500, 1)

if st.button('Predict PSQM of Property'):
    # Create a DataFrame with the input features
    input_df = pd.DataFrame({
        'age': [age],
        'distance_from_town': [distance_from_town]
    })

    # Make the prediction
    psqm_prediction = model.predict(input_df)[0]

    # Show the prediction
    st.write(f"Predicted PSQM: {psqm_prediction:.2f}")
