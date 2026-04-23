import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

st.write("Enter patient details:")

# Input fields
preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Prediction
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
